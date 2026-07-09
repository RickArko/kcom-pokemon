"""Verify a candidate deck/agent against a baseline and the real meta.

This is the regression-and-improvement gate for the deck-optimization plan.
It runs three kinds of matches:

  1. **Head-to-head** — candidate vs baseline (both sides swapped), to see if
     the candidate directly beats its predecessor.
  2. **Meta gauntlet** — candidate and baseline each vs the meta proxy decks
     (extracted from Kaggle episode data), to measure real-meta performance
     and detect overfitting to the mirror.
  3. **Sanity** — candidate vs random, to confirm it still crushes the
     baseline-of-baselines (a regression here means the agent broke).

It then computes Wilson 95% CIs and a pass/fail :class:`pokemon.stats.Verdict`:
the candidate must beat the baseline head-to-head (>= ``--min-h2h-wr``) AND not
regress vs the meta field by more than ``--regression-margin``.  Results are
saved to ``workspace/results/verify_<candidate>.json`` and a markdown report.

Usage:
    make verify-deck ARGS="--candidate exp009_deck_tuned --baseline exp008_full_mcts"
    uv run python scripts/verify_deck.py --candidate exp009_deck_tuned \
        --baseline exp008_full_mcts --n-matches 30
    uv run python scripts/verify_deck.py --candidate exp009_deck_tuned --fast
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

WORKSPACE = Path("workspace")

from pokemon.agent import Agent, RandomAgent  # noqa: E402
from pokemon.deck import Deck  # noqa: E402
from pokemon.harness import run_match  # noqa: E402
from pokemon.meta_decks import make_meta_proxies  # noqa: E402
from pokemon.stats import (  # noqa: E402
    GauntletStats,
    MatchResult,
    Verdict,
    decide,
)


def _load_experiment(name: str, random_seed: int = 42) -> Agent | None:
    """Load an agent from a workspace experiment directory (agent.py + deck.csv)."""
    path = Path(name)
    exp_dir = path if path.is_dir() else WORKSPACE / name
    agent_path = exp_dir / "agent.py"
    deck_path = exp_dir / "deck.csv"
    if not agent_path.exists() or not deck_path.exists():
        logger.error("Experiment %s missing agent.py or deck.csv", exp_dir)
        return None
    deck = Deck.from_csv(str(deck_path)).cards
    spec = importlib.util.spec_from_file_location(f"_verify_{exp_dir.name}", str(agent_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for _, cls in inspect.getmembers(mod, inspect.isclass):
        if issubclass(cls, Agent) and cls not in (Agent, RandomAgent):
            try:
                return cls(deck=deck, random_seed=random_seed)
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to instantiate %s: %s", cls.__name__, e)
                return None
    logger.error("No Agent subclass found in %s", agent_path)
    return None


def _run_pairing(
    agent_a, agent_b, n_matches: int, max_turns: int, label: str
) -> tuple[int, int, int]:
    """Run ``n_matches`` each side. Returns (a_wins, b_wins, draws) from A's view."""
    a_wins = b_wins = draws = 0
    for side in range(2):
        for m in range(n_matches):
            if side == 0:
                res = run_match(agent_a, agent_b, max_turns=max_turns)
                w = res["winner"]
            else:
                res = run_match(agent_b, agent_a, max_turns=max_turns)
                w = 1 - res["winner"] if res["winner"] in (0, 1) else -1
            if w == 0:
                a_wins += 1
            elif w == 1:
                b_wins += 1
            else:
                draws += 1
            tag = "err" if res["error"] else "ok"
            logger.info(
                "    [%s %d/%d] %s: A-%d B-%d D-%d (%s)",
                label,
                side * n_matches + m + 1,
                n_matches * 2,
                label,
                a_wins,
                b_wins,
                draws,
                tag,
            )
    return a_wins, b_wins, draws


def _format_pct(x: float) -> str:
    return f"{x:.1%}"


def _verdict_markdown(verdict: Verdict, candidate: GauntletStats, baseline: GauntletStats) -> str:
    lines: list[str] = []
    lines.append("# Deck Verification Report\n")
    status = (
        "PASS — candidate is an improvement"
        if verdict.passed
        else "FAIL — no improvement / regression"
    )
    lines.append(f"**Verdict: {status}**\n")
    lines.append(f"Reason: {verdict.reason}\n")
    lines.append("## Overall Win Rates (vs meta + random field)\n")
    lines.append("| Agent | Wins | Losses | Draws | WR | 95% CI | MoE |")
    lines.append("|-------|-----:|-------:|------:|---:|:------:|----:|")
    for name, st in [(candidate.name, candidate), (baseline.name, baseline)]:
        lo, hi = st.ci_low(), st.ci_high()
        lines.append(
            f"| {name} | {st.wins} | {st.losses} | {st.draws} | {_format_pct(st.win_rate)} "
            f"| [{_format_pct(lo)}, {_format_pct(hi)}] | ±{_format_pct(st.margin_of_error())} |"
        )
    lines.append("")
    lines.append("## Head-to-Head (candidate vs baseline)\n")
    lo, hi = verdict.head_to_head_ci
    lines.append(
        f"- Candidate WR: **{_format_pct(verdict.head_to_head_wr)}** "
        f"(95% CI [{_format_pct(lo)}, {_format_pct(hi)}], n={verdict.head_to_head_n})"
    )
    lines.append("")
    lines.append("## Per-Opponent Breakdown\n")
    names = sorted(set(candidate.per_opponent) | set(baseline.per_opponent))
    lines.append("| Opponent | Candidate WR | Baseline WR | Delta |")
    lines.append("|----------|-------------:|------------:|------:|")
    for opp in names:
        c = candidate.per_opponent.get(opp)
        b = baseline.per_opponent.get(opp)
        cwr = _format_pct(c.win_rate) if c else "—"
        bwr = _format_pct(b.win_rate) if b else "—"
        delta = _format_pct(c.win_rate - b.win_rate) if (c and b) else "—"
        lines.append(f"| {opp} | {cwr} | {bwr} | {delta} |")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by scripts/verify_deck.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a candidate deck vs baseline + meta.")
    parser.add_argument("--candidate", required=True, help="Candidate experiment name/dir")
    parser.add_argument("--baseline", required=True, help="Baseline experiment name/dir")
    parser.add_argument("--n-matches", type=int, default=20, help="Matches per side per pairing")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--results-dir", default="workspace/results")
    parser.add_argument("--no-meta", action="store_true", help="Skip meta proxy opponents")
    parser.add_argument("--no-random", action="store_true", help="Skip random sanity check")
    parser.add_argument("--min-h2h-wr", type=float, default=0.52, help="Required head-to-head WR")
    parser.add_argument("--regression-margin", type=float, default=0.03)
    parser.add_argument(
        "--no-meta-gate", action="store_true", help="Disable the meta-regression check"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Quick smoke test (3 matches, MCTS budget cut)"
    )
    args = parser.parse_args()

    if args.fast:
        os.environ["GAUNTLET_FAST"] = "1"
        args.n_matches = 3
        logger.info("Fast mode: n_matches=3, GAUNTLET_FAST=1")

    try:
        import cg.game  # noqa: F401
    except ImportError:
        logger.error("Game engine (cg) not found. Run: make sim-download")
        sys.exit(1)

    candidate = _load_experiment(args.candidate)
    baseline = _load_experiment(args.baseline)
    if candidate is None or baseline is None:
        sys.exit(1)

    # Opponent field.
    opponents: list[tuple[str, Agent]] = []
    if not args.no_meta:
        opponents.extend(make_meta_proxies(random_seed=7))
    if not args.no_random:
        sample_deck_path = Path("data/sim_sample/sample_submission/deck.csv")
        rand_deck = (
            [int(line.strip()) for line in sample_deck_path.read_text().split("\n") if line.strip()]
            if sample_deck_path.exists()
            else [1] * 60
        )
        opponents.append(("random", RandomAgent(deck=rand_deck, random_seed=1)))

    cand_stats = GauntletStats(name=args.candidate)
    base_stats = GauntletStats(name=args.baseline)
    h2h = MatchResult(name="head_to_head")

    t0 = time.time()
    # 1. Head-to-head (candidate vs baseline).
    logger.info("Head-to-head: %s vs %s (%d/side)", args.candidate, args.baseline, args.n_matches)
    a_w, b_w, d = _run_pairing(candidate, baseline, args.n_matches, args.max_turns, "h2h")
    h2h.wins = a_w
    h2h.losses = b_w
    h2h.draws = d
    cand_stats.wins += a_w
    base_stats.wins += b_w
    cand_stats.losses += b_w
    base_stats.losses += a_w
    cand_stats.draws += d
    base_stats.draws += d
    cand_stats.per_opponent[args.baseline] = MatchResult(args.baseline, a_w, b_w, d)
    base_stats.per_opponent[args.candidate] = MatchResult(args.candidate, b_w, a_w, d)
    logger.info("  h2h: %s %d-%d-%d %s", args.candidate, a_w, b_w, d, args.baseline)

    # 2 & 3. Each of candidate & baseline vs the opponent field.
    for opp_name, opp_agent in opponents:
        logger.info("%s vs %s (%d/side)", args.candidate, opp_name, args.n_matches)
        ca, cb, cd = _run_pairing(
            candidate, opp_agent, args.n_matches, args.max_turns, f"c-{opp_name}"
        )
        cand_stats.wins += ca
        cand_stats.losses += cb
        cand_stats.draws += cd
        cand_stats.per_opponent[opp_name] = MatchResult(opp_name, ca, cb, cd)

        logger.info("%s vs %s (%d/side)", args.baseline, opp_name, args.n_matches)
        ba, bb, bd = _run_pairing(
            baseline, opp_agent, args.n_matches, args.max_turns, f"b-{opp_name}"
        )
        base_stats.wins += ba
        base_stats.losses += bb
        base_stats.draws += bd
        base_stats.per_opponent[opp_name] = MatchResult(opp_name, ba, bb, bd)

    elapsed = time.time() - t0
    logger.info("All matches complete in %.1fs", elapsed)

    verdict: Verdict = decide(
        cand_stats,
        base_stats,
        h2h,
        min_head_to_head_wr=args.min_h2h_wr,
        regression_margin=args.regression_margin,
        require_meta_improvement=not args.no_meta_gate,
    )

    # Save JSON + markdown.
    results_path = Path(args.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    report = {
        "candidate": {
            "name": cand_stats.name,
            "wins": cand_stats.wins,
            "losses": cand_stats.losses,
            "draws": cand_stats.draws,
            "win_rate": round(cand_stats.win_rate, 4),
            "ci95": [round(cand_stats.ci_low(), 4), round(cand_stats.ci_high(), 4)],
            "per_opponent": {
                k: {
                    "wins": v.wins,
                    "losses": v.losses,
                    "draws": v.draws,
                    "wr": round(v.win_rate, 4),
                }
                for k, v in cand_stats.per_opponent.items()
            },
        },
        "baseline": {
            "name": base_stats.name,
            "wins": base_stats.wins,
            "losses": base_stats.losses,
            "draws": base_stats.draws,
            "win_rate": round(base_stats.win_rate, 4),
            "ci95": [round(base_stats.ci_low(), 4), round(base_stats.ci_high(), 4)],
            "per_opponent": {
                k: {
                    "wins": v.wins,
                    "losses": v.losses,
                    "draws": v.draws,
                    "wr": round(v.win_rate, 4),
                }
                for k, v in base_stats.per_opponent.items()
            },
        },
        "head_to_head": {
            "candidate_wins": h2h.wins,
            "baseline_wins": h2h.losses,
            "draws": h2h.draws,
            "candidate_wr": round(verdict.head_to_head_wr, 4),
            "ci95": [round(verdict.head_to_head_ci[0], 4), round(verdict.head_to_head_ci[1], 4)],
            "n": verdict.head_to_head_n,
        },
        "verdict": {
            "passed": verdict.passed,
            "reason": verdict.reason,
        },
        "config": {
            "n_matches_per_side": args.n_matches,
            "min_h2h_wr": args.min_h2h_wr,
            "regression_margin": args.regression_margin,
            "meta_gate": not args.no_meta_gate,
            "elapsed_seconds": round(elapsed, 1),
        },
    }
    json_path = results_path / f"verify_{args.candidate}.json"
    json_path.write_text(json.dumps(report, indent=2))
    md_path = results_path / f"verify_{args.candidate}.md"
    md_path.write_text(_verdict_markdown(verdict, cand_stats, base_stats))
    logger.info("Report: %s / %s", json_path, md_path)

    # Console summary.
    print()
    print(f"{'Agent':<28} {'WR':<7} {'95% CI':<22} {'W-L-D'}")
    print("-" * 70)
    for st in (cand_stats, base_stats):
        lo, hi = st.ci_low(), st.ci_high()
        print(
            f"{st.name:<28} {st.win_rate:<6.1%} "
            f"[{lo:>5.1%}, {hi:>5.1%}]   {st.wins}-{st.losses}-{st.draws}"
        )
    print()
    print(f"Head-to-head: {verdict.head_to_head_wr:.1%} ({h2h.wins}-{h2h.losses}-{h2h.draws})")
    status = "PASS" if verdict.passed else "FAIL"
    print(f"Verdict: {status} — {verdict.reason}")

    sys.exit(0 if verdict.passed else 2)


if __name__ == "__main__":
    main()
