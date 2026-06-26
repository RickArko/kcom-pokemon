"""Run the exp002 heuristic agent against the 4 official sample decks.

The four sample decks (Mega Lucario ex, Dragapult ex, Iono's Deck, Mega
Abomasnow ex) are defined in code below because ``*.csv`` is gitignored.  Each
sample deck is played by a neutral :class:`RuleBasedAgent` (first-valid picker)
so the gauntlet measures the exp002 agent + Lucario deck vs each sample deck.

Usage:
    uv run python scripts/run_sample_deck_gauntlet.py                 # 20/matchup
    uv run python scripts/run_sample_deck_gauntlet.py --n-matches 50   # 50/matchup
    uv run python scripts/run_sample_deck_gauntlet.py --agent exp002_lucario_heuristic
    uv run python scripts/run_sample_deck_gauntlet.py --opponent random   # use RandomAgent
    uv run python scripts/run_sample_deck_gauntlet.py --list             # show decks only

Requires the ``cg`` engine: ``make sim-download``.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import logging
import sys
import time
from pathlib import Path

from pokemon.agent import Agent, RandomAgent, RuleBasedAgent
from pokemon.harness import run_match
from pokemon.sample_decks import SAMPLE_DECKS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

WORKSPACE = Path("workspace")
RESULTS_DIR = Path("workspace/results")

# SAMPLE_DECKS is imported from pokemon.sample_decks (shared with opponent.py).


def _check_decks() -> None:
    for name, deck in SAMPLE_DECKS.items():
        if len(deck) != 60:
            logger.error("Sample deck '%s' has %d cards (must be 60).", name, len(deck))
            raise SystemExit(1)


def _load_experiment_agent(exp_dir: Path, random_seed: int = 42) -> tuple[Agent, list[int]] | None:
    agent_path = exp_dir / "agent.py"
    deck_path = exp_dir / "deck.csv"
    if not agent_path.exists() or not deck_path.exists():
        logger.error("Missing agent.py or deck.csv in %s", exp_dir)
        return None
    from pokemon.deck import Deck

    deck = Deck.from_csv(str(deck_path)).cards
    try:
        mod_name = f"_exp_{exp_dir.name}"
        spec = importlib.util.spec_from_file_location(mod_name, str(agent_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to import %s: %s", agent_path, e)
        return None
    for _, cls in inspect.getmembers(mod, inspect.isclass):
        if issubclass(cls, Agent) and cls not in (Agent, RandomAgent, RuleBasedAgent):
            try:
                return cls(deck=deck, random_seed=random_seed), deck
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to instantiate %s: %s", cls.__name__, e)
                return None
    logger.error("No Agent subclass found in %s", agent_path)
    return None


def _build_opponent(kind: str, deck: list[int], seed: int) -> Agent:
    if kind == "random":
        return RandomAgent(deck=deck, random_seed=seed)
    return RuleBasedAgent(deck=deck, random_seed=seed)


def _run_matchup(
    agent: Agent,
    agent_name: str,
    opp: Agent,
    opp_name: str,
    n_matches: int,
    max_turns: int,
) -> dict:
    """Play `n_matches` games each side; return per-matchup stats for the agent."""
    wins = 0
    losses = 0
    errors = 0
    turns_total = 0
    t0 = time.time()
    # agent as player 0, then swapped (agent as player 1).
    for m in range(n_matches):
        result = run_match(agent, opp, max_turns=max_turns)
        turns_total += result["turns"]
        if result["winner"] == 0:
            wins += 1
        elif result["winner"] == 1:
            losses += 1
        else:
            errors += 1
        # swap sides
        result = run_match(opp, agent, max_turns=max_turns)
        turns_total += result["turns"]
        if result["winner"] == 1:
            wins += 1
        elif result["winner"] == 0:
            losses += 1
        else:
            errors += 1
    total = wins + losses + errors
    elapsed = time.time() - t0
    wr = wins / total if total else 0.0
    return {
        "agent": agent_name,
        "opponent": opp_name,
        "wins": wins,
        "losses": losses,
        "errors": errors,
        "matches": total,
        "win_rate": round(wr, 4),
        "avg_turns": round(turns_total / total, 1) if total else 0,
        "elapsed_seconds": round(elapsed, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exp002 against the 4 sample decks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--agent", default="exp002_lucario_heuristic", help="Experiment dir name.")
    parser.add_argument("--n-matches", type=int, default=20, help="Matches per side per matchup.")
    parser.add_argument("--max-turns", type=int, default=100, help="Max turns per match.")
    parser.add_argument(
        "--opponent",
        choices=["rule", "random"],
        default="rule",
        help="Baseline agent that plays the sample decks.",
    )
    parser.add_argument("--list", action="store_true", help="List sample decks and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _check_decks()

    if args.list:
        print(f"{'Sample deck':<20} {'cards':<7} {'basics (sample)'}")
        print("-" * 50)
        for name, deck in SAMPLE_DECKS.items():
            print(f"{name:<20} {len(deck):<7}")
        return

    try:
        import cg.game  # noqa: F401
    except ImportError:
        logger.error("Game engine (cg) not found. Run: make sim-download")
        sys.exit(1)

    exp_dir = WORKSPACE / args.agent
    if not exp_dir.is_dir():
        logger.error("Experiment not found: %s", exp_dir)
        sys.exit(1)

    loaded = _load_experiment_agent(exp_dir)
    if loaded is None:
        sys.exit(1)
    agent, agent_deck = loaded
    logger.info("Loaded agent: %s (deck %d cards)", args.agent, len(agent_deck))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    print(f"\n{'Matchup':<42} {'WR':<7} {'W-L-Err':<11} {'avgTurns':<9} {'time'}")
    print("-" * 80)
    for name, deck in SAMPLE_DECKS.items():
        opp = _build_opponent(args.opponent, deck, seed=hash(name) % 1000)
        stat = _run_matchup(agent, args.agent, opp, name, args.n_matches, args.max_turns)
        results.append(stat)
        wle = f"{stat['wins']}-{stat['losses']}-{stat['errors']}"
        print(
            f"{args.agent + ' vs ' + name:<42} {stat['win_rate']:<7.3f} "
            f"{wle:<11} {stat['avg_turns']:<9} {stat['elapsed_seconds']}s"
        )

    total_wins = sum(r["wins"] for r in results)
    total_matches = sum(r["matches"] for r in results)
    overall = total_wins / total_matches if total_matches else 0.0
    print("-" * 80)
    print(f"{'OVERALL':<42} {overall:<7.3f} {total_wins}/{total_matches}")

    out = {
        "agent": args.agent,
        "opponent": args.opponent,
        "n_matches_per_side": args.n_matches,
        "overall_win_rate": round(overall, 4),
        "matchups": results,
    }
    out_path = RESULTS_DIR / "sample_deck_gauntlet.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
