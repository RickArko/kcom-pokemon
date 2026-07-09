"""Deck gap analysis: compare a candidate deck against the real Kaggle meta.

Reads aggregated Kaggle episode data, computes card-frequency distributions of
the meta (and optionally of winning decks only), then compares a candidate
deck against that distribution.  Produces a markdown report with:

  - Current deck composition (pokemon / supporter / item / tool / stadium / energy)
  - Deck-building rule validation (60 cards, <=4 copies by name, <=1 ACE SPEC)
  - "Cards we run but meta rarely does" (potential cuts)
  - "Cards meta runs but we don't" (potential additions, ranked by meta frequency)
  - A per-archetype matchup reference (our deck's observed WR if it appears in
    the meta data)

This is the data-driven replacement for the hand-written gap table in
``.ai/plans/improvements.md``: instead of assuming the meta is the 4 sample
decks, it measures the *actual* Kaggle meta from downloaded episodes.

Usage:
    uv run python scripts/deck_gap.py --deck workspace/exp009_deck_tuned/deck.csv
    uv run python scripts/deck_gap.py --deck workspace/exp008_full_mcts/deck.csv --winners-only
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

from pokemon.card_db import CardDB, get_card_db, validate_deck  # noqa: E402


def _deck_composition(deck: list[int], db: CardDB) -> dict[str, int]:
    counts = {"pokemon": 0, "supporter": 0, "item": 0, "tool": 0, "stadium": 0, "energy": 0}
    for cid in deck:
        info = db.get(cid)
        if info is None:
            continue
        if info.is_pokemon:
            counts["pokemon"] += 1
        elif info.card_type_name in counts:
            counts[info.card_type_name] += 1
        elif info.is_energy:
            counts["energy"] += 1
    return counts


def _meta_card_frequency(
    results: pd.DataFrame, source: str | None, winners_only: bool
) -> tuple[Counter, Counter, int]:
    """Return (copy_counter, deck_counter, total_decks) over the meta."""
    df = results.copy()
    if source and "source" in df:
        df = df[df["source"] == source]

    copy_counter: Counter[int] = Counter()
    deck_counter: Counter[int] = Counter()
    total_decks = 0

    for side, deck_col, arch_col in ((0, "deck0", "arch0"), (1, "deck1", "arch1")):
        for _, row in df.iterrows():
            if winners_only and row.get("winner", -1) != side:
                continue
            deck = row.get(deck_col)
            if not hasattr(deck, "__iter__") or len(deck) != 60:
                continue
            total_decks += 1
            seen: set[int] = set()
            for cid in deck:
                cid = int(cid)
                copy_counter[cid] += 1
                if cid not in seen:
                    deck_counter[cid] += 1
                    seen.add(cid)
    return copy_counter, deck_counter, total_decks


def _format_card_row(cid: int, db: CardDB, meta_pct: float, our_count: int) -> str:
    info = db.get(cid)
    name = info.name[:28] if info else f"Card#{cid}"
    ctype = info.card_type_name[:10] if info else "?"
    ace = " **ACE SPEC**" if (info and info.is_ace_spec) else ""
    ours = str(our_count) if our_count > 0 else "—"
    return f"| {cid} | {name} | {ctype} | {meta_pct:.1f}% | {ours} |{ace}"


def generate_gap_report(
    deck: list[int],
    results: pd.DataFrame,
    db: CardDB,
    source: str | None = "kaggle",
    winners_only: bool = False,
    top_n: int = 25,
) -> str:
    """Build the markdown gap report for ``deck`` against the meta."""
    lines: list[str] = []
    label = "winning decks" if winners_only else "all decks"
    lines.append("# Deck Gap Analysis")
    lines.append(f"(meta source: {source or 'all'}, {label})\n")

    # --- our deck composition ---
    comp = _deck_composition(deck, db)
    ok, errs = validate_deck(deck, db)
    lines.append("## Candidate Deck")
    lines.append("")
    lines.append("| Pokemon | Supporter | Item | Tool | Stadium | Energy | Valid |")
    lines.append("|--------:|----------:|-----:|----:|--------:|--------:|:-----:|")
    lines.append(
        f"| {comp['pokemon']} | {comp['supporter']} | {comp['item']} | {comp['tool']} "
        f"| {comp['stadium']} | {comp['energy']} | {'yes' if ok else 'NO'} |"
    )
    if not ok:
        lines.append("")
        lines.append("**VALIDATION ERRORS:**")
        for e in errs:
            lines.append(f"- {e}")
    lines.append("")

    our_counts = Counter(deck)

    # --- meta frequency ---
    copies, decks, total = _meta_card_frequency(results, source, winners_only)
    if total == 0:
        lines.append("No meta decks found in the data.\n")
        return "\n".join(lines)
    lines.append(f"**Meta sample:** {total} decks\n")

    # --- cards we run but meta rarely does (potential cuts) ---
    lines.append("## Cards We Run But Meta Rarely Does (potential cuts)")
    lines.append("")
    lines.append("| Card ID | Name | Type | Our copies | Meta % decks | Avg copies/deck |")
    lines.append("|--------:|------|:----:|-----------:|-------------:|----------------:|")
    rare = []
    for cid in sorted(our_counts, key=lambda c: (decks.get(c, 0), c)):
        meta_pct = decks.get(cid, 0) / total * 100
        if meta_pct < 15 and not (db.get(cid) and db.get(cid).is_basic_energy):
            rare.append((cid, meta_pct))
    for cid, meta_pct in sorted(rare, key=lambda x: x[1])[:top_n]:
        info = db.get(cid)
        name = info.name[:28] if info else f"Card#{cid}"
        ctype = info.card_type_name[:10] if info else "?"
        avg = copies.get(cid, 0) / max(decks.get(cid, 1), 1)
        lines.append(
            f"| {cid} | {name} | {ctype} | {our_counts[cid]} | {meta_pct:.1f}% | {avg:.1f} |"
        )
    lines.append("")

    # --- cards meta runs but we don't (potential adds) ---
    lines.append("## Cards Meta Runs But We Don't (potential adds, ranked)")
    lines.append("")
    lines.append("| Card ID | Name | Type | Meta % decks | Our copies | Note |")
    lines.append("|--------:|------|:----:|-------------:|-----------:|------|")
    adds = []
    for cid in decks:
        if cid in our_counts:
            continue
        meta_pct = decks[cid] / total * 100
        if meta_pct >= 20:
            adds.append((cid, meta_pct))
    for cid, meta_pct in sorted(adds, key=lambda x: -x[1])[:top_n]:
        info = db.get(cid)
        name = info.name[:28] if info else f"Card#{cid}"
        ctype = info.card_type_name[:10] if info else "?"
        ace = "ACE SPEC (max 1)" if (info and info.is_ace_spec) else ""
        lines.append(f"| {cid} | {name} | {ctype} | {meta_pct:.1f}% | 0 | {ace} |")
    lines.append("")

    # --- under-represented staples (we run some but less than meta) ---
    lines.append("## Under-represented Staples (we run fewer than meta average)")
    lines.append("")
    lines.append("| Card ID | Name | Our copies | Meta avg copies | Meta % decks |")
    lines.append("|--------:|------|-----------:|----------------:|-------------:|")
    for cid in sorted(our_counts):
        if decks.get(cid, 0) == 0:
            continue
        meta_pct = decks[cid] / total * 100
        if meta_pct < 20:
            continue
        avg = copies[cid] / decks[cid]
        if our_counts[cid] < avg - 0.5:
            info = db.get(cid)
            name = info.name[:28] if info else f"Card#{cid}"
            lines.append(f"| {cid} | {name} | {our_counts[cid]} | {avg:.1f} | {meta_pct:.1f}% |")
    lines.append("")

    # --- observed meta WR for our deck archetype (if present) ---
    lines.append("## Observed Meta Win Rate for this Deck's Archetype")
    lines.append("")
    our_arch_wr = _our_archetype_wr(deck, results, db, source)
    if our_arch_wr:
        lines.append("| Archetype | Games | Wins | Win Rate |")
        lines.append("|-----------|------:|-----:|---------:|")
        for arch, stats in our_arch_wr.items():
            lines.append(f"| {arch} | {stats['games']} | {stats['wins']} | {stats['wr']:.1%} |")
    else:
        lines.append(
            "_Deck archetype not observed in meta data (or no archetype labels available)._"
        )
    lines.append("")
    lines.append("---")
    lines.append("*Generated by scripts/deck_gap.py*")
    return "\n".join(lines)


def _our_archetype_wr(
    deck: list[int], results: pd.DataFrame, db: CardDB, source: str | None
) -> dict[str, dict]:
    """If our deck's signature Pokemon match a labeled archetype, report its WR."""
    df = results.copy()
    if source and "source" in df:
        df = df[df["source"] == source]
    if df.empty or "arch0" not in df:
        return {}
    our_key_pokemon = {cid for cid in deck if db.get(cid) and db.get(cid).is_mega_ex}
    if not our_key_pokemon:
        return {}
    rows: dict[str, dict] = {}
    for side, arch_col, deck_col in ((0, "arch0", "deck0"), (1, "arch1", "deck1")):
        for _, row in df.iterrows():
            arch = row.get(arch_col)
            if not arch:
                continue
            deck_list = row.get(deck_col)
            if not hasattr(deck_list, "__iter__"):
                continue
            if our_key_pokemon & set(int(x) for x in deck_list):
                d = rows.setdefault(arch, {"games": 0, "wins": 0})
                d["games"] += 1
                if row.get("winner") == side:
                    d["wins"] += 1
    return {
        a: {"games": v["games"], "wins": v["wins"], "wr": v["wins"] / v["games"]}
        for a, v in rows.items()
        if v["games"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deck gap analysis vs real Kaggle meta.")
    parser.add_argument("--deck", required=True, help="Path to candidate deck.csv")
    parser.add_argument("--input", default="data/matches/aggregated", help="Parquet tables dir")
    parser.add_argument("--source", choices=["kaggle", "local", "all"], default="kaggle")
    parser.add_argument("--winners-only", action="store_true", help="Use winning decks only")
    parser.add_argument("--output", default="", help="Write report to file (default: stdout)")
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()

    deck = [int(line) for line in Path(args.deck).read_text().split("\n") if line.strip()]
    results = pd.read_parquet(Path(args.input) / "results.parquet")
    db = get_card_db(use_engine=True)
    source = None if args.source == "all" else args.source

    report = generate_gap_report(
        deck, results, db, source=source, winners_only=args.winners_only, top_n=args.top_n
    )
    if args.output:
        Path(args.output).write_text(report)
        logger.info("Report written to %s", args.output)
    else:
        print(report)


if __name__ == "__main__":
    main()
