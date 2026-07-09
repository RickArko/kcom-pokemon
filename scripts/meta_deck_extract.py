"""Extract canonical meta decklists from aggregated Kaggle episode data.

For each archetype observed in the Kaggle Simulation Arena, find the most common
*exact* 60-card decklist (the mode), validate it against the engine's
deck-building rules, and write it to ``data/meta_decks/<archetype>.csv`` plus a
JSON manifest.  These decks become local "meta proxy" opponents so we can gauge
our agent's real-meta performance without uploading to Kaggle.

Usage:
    uv run python scripts/meta_deck_extract.py
    uv run python scripts/meta_deck_extract.py --input data/matches/aggregated
    uv run python scripts/meta_deck_extract.py --min-count 10 --source kaggle
"""

from __future__ import annotations

import argparse
import json
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

DATA_DIR = Path("data/meta_decks")


def _deck_key(deck) -> tuple[int, ...]:
    """Normalize a parquet deck array into a sortable, hashable tuple."""
    return tuple(sorted(int(x) for x in deck))


def extract_meta_decks(
    results: pd.DataFrame,
    db: CardDB,
    source: str | None = "kaggle",
    min_count: int = 5,
    top_k: int = 1,
) -> dict[str, dict]:
    """Return ``{archetype: {deck, count, valid, errors}}`` for the mode decklist.

    Only decks that are exactly 60 cards and pass engine validation are kept.
    If the single most common decklist is invalid, the next most common valid
    one is used (so a noisy/aggregated archetype still yields a legal deck).
    """
    df = results.copy()
    if source and "source" in df:
        df = df[df["source"] == source]
    if df.empty:
        return {}

    # Accumulate exact decklists per archetype (both sides).
    per_arch: dict[str, Counter] = {}
    for col, arch_col in (("deck0", "arch0"), ("deck1", "arch1")):
        for _, row in df.iterrows():
            arch = row.get(arch_col)
            deck = row.get(col)
            if not arch or not hasattr(deck, "__iter__") or len(deck) != 60:
                continue
            per_arch.setdefault(arch, Counter())[_deck_key(deck)] += 1

    out: dict[str, dict] = {}
    for arch, counter in per_arch.items():
        chosen = None
        for deck_tuple, count in counter.most_common(top_k + 5):
            if count < min_count:
                break
            deck = list(deck_tuple)
            ok, errs = validate_deck(deck, db)
            if ok:
                chosen = {"deck": deck, "count": count, "valid": True, "errors": []}
                break
            logger.debug("  %s mode deck (n=%d) invalid: %s", arch, count, errs)
        if chosen is None:
            # Fall back to the most common decklist regardless of validity, but
            # flag it so callers know not to use it as a real opponent.
            deck_tuple, count = counter.most_common(1)[0]
            deck = list(deck_tuple)
            ok, errs = validate_deck(deck, db)
            chosen = {"deck": deck, "count": count, "valid": ok, "errors": errs}
        out[arch] = chosen
    return out


def _deck_summary(deck: list[int], db: CardDB) -> dict:
    """Composition stats for a deck (pokemon/supporter/item/tool/stadium/energy)."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract canonical meta decklists.")
    parser.add_argument("--input", default="data/matches/aggregated", help="Parquet tables dir")
    parser.add_argument("--output", default=str(DATA_DIR), help="Output dir for deck CSVs")
    parser.add_argument("--source", choices=["kaggle", "local", "all"], default="kaggle")
    parser.add_argument("--min-count", type=int, default=5, help="Min occurrences for a decklist")
    parser.add_argument("--top-k", type=int, default=1, help="Write top-K variants per archetype")
    args = parser.parse_args()

    results_path = Path(args.input) / "results.parquet"
    if not results_path.exists():
        logger.error("No results.parquet at %s — run 'make kaggle-all' first.", results_path)
        raise SystemExit(1)

    logger.info("Loading %s ...", results_path)
    results = pd.read_parquet(results_path)
    db = get_card_db(use_engine=True)
    source = None if args.source == "all" else args.source

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = extract_meta_decks(
        results, db, source=source, min_count=args.min_count, top_k=args.top_k
    )
    if not meta:
        logger.warning("No 60-card decks found (source=%s).", args.source)
        return

    manifest: dict[str, dict] = {}
    header = (
        f"\n{'Archetype':<22} {'N':>5} {'Valid':<6} "
        f"{'Pkmn':>5} {'Supp':>5} {'Item':>5} {'Tool':>5} {'Stad':>5} {'Nrgy':>5}"
    )
    print(header)
    print("-" * 75)
    for arch in sorted(meta, key=lambda a: meta[a]["count"], reverse=True):
        info = meta[arch]
        deck = info["deck"]
        comp = _deck_summary(deck, db)
        valid = "yes" if info["valid"] else "NO"
        print(
            f"{arch:<22} {info['count']:>5} {valid:<6} {comp['pokemon']:>5} "
            f"{comp['supporter']:>5} {comp['item']:>5} {comp['tool']:>5} "
            f"{comp['stadium']:>5} {comp['energy']:>5}"
        )
        # Write CSV (one card ID per line, sorted for readability).
        csv_path = out_dir / f"{arch}.csv"
        csv_path.write_text("\n".join(str(c) for c in sorted(deck)) + "\n")
        manifest[arch] = {
            "count": info["count"],
            "valid": info["valid"],
            "errors": info["errors"],
            "composition": comp,
            "csv": str(csv_path),
        }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote %d meta decks to %s (manifest: %s)", len(meta), out_dir, manifest_path)


if __name__ == "__main__":
    main()
