"""Package an agent for Kaggle Simulation submission.

Usage:
    uv run python scripts/build_submission.py \\
        --agent workspace/exp001/agent.py \\
        --deck workspace/exp001/deck.csv \\
        --out submit/
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tarfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

_REQUIRED_FILES = ["main.py", "deck.csv"]


def _load_deck(path: Path) -> list[int]:
    return [int(line) for line in path.read_text().split("\n") if line.strip()][:60]


def _validate_deck_or_exit(deck_path: Path, cg_dir: Path) -> list[int]:
    """Load and validate the deck against the engine's deck-building rules.

    Aborts the build (exit 1) on any violation so we never upload a deck the
    Kaggle engine will reject at ``battle_start`` (e.g. >1 ACE SPEC, >4 copies
    by name, or !=60 cards).  Prefers the engine card data (which encodes the
    ACE SPEC flag correctly) and falls back to the CSV source (which does not,
    so ACE SPEC violations may be missed offline).
    """
    deck = _load_deck(deck_path)
    # Make the bundled cg engine importable so ACE SPEC flags resolve correctly.
    if cg_dir.exists():
        sys.path.insert(0, str(cg_dir.parent if cg_dir.name == "cg" else cg_dir))
    from pokemon.card_db import CardDB, validate_deck

    db = CardDB.load(use_engine=True)
    ok, errors = validate_deck(deck, db)
    if not ok:
        logger.error("Deck validation FAILED for %s:", deck_path)
        for e in errors:
            logger.error("  - %s", e)
        logger.error("Fix the deck before submitting — the Kaggle engine will reject it.")
        raise SystemExit(1)
    logger.info("Deck validated: 60 cards, <=4 copies by name, <=1 ACE SPEC.")
    return deck


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package agent for Kaggle submission.")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="Path to the agent's main.py",
    )
    parser.add_argument(
        "--deck",
        type=str,
        required=True,
        help="Path to the agent's deck.csv",
    )
    parser.add_argument(
        "--cg-dir",
        type=str,
        default="data/sim_sample/cg",
        help="Path to the cg engine package",
    )
    parser.add_argument(
        "--pkg-dir",
        type=str,
        default="src/pokemon",
        help="Path to the pokemon helper package to bundle (so `import pokemon` works)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="submit/",
        help="Output directory for the submission tarball",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent_path = Path(args.agent)
    deck_path = Path(args.deck)
    cg_dir = Path(args.cg_dir)
    pkg_dir = Path(args.pkg_dir)
    out_dir = Path(args.out)

    for path, name in [
        (agent_path, "agent"),
        (deck_path, "deck"),
        (cg_dir, "cg engine"),
        (pkg_dir, "pokemon package"),
    ]:
        if not path.exists():
            logger.error("%s not found at %s", name, path)
            raise SystemExit(1)

    # Validate the deck *before* packaging so invalid decks never get uploaded.
    _validate_deck_or_exit(deck_path, cg_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = out_dir / "bundle"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir()

    shutil.copy(agent_path, bundle_dir / "main.py")
    shutil.copy(deck_path, bundle_dir / "deck.csv")
    shutil.copytree(cg_dir, bundle_dir / "cg", dirs_exist_ok=True)
    # Bundle the pokemon helper package so `from pokemon...` imports resolve on
    # Kaggle (the simulation env only ships the cg engine + the agent tarball).
    shutil.copytree(pkg_dir, bundle_dir / "pokemon", dirs_exist_ok=True)
    # Strip pycache from the bundled package to keep the tarball lean.
    for pycache in bundle_dir.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)

    tarball = out_dir / "submission.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        for f in bundle_dir.iterdir():
            tar.add(f, arcname=f.name)

    shutil.rmtree(bundle_dir)
    logger.info("Submission tarball created: %s", tarball)
    logger.info("Upload at: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/submit")


if __name__ == "__main__":
    main()
