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
import tarfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

_REQUIRED_FILES = ["main.py", "deck.csv"]


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
