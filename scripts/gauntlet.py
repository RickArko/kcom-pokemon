"""Run agent gauntlet tournaments.

Usage:
    uv run python scripts/gauntlet.py
    uv run python scripts/gauntlet.py --n-matches 50
"""

from __future__ import annotations

import argparse
import logging
import sys

from pokemon.agent import RandomAgent
from pokemon.harness import run_gauntlet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent gauntlet tournaments.")
    parser.add_argument(
        "--n-matches",
        type=int,
        default=20,
        help="Matches per pairing (will be doubled for both sides)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=100,
        help="Max turns per match",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="workspace/results",
        help="Directory to save results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from cg.game import Game  # noqa: F401
    except ImportError:
        logger.error(
            "Game engine (cg) not found.  Download the Simulation SDK:\n"
            "  make sim-download\n"
            "Then place the cg/ package at data/sim_sample/cg/"
        )
        sys.exit(1)

    agents = [
        ("random_a", RandomAgent(random_seed=1)),
        ("random_b", RandomAgent(random_seed=2)),
    ]

    logger.info("Running gauntlet: %d matches per pairing", args.n_matches)
    run_gauntlet(
        agents=agents,
        n_matches=args.n_matches,
        max_turns=args.max_turns,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
