"""Run and compare experiments.

Usage:
    uv run python scripts/gauntlet.py                      # auto-discover all
    uv run python scripts/gauntlet.py exp001 exp002         # specific experiments
    uv run python scripts/gauntlet.py --list               # list only
    uv run python scripts/gauntlet.py --n-matches 50       # custom matches
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import logging
import sys
from pathlib import Path

from pokemon.agent import Agent, RandomAgent, RuleBasedAgent
from pokemon.deck import Deck
from pokemon.harness import run_gauntlet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

WORKSPACE = Path("workspace")


def _resolve_experiments(names: list[str]) -> list[Path]:
    """Convert experiment names/patterns to directories."""
    if not names:
        dirs = sorted(WORKSPACE.glob("exp*/"))
        if not dirs:
            logger.warning("No experiments found in workspace/")
        return dirs

    dirs: list[Path] = []
    for name in names:
        path = Path(name)
        if path.is_dir():
            dirs.append(path)
        elif (WORKSPACE / name).is_dir():
            dirs.append(WORKSPACE / name)
        else:
            logger.warning("Experiment not found: %s", name)
    return dirs


def _load_agent(exp_dir: Path, random_seed: int = 42) -> Agent | None:
    agent_path = exp_dir / "agent.py"
    deck_path = exp_dir / "deck.csv"

    if not agent_path.exists():
        logger.warning("  no agent.py in %s", exp_dir)
        return None
    if not deck_path.exists():
        logger.warning("  no deck.csv in %s", exp_dir)
        return None

    try:
        deck = Deck.from_csv(str(deck_path)).cards
    except Exception as e:
        logger.warning("  failed to load deck from %s: %s", deck_path, e)
        return None

    try:
        mod_name = f"_exp_{exp_dir.name}"
        spec = importlib.util.spec_from_file_location(mod_name, str(agent_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        logger.warning("  failed to import %s: %s", agent_path, e)
        return None

    for _, cls in inspect.getmembers(mod, inspect.isclass):
        if issubclass(cls, Agent) and cls not in (Agent, RandomAgent, RuleBasedAgent):
            try:
                return cls(deck=deck, random_seed=random_seed)
            except Exception as e:
                logger.warning("  failed to instantiate %s: %s", cls.__name__, e)
                return None

    logger.warning("  no Agent subclass found in %s", agent_path)
    return None


def _list_experiments() -> None:
    dirs = sorted(WORKSPACE.glob("exp*/"))
    if not dirs:
        print("No experiments found in workspace/")
        return

    print(f"{'Experiment':<25} {'agent.py':<10} {'deck.csv':<10}")
    print("-" * 50)
    for d in dirs:
        has_agent = "✓" if (d / "agent.py").exists() else "—"
        has_deck = "✓" if (d / "deck.csv").exists() else "—"
        print(f"{d.name:<25} {has_agent:<10} {has_deck:<10}")
    print(f"\n{len(dirs)} experiment(s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and compare experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "experiments",
        nargs="*",
        metavar="EXP",
        help="Experiment names (directories under workspace/). Default: all.",
    )
    parser.add_argument(
        "--n-matches",
        type=int,
        default=20,
        help="Matches per pairing (doubled for both sides)",
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
    parser.add_argument(
        "--list",
        action="store_true",
        help="List experiments and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        _list_experiments()
        return

    try:
        import cg.game  # noqa: F401
    except ImportError:
        logger.error(
            "Game engine (cg) not found.  Download the Simulation SDK:\n  make sim-download"
        )
        sys.exit(1)

    exp_dirs = _resolve_experiments(args.experiments)

    # Load a valid sample deck for the baseline RandomAgent so it can start
    # real battles against the cg engine.
    sample_deck_path = Path("data/sim_sample/sample_submission/deck.csv")
    if sample_deck_path.exists():
        try:
            random_deck = [
                int(line.strip())
                for line in sample_deck_path.read_text().strip().split("\n")
                if line.strip()
            ]
        except Exception:
            random_deck = [1] * 60
    else:
        random_deck = [1] * 60

    agents = [("random", RandomAgent(deck=random_deck, random_seed=1))]
    for d in exp_dirs:
        agent = _load_agent(d)
        if agent is not None:
            agents.append((d.name, agent))
            logger.info("  loaded %s", d.name)
        else:
            logger.info("  skipped %s", d.name)

    if len(agents) < 2:
        logger.error("Need at least 2 agents to run a gauntlet (found %d)", len(agents))
        sys.exit(1)

    logger.info(
        "Running gauntlet: %d agents, %d matches per pairing",
        len(agents),
        args.n_matches,
    )
    results = run_gauntlet(
        agents=agents,
        n_matches=args.n_matches,
        max_turns=args.max_turns,
        results_dir=args.results_dir,
    )

    print()
    print(f"{'Agent':<25} {'Win Rate':<10} {'Wins':<8} {'Matches':<8}")
    print("-" * 55)
    for name, stats in sorted(results.items(), key=lambda x: x[1]["win_rate"], reverse=True):
        wr = stats["win_rate"]
        bar = "█" * int(wr * 20) + "░" * (20 - int(wr * 20))
        print(f"{name:<25} {wr:<8.3f}  {stats['wins']:<6}  {stats['matches']:<6}  {bar}")


if __name__ == "__main__":
    main()
