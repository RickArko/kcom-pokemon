from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def run_match(agent0, agent1, env, max_turns: int = 100) -> dict:
    """Run a single match between two agents and return the result.

    Requires the ``cg`` game engine to be installed in ``data/sim_sample/cg/``.
    See README.md for setup instructions.

    Parameters
    ----------
    agent0, agent1:
        Callable agents accepting ``agent(obs_dict) -> list[int]``.
    env:
        Game engine environment (e.g. ``cg.game.Game()``).
    max_turns:
        Abort if the match exceeds this many turns.

    Returns
    -------
    dict with keys: winner (0 or 1), turns, score0, score1, error.
    """
    obs = env.reset()
    turn = 0

    try:
        while not obs.get("done", False) and turn < max_turns:
            current_player = obs.get("current_player", 0)
            agent = agent0 if current_player == 0 else agent1
            actions = agent(obs)
            obs = env.step(actions)
            turn += 1
    except Exception as e:
        logger.warning("Match error at turn %d: %s", turn, e)
        return {
            "winner": -1,
            "turns": turn,
            "score0": 0,
            "score1": 0,
            "error": str(e),
        }

    scores = obs.get("scores", [0, 0])
    winner = np.argmax(scores) if scores[0] != scores[1] else -1
    return {
        "winner": int(winner),
        "turns": turn,
        "score0": int(scores[0]),
        "score1": int(scores[1]),
        "error": None,
    }


def run_gauntlet(
    agents: list,
    n_matches: int = 20,
    env_factory=None,
    results_dir: str = "workspace/results",
    max_turns: int = 100,
) -> dict:
    """Run a round-robin tournament between agents.

    Each pair plays ``n_matches`` games with swapped first/second positions to
    cancel starting-player bias.

    Parameters
    ----------
    agents:
        List of ``(name, agent_callable)`` tuples.
    n_matches:
        Number of matches per pairing (will be doubled for both sides).
    env_factory:
        Callable that returns a fresh game environment.  If None, expects the
        ``cg`` package to be importable and uses ``cg.game.Game``.
    results_dir:
        Directory to save per-agent result JSON files.

    Returns
    -------
    dict mapping agent name to win-rate stats.
    """
    if env_factory is None:
        try:
            from cg.game import Game as _Game

            env_factory = _Game
        except ImportError:
            raise RuntimeError(
                "Game engine not found. Download the simulation SDK:\n  make sim-download"
            )

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    win_counts = {name: 0 for name, _ in agents}
    match_counts = {name: 0 for name, _ in agents}
    total_start = time.time()

    n_agents = len(agents)
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue
            name_a, agent_a = agents[i]
            name_b, agent_b = agents[j]

            wins_a = 0
            for m in range(n_matches):
                env = env_factory()
                result = run_match(agent_a, agent_b, env, max_turns=max_turns)
                if result["winner"] == 0:
                    wins_a += 1
                    win_counts[name_a] += 1
                elif result["winner"] == 1:
                    win_counts[name_b] += 1
                match_counts[name_a] += 1
                match_counts[name_b] += 1

            elapsed = time.time() - total_start
            logger.info(
                "  %s vs %s: %d-%d (%.1fs)",
                name_a,
                name_b,
                wins_a,
                n_matches - wins_a,
                elapsed,
            )

    results = {}
    for name in win_counts:
        n = match_counts[name]
        wr = win_counts[name] / n if n > 0 else 0.0
        results[name] = {"win_rate": round(wr, 4), "wins": win_counts[name], "matches": n}

    with open(results_path / "gauntlet_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Gauntlet complete in %.1fs", time.time() - total_start)
    for name, stats in results.items():
        logger.info(
            "  %-20s  WR: %.3f  (%d/%d)",
            name,
            stats["win_rate"],
            stats["wins"],
            stats["matches"],
        )

    return results
