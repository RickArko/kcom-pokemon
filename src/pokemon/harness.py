from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _flatten_obs(obs: dict | None) -> dict:
    """Convert raw cg observation dict to the format expected by project agents.

    The cg engine returns a nested dict (``select.option``, ``current.yourIndex``,
    etc.).  Project agents expect a flatter shape (``options``, ``minCount``,
    ``current_player``, ``done``, etc.).  This helper bridges the two formats so
    existing agents (RandomAgent, RuleBasedAgent) work without modification.
    """
    if obs is None or obs.get("select") is None:
        return {"select": None}
    flat = dict(obs)
    select = flat["select"]
    flat["options"] = list(range(len(select["option"])))
    flat["minCount"] = select["minCount"]
    flat["maxCount"] = select["maxCount"]
    current = flat.get("current", {})
    flat["current_player"] = current.get("yourIndex", 0)
    flat["done"] = current.get("result", -1) != -1
    flat["scores"] = [0, 0]  # compatibility placeholder
    return flat


def run_match(agent0, agent1, max_turns: int = 100) -> dict:
    """Run a single match between two agents and return the result.

    Requires the ``cg`` game engine to be installed in ``data/sim_sample/cg/``.
    See README.md for setup instructions.

    Parameters
    ----------
    agent0, agent1:
        Callable agents accepting ``agent(obs_dict) -> list[int]``.
    max_turns:
        Abort if the match exceeds this many turns.

    Returns
    -------
    dict with keys: winner (0 or 1), turns, score0, score1, error.
    """
    try:
        import cg.game as _cg_game
    except ImportError:
        raise RuntimeError(
            "Game engine not found. Download the simulation SDK:\n  make sim-download"
        )

    # First call: agents return their 60-card decks
    deck_obs = {"select": None}
    try:
        deck0 = agent0(deck_obs)
        deck1 = agent1(deck_obs)
    except Exception as e:
        logger.warning("Deck selection error: %s", e)
        return {
            "winner": -1,
            "turns": 0,
            "score0": 0,
            "score1": 0,
            "error": str(e),
        }

    try:
        obs, _ = _cg_game.battle_start(deck0, deck1)
    except Exception as e:
        logger.warning("Battle start error: %s", e)
        return {
            "winner": -1,
            "turns": 0,
            "score0": 0,
            "score1": 0,
            "error": str(e),
        }

    flat_obs = _flatten_obs(obs)
    turn = 0
    try:
        while not flat_obs.get("done", False) and turn < max_turns:
            current_player = flat_obs["current_player"]
            agent = agent0 if current_player == 0 else agent1
            actions = agent(flat_obs)
            obs = _cg_game.battle_select(actions)
            flat_obs = _flatten_obs(obs)
            turn += 1
    except Exception as e:
        logger.warning("Match error at turn %d: %s", turn, e)
        _cg_game.battle_finish()
        return {
            "winner": -1,
            "turns": turn,
            "score0": 0,
            "score1": 0,
            "error": str(e),
        }

    _cg_game.battle_finish()

    result = obs.get("current", {}).get("result", -1) if obs else -1
    if result == 0:
        winner = 0
    elif result == 1:
        winner = 1
    else:
        winner = -1

    return {
        "winner": winner,
        "turns": turn,
        "score0": 1 if winner == 0 else 0,
        "score1": 1 if winner == 1 else 0,
        "error": None,
    }


def run_gauntlet(
    agents: list,
    n_matches: int = 20,
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
    results_dir:
        Directory to save per-agent result JSON files.

    Returns
    -------
    dict mapping agent name to win-rate stats.
    """
    try:
        import cg.game  # noqa: F401
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
            pairing_start = time.time()
            for m in range(n_matches):
                result = run_match(agent_a, agent_b, max_turns=max_turns)
                if result["winner"] == 0:
                    wins_a += 1
                    win_counts[name_a] += 1
                elif result["winner"] == 1:
                    win_counts[name_b] += 1
                match_counts[name_a] += 1
                match_counts[name_b] += 1
                p_elapsed = time.time() - pairing_start
                avg = p_elapsed / (m + 1)
                eta = avg * (n_matches - m - 1)
                tag = "err" if result["error"] else "ok"
                logger.info(
                    "    [%d/%d] %s vs %s: %d-%d  %.1fs  avg=%.2fs eta=%.0fs (%s)",
                    m + 1,
                    n_matches,
                    name_a,
                    name_b,
                    wins_a,
                    m + 1 - wins_a,
                    p_elapsed,
                    avg,
                    eta,
                    tag,
                )

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
