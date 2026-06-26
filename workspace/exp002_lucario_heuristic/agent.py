"""exp002 — Lucario heuristic agent (re-exports from pokemon.heuristic)."""

from __future__ import annotations

import os

from pokemon.heuristic import LucarioHeuristicAgent

# --- Kaggle submission entry point ------------------------------------------


def _read_deck_csv() -> list[int]:
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/" + path
    with open(path) as f:
        return [int(line) for line in f.read().split("\n") if line.strip()][:60]


_agent_instance: LucarioHeuristicAgent | None = None


def agent(obs_dict: dict) -> list[int]:
    """Kaggle entry point: ``agent(obs_dict) -> list[int]``."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = LucarioHeuristicAgent(deck=_read_deck_csv())
    return _agent_instance(obs_dict)
