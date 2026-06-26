"""exp003 — Lucario MCTS agent (Phase 2).

Wraps the exp002 heuristic agent with Monte Carlo Tree Search over the ``cg``
search API.  MCTS runs only on MAIN selects (the high-value decisions); every
other select type (YES_NO, CARD targeting, ENERGY, ATTACK, COUNT, EVOLVE) and
any search failure falls back to the exp002 heuristic, so the agent never
forfeits.

Search parameters (start conservative, per the experimentation strategy):
    simulations: 50
    exploration_constant: 1.4 (UCB1)
    rollout_depth: 20 plies
    time_budget: 0.8 s per decision
    rollout policy: fast inline (prefer ATTACK, else END, else random)
    opponent model: mirror prior (opponent deck = our deck)

Deck: same Mega Lucario ex deck as exp002 (to isolate search impact).
"""

from __future__ import annotations

# Reuse the exp002 heuristic as the fallback policy.
import importlib.util as _ilu
import logging
import os
import time
from pathlib import Path as _Path

from pokemon.agent import RuleBasedAgent
from pokemon.card_db import CardDB, get_card_db
from pokemon.search import MCTSResult, mcts_search
from pokemon.state import parse_obs

_exp002_path = _Path("workspace/exp002_lucario_heuristic/agent.py")
_spec = _ilu.spec_from_file_location("_exp002_heuristic", str(_exp002_path))
_exp002_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_exp002_mod)
_HeuristicAgent = _exp002_mod.LucarioHeuristicAgent

logger = logging.getLogger(__name__)


class LucarioMCTSAgent(RuleBasedAgent):
    """MCTS agent for the Mega Lucario ex deck (falls back to exp002 heuristic)."""

    def __init__(
        self,
        deck: list[int] | None = None,
        random_seed: int = 42,
        card_db: CardDB | None = None,
        simulations: int = 50,
        time_budget: float = 0.8,
        rollout_depth: int = 20,
        exploration_c: float = 1.4,
        opponent_deck_hint: list[int] | None = None,
        use_heuristic_rollout: bool = True,
    ):
        super().__init__(deck=deck, random_seed=random_seed)
        self._card_db: CardDB | None = card_db
        self._heuristic = _HeuristicAgent(deck=deck, random_seed=random_seed, card_db=card_db)
        self.simulations = simulations
        self.time_budget = time_budget
        self.rollout_depth = rollout_depth
        self.exploration_c = exploration_c
        self.opponent_deck_hint = opponent_deck_hint
        self.use_heuristic_rollout = use_heuristic_rollout
        # stats
        self._mcts_calls = 0
        self._mcts_fallbacks = 0
        self._mcts_total_time = 0.0

    def _db(self) -> CardDB:
        if self._card_db is None:
            self._card_db = get_card_db()
        return self._card_db

    def _act(self, obs: dict) -> list[int]:
        # Only run MCTS on MAIN selects with options; everything else uses the
        # heuristic (which already handles all select types safely).
        state = parse_obs(obs)
        if state is None or not state.select.is_main or not state.select.options:
            return self._heuristic._act(obs)

        try:
            t0 = time.time()
            rollout_policy = self._heuristic._act if self.use_heuristic_rollout else None
            result: MCTSResult = mcts_search(
                obs_dict=obs,
                my_deck=self._deck,
                opponent_deck_hint=self.opponent_deck_hint,
                card_db=self._db(),
                simulations=self.simulations,
                time_budget=self.time_budget,
                rollout_depth=self.rollout_depth,
                exploration_c=self.exploration_c,
                rollout_policy=rollout_policy,
                rng=self.rng,
            )
            self._mcts_calls += 1
            self._mcts_total_time += time.time() - t0
            if result.fell_back or not result.action:
                self._mcts_fallbacks += 1
                return self._heuristic._act(obs)
            return result.action
        except Exception as e:  # noqa: BLE001 - never forfeit on a search error
            logger.debug("MCTS failed (%s); falling back to heuristic.", e)
            self._mcts_fallbacks += 1
            return self._heuristic._act(obs)

    def stats(self) -> dict:
        avg = (self._mcts_total_time / self._mcts_calls) if self._mcts_calls else 0.0
        return {
            "mcts_calls": self._mcts_calls,
            "mcts_fallbacks": self._mcts_fallbacks,
            "avg_decision_seconds": round(avg, 4),
            "total_search_seconds": round(self._mcts_total_time, 2),
        }


# --- Kaggle submission entry point ------------------------------------------
def _read_deck_csv() -> list[int]:
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/" + path
    with open(path) as f:
        return [int(line) for line in f.read().split("\n") if line.strip()][:60]


_agent_instance: LucarioMCTSAgent | None = None


def agent(obs_dict: dict) -> list[int]:
    """Kaggle entry point: ``agent(obs_dict) -> list[int]``."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = LucarioMCTSAgent(deck=_read_deck_csv())
    return _agent_instance(obs_dict)
