"""exp008 — Full heuristic MCTS + opponent classifier.

Wraps exp007 heuristic (which includes prize race, phase, and 2HKO) with
MCTS.  Improvements over exp003:

1. Full heuristic rollout (deterministic) — removes epsilon-greedy noise
   so MCTS simulations are higher quality.

2. Opponent classifier — ports the exp004 archetype classifier into the
   MCTS loop.  Once the opponent is identified with >= 0.5 confidence,
   passes their actual deck list as opponent_deck_hint (replaces mirror
   assumption).

3. Uses the optimized deck from exp005.
"""

from __future__ import annotations

import importlib.util as _ilu
import logging
import os
import time
from pathlib import Path as _Path

from pokemon.agent import RuleBasedAgent
from pokemon.card_db import CardDB, get_card_db
from pokemon.opponent import (
    OpponentClassifier,
    archetype_deck_list,
    counter_strategy,
)
from pokemon.search import MCTSResult, mcts_search
from pokemon.state import parse_obs

_exp007_path = _Path("workspace/exp007_2hko_bench/agent.py")
_spec = _ilu.spec_from_file_location("_exp007_heuristic", str(_exp007_path))
_exp007_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_exp007_mod)
_HeuristicAgent = _exp007_mod.TwoHKOBenchAgent

logger = logging.getLogger(__name__)


class FullMCTSAgent(RuleBasedAgent):
    """MCTS agent with full heuristic rollout and opponent classifier."""

    def __init__(
        self,
        deck: list[int] | None = None,
        random_seed: int = 42,
        card_db: CardDB | None = None,
        simulations: int = 150,
        time_budget: float = 1.5,
        rollout_depth: int = 25,
        exploration_c: float = 1.4,
        override_threshold: float = 0.05,
        mcts_when_second: bool = False,
    ):
        super().__init__(deck=deck, random_seed=random_seed)
        self._card_db: CardDB | None = card_db
        self._heuristic = _HeuristicAgent(deck=deck, random_seed=random_seed, card_db=card_db)
        self.simulations = simulations
        self.time_budget = time_budget
        self.rollout_depth = rollout_depth
        self.exploration_c = exploration_c
        self.override_threshold = override_threshold
        self.mcts_when_second = mcts_when_second

        # Opponent classifier (persistent across calls)
        self._classifier: OpponentClassifier | None = None
        self._opp_hint: list[int] | None = None
        self._strategy_hints: dict | None = None

        # Gauntlet --fast override
        if os.environ.get("GAUNTLET_FAST"):
            self.simulations = 20
            self.time_budget = 0.3
            self.rollout_depth = 15

        # Stats
        self._mcts_calls = 0
        self._mcts_fallbacks = 0
        self._mcts_total_time = 0.0
        self._classifications = 0

    def _db(self) -> CardDB:
        if self._card_db is None:
            self._card_db = get_card_db()
        return self._card_db

    def _ensure_classifier(self, your_index: int) -> OpponentClassifier:
        if self._classifier is None:
            self._classifier = OpponentClassifier(your_index=your_index)
        return self._classifier

    def _update_classifier(self, state) -> None:
        """Update opponent classifier from state logs, then sync hints."""
        if state is None:
            return
        clf = self._ensure_classifier(state.your_index)
        clf.update(state)
        result = clf.classify()
        if result.identified:
            self._classifications += 1
            self._opp_hint = archetype_deck_list(result.archetype)
            self._strategy_hints = counter_strategy(result.archetype)
            # Also forward archetype knowledge to the heuristic for bench protection
            self._heuristic._known_archetype = result.archetype

    # --- Rollout: full heuristic (deterministic) ---------------------------
    def _make_rollout_policy(self):
        """Deterministic heuristic rollout — no epsilon.

        Every simulation uses the exp007 heuristic's _act, which already
        includes prize race, phase, and 2HKO awareness.  This gives
        higher-quality simulations than epsilon-greedy.
        """
        return self._heuristic._act

    def _act(self, obs: dict) -> list[int]:
        state = parse_obs(obs)

        # Update opponent classifier from this observation's logs.
        self._update_classifier(state)

        # Only run MCTS on MAIN selects with options.
        if state is None or not state.select.is_main or not state.select.options:
            return self._heuristic._act(obs)

        # Initiative-conditional MCTS.
        if not self.mcts_when_second and state.first_player != -1:
            if state.your_index != state.first_player:
                return self._heuristic._act(obs)

        # Heuristic baseline.
        heuristic_action = self._heuristic._act(obs)

        try:
            t0 = time.time()
            result: MCTSResult = mcts_search(
                obs_dict=obs,
                my_deck=self._deck,
                opponent_deck_hint=self._opp_hint,  # Classified or None
                card_db=self._db(),
                simulations=self.simulations,
                time_budget=self.time_budget,
                rollout_depth=self.rollout_depth,
                exploration_c=self.exploration_c,
                rollout_policy=self._make_rollout_policy(),
                rng=self.rng,
            )
            self._mcts_calls += 1
            self._mcts_total_time += time.time() - t0
            if result.fell_back or not result.action:
                self._mcts_fallbacks += 1
                return heuristic_action

            # Hybrid override.
            mcts_idx = result.action[0] if result.action else -1
            heur_idx = heuristic_action[0] if heuristic_action else -1
            if mcts_idx == heur_idx:
                return heuristic_action
            mcts_wr = result.win_rates.get(mcts_idx, 0.0)
            heur_wr = result.win_rates.get(heur_idx, 0.0)
            if mcts_wr - heur_wr > self.override_threshold:
                return result.action
            return heuristic_action
        except Exception as e:  # noqa: BLE001
            logger.debug("MCTS failed (%s); falling back to heuristic.", e)
            self._mcts_fallbacks += 1
            return heuristic_action

    def stats(self) -> dict:
        avg = (self._mcts_total_time / self._mcts_calls) if self._mcts_calls else 0.0
        return {
            "mcts_calls": self._mcts_calls,
            "mcts_fallbacks": self._mcts_fallbacks,
            "classifications": self._classifications,
            "avg_decision_seconds": round(avg, 4),
            "total_search_seconds": round(self._mcts_total_time, 2),
        }


# --- Kaggle entry point -------------------------------------------------------
def _read_deck_csv() -> list[int]:
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/" + path
    with open(path) as f:
        return [int(line) for line in f.read().split("\n") if line.strip()][:60]


_agent_instance: FullMCTSAgent | None = None


def agent(obs_dict: dict) -> list[int]:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = FullMCTSAgent(deck=_read_deck_csv())
    return _agent_instance(obs_dict)
