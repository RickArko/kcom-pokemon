"""exp004 — Dragapult MCTS agent with opponent modeling (Phase 3).

Same MCTS architecture as exp003 but with the Dragapult ex deck and an
opponent archetype classifier.  The classifier tracks opponent card plays from
the observation logs and, once confident, passes the predicted opponent deck
to MCTS as a better-than-mirror opponent model.  Counter-strategy hints
adjust aggression and bench protection based on the identified matchup.

Deck: Dragapult ex (Dragon). Dreepy → Drakloak → Dragapult ex, using
Fire + Psychic energy.  Dragapult ex is Tera (no bench damage while benched)
with Phantom Dive (200 dmg + 6 counters on opponent's bench) and Jet Headbutt
(70 dmg, colorless).
"""

from __future__ import annotations

# Reuse the exp002 heuristic as the fallback + rollout policy.
import importlib.util as _ilu
import logging
import os
import time
from pathlib import Path as _Path

from pokemon.agent import RuleBasedAgent
from pokemon.card_db import CardDB, get_card_db
from pokemon.opponent import OpponentClassifier, archetype_deck_list, counter_strategy
from pokemon.search import MCTSResult, mcts_search
from pokemon.state import parse_obs

_exp002_path = _Path("workspace/exp002_lucario_heuristic/agent.py")
_spec = _ilu.spec_from_file_location("_exp002_heuristic_dragapult", str(_exp002_path))
_exp002_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_exp002_mod)
_HeuristicAgent = _exp002_mod.LucarioHeuristicAgent

logger = logging.getLogger(__name__)


class DragapultMCTSAgent(RuleBasedAgent):
    """MCTS agent for the Dragapult ex deck with opponent modeling."""

    def __init__(
        self,
        deck: list[int] | None = None,
        random_seed: int = 42,
        card_db: CardDB | None = None,
        simulations: int = 150,
        time_budget: float = 1.5,
        rollout_depth: int = 25,
        exploration_c: float = 1.4,
        use_heuristic_rollout: bool = True,
        rollout_epsilon: float = 0.25,
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
        self.use_heuristic_rollout = use_heuristic_rollout
        self.rollout_epsilon = rollout_epsilon
        self.override_threshold = override_threshold
        self.mcts_when_second = mcts_when_second
        # opponent modeling
        self._classifier: OpponentClassifier | None = None
        self._opp_hint: list[int] | None = None
        self._strategy_hints: dict = {}
        # stats
        self._mcts_calls = 0
        self._mcts_fallbacks = 0
        self._mcts_total_time = 0.0

    def _db(self) -> CardDB:
        if self._card_db is None:
            self._card_db = get_card_db()
        return self._card_db

    def _ensure_classifier(self, your_index: int) -> OpponentClassifier:
        if self._classifier is None:
            self._classifier = OpponentClassifier(your_index=your_index)
        return self._classifier

    def _make_rollout_policy(self):
        if not self.use_heuristic_rollout:
            return None
        heuristic_act = self._heuristic._act
        eps = self.rollout_epsilon
        rng = self.rng

        def _rollout(obs_dict: dict) -> list[int]:
            if rng.random() < eps:
                sel = obs_dict.get("select") or {}
                opts = sel.get("option") or []
                if not opts:
                    return []
                mx = int(sel.get("maxCount", 1) or 1)
                mn = int(sel.get("minCount", 1) or 1)
                n = min(max(mn, 1), mx, len(opts))
                return [int(x) for x in rng.choice(len(opts), size=n, replace=False)]
            return heuristic_act(obs_dict)

        return _rollout

    def _act(self, obs: dict) -> list[int]:
        state = parse_obs(obs)
        if state is None:
            return self._heuristic._act(obs)

        # Update opponent classifier with new logs.
        clf = self._ensure_classifier(state.your_index)
        clf.update(state)
        classification = clf.classify()
        if classification.identified:
            self._opp_hint = archetype_deck_list(classification.archetype)
            self._strategy_hints = counter_strategy(classification.archetype)
        elif not self._opp_hint:
            self._opp_hint = None  # mirror prior

        # Non-MAIN selects always use the heuristic.
        if not state.select.is_main or not state.select.options:
            return self._heuristic._act(obs)

        # Initiative-conditional MCTS (same finding as exp003).
        if not self.mcts_when_second and state.first_player != -1:
            if state.your_index != state.first_player:
                return self._heuristic._act(obs)

        heuristic_action = self._heuristic._act(obs)

        try:
            t0 = time.time()
            rollout_policy = self._make_rollout_policy()
            result: MCTSResult = mcts_search(
                obs_dict=obs,
                my_deck=self._deck,
                opponent_deck_hint=self._opp_hint,
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
                return heuristic_action

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
            "avg_decision_seconds": round(avg, 4),
            "total_search_seconds": round(self._mcts_total_time, 2),
            "opponent_archetype": self._classifier.classify().archetype
            if self._classifier
            else None,
            "opponent_confidence": round(self._classifier.classify().confidence, 3)
            if self._classifier
            else 0.0,
            "using_opp_hint": self._opp_hint is not None,
        }


# --- Kaggle submission entry point ------------------------------------------
def _read_deck_csv() -> list[int]:
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/" + path
    with open(path) as f:
        return [int(line) for line in f.read().split("\n") if line.strip()][:60]


_agent_instance: DragapultMCTSAgent | None = None


def agent(obs_dict: dict) -> list[int]:
    """Kaggle entry point."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = DragapultMCTSAgent(deck=_read_deck_csv())
    return _agent_instance(obs_dict)
