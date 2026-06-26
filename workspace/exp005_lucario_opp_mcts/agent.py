"""exp005 — Lucario MCTS with opponent modeling + effect-damage (Phase 4).

Evolution of exp003 integrating:
1. **Opponent classifier** — classifies opponent deck from early-game logs,
   passes the real opponent deck to MCTS (replacing the mirror prior).
2. **Effect-damage attack evaluation** — estimates damage for attacks with
   printed damage=0 (e.g. Kyogre Riptide: "20 damage for each {W} Energy in
   discard pile") by parsing the attack text and counting relevant cards.
3. **MCTS** — same architecture as exp003 (initiative-conditional, hybrid
   override, epsilon-greedy heuristic rollout).

Deck: same Mega Lucario ex deck as exp002/exp003.
"""

from __future__ import annotations

# Reuse the exp002 heuristic as the fallback + rollout policy.
import logging
import os
import time

from pokemon.agent import RuleBasedAgent
from pokemon.card_db import CardDB, get_card_db
from pokemon.heuristic import LucarioHeuristicAgent as _HeuristicAgent
from pokemon.opponent import OpponentClassifier, archetype_deck_list, counter_strategy
from pokemon.search import MCTSResult, mcts_search
from pokemon.state import (
    GameState,
    OptionInfo,
    PokemonState,
    can_pay_cost,
    effective_damage,
    parse_obs,
)

# Basic energy card IDs (for discard-pile counting in effect-damage estimation).
_BASIC_ENERGY_IDS = {1, 2, 3, 4, 5, 6, 7, 8}
_WATER_ENERGY_ID = 3
_FIRE_ENERGY_ID = 2
_PSYCHIC_ENERGY_ID = 5
_FIGHTING_ENERGY_ID = 6

logger = logging.getLogger(__name__)


class ImprovedHeuristicAgent(_HeuristicAgent):
    """Heuristic with effect-damage attack evaluation."""

    def _count_discard_energy(self, state: GameState, energy_id: int) -> int:
        """Count cards with the given basic energy ID in our discard pile."""
        return sum(1 for cid in state.me.discard if cid == energy_id)

    def _attack_damage(self, atk, state: GameState, me_act: PokemonState) -> int:
        """Return effective damage for an attack, including effect-damage estimates."""
        if atk.damage > 0:
            return atk.damage
        # Effect-damage: estimate from attack text + game state.
        energy_count = me_act.energy_count
        # Count relevant energy in discard pile for "discard pile" attacks.
        discard_water = self._count_discard_energy(state, _WATER_ENERGY_ID)
        discard_fighting = self._count_discard_energy(state, _FIGHTING_ENERGY_ID)
        discard_count = max(discard_water, discard_fighting)
        return atk.estimate_damage(energy_count=energy_count, discard_energy_count=discard_count)

    def _find_lethal_attack(self, state: GameState) -> OptionInfo | None:
        opp = state.opp_active
        me_act = state.my_active
        if opp is None or me_act is None:
            return None
        db = self._db()
        atk_info = db.get(me_act.id)
        atk_type = atk_info.energy_type if atk_info else None
        opp_weak = self._weakness_of(opp, db)
        for o in state.attack_options():
            atk = db.attack(o.attack_id) if o.attack_id is not None else None
            if atk is None:
                continue
            if not can_pay_cost(me_act.energies, atk.energies):
                continue
            base_dmg = self._attack_damage(atk, state, me_act)
            if base_dmg <= 0:
                continue
            if effective_damage(base_dmg, atk_type, opp_weak) >= opp.hp:
                return o
        return None

    def _find_best_damage_attack(self, state: GameState) -> OptionInfo | None:
        me_act = state.my_active
        if me_act is None:
            return None
        db = self._db()
        atk_info = db.get(me_act.id)
        atk_type = atk_info.energy_type if atk_info else None
        opp = state.opp_active
        opp_weak = self._weakness_of(opp, db) if opp else None
        best, best_dmg = None, -1
        for o in state.attack_options():
            atk = db.attack(o.attack_id) if o.attack_id is not None else None
            if atk is None:
                continue
            if not can_pay_cost(me_act.energies, atk.energies):
                continue
            base_dmg = self._attack_damage(atk, state, me_act)
            if base_dmg <= 0:
                continue
            dmg = effective_damage(base_dmg, atk_type, opp_weak)
            if dmg > best_dmg:
                best, best_dmg = o, dmg
        return best


class LucarioMCTSOpponentAgent(RuleBasedAgent):
    """MCTS agent with opponent modeling + effect-damage evaluation (Phase 4)."""

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
        self._heuristic = ImprovedHeuristicAgent(
            deck=deck, random_seed=random_seed, card_db=card_db
        )
        self.simulations = simulations
        self.time_budget = time_budget
        self.rollout_depth = rollout_depth
        self.exploration_c = exploration_c
        self.use_heuristic_rollout = use_heuristic_rollout
        self.rollout_epsilon = rollout_epsilon
        self.override_threshold = override_threshold
        self.mcts_when_second = mcts_when_second
        self._classifier: OpponentClassifier | None = None
        self._opp_hint: list[int] | None = None
        self._strategy_hints: dict = {}
        self._mcts_calls = 0
        self._mcts_fallbacks = 0
        self._mcts_total_time = 0.0
        self._max_decision_time = 0.0

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

        if not state.select.is_main or not state.select.options:
            return self._heuristic._act(obs)

        # Initiative-conditional MCTS.
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
            dt = time.time() - t0
            self._mcts_calls += 1
            self._mcts_total_time += dt
            self._max_decision_time = max(self._max_decision_time, dt)
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
        clf_result = self._classifier.classify() if self._classifier else None
        return {
            "mcts_calls": self._mcts_calls,
            "mcts_fallbacks": self._mcts_fallbacks,
            "avg_decision_seconds": round(avg, 4),
            "max_decision_seconds": round(self._max_decision_time, 4),
            "total_search_seconds": round(self._mcts_total_time, 2),
            "opponent_archetype": clf_result.archetype if clf_result else None,
            "opponent_confidence": round(clf_result.confidence, 3) if clf_result else 0.0,
            "using_opp_hint": self._opp_hint is not None,
        }


# --- Kaggle submission entry point ------------------------------------------
def _read_deck_csv() -> list[int]:
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/" + path
    with open(path) as f:
        return [int(line) for line in f.read().split("\n") if line.strip()][:60]


_agent_instance: LucarioMCTSOpponentAgent | None = None


def agent(obs_dict: dict) -> list[int]:
    """Kaggle entry point."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = LucarioMCTSOpponentAgent(deck=_read_deck_csv())
    return _agent_instance(obs_dict)
