"""exp006 — Prize race + phase heuristics.

Extends the exp002 LucarioHeuristicAgent with:
1. Phase detection (early/mid/late based on turn + prizes)
2. Prize-aware attack selection (prefer high-value targets)
3. Phase-gated behavior weights
"""

from __future__ import annotations

# Reuse the exp002 heuristic for everything except _decide_main.
import importlib.util as _ilu
import logging
import os
from pathlib import Path as _Path

from pokemon.card_db import CardDB
from pokemon.state import (
    GameState,
    OptionInfo,
    PokemonState,
    can_pay_cost,
    effective_damage,
)

_exp002_path = _Path("workspace/exp002_lucario_heuristic/agent.py")
_spec = _ilu.spec_from_file_location("_exp002_heuristic", str(_exp002_path))
_exp002_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_exp002_mod)
_HeuristicAgent = _exp002_mod.LucarioHeuristicAgent

logger = logging.getLogger(__name__)


class PrizePhaseAgent(_HeuristicAgent):
    """Heuristic agent with prize-aware and phase-gated decision making."""

    def __init__(
        self, deck: list[int] | None = None, random_seed: int = 42, card_db: CardDB | None = None
    ):
        super().__init__(deck=deck, random_seed=random_seed, card_db=card_db)

    # --- phase detection ---------------------------------------------------
    @staticmethod
    def _detect_phase(state: GameState) -> str:
        """Return 'early', 'mid', or 'late' based on turn + prize counts."""
        prizes_left = 6 - state.me.prizes_taken
        opp_prizes_left = 6 - state.opp.prizes_taken
        if state.turn <= 3 and prizes_left >= 5:
            return "early"
        if opp_prizes_left <= 2 or prizes_left <= 2:
            return "late"
        return "mid"

    # --- prize-aware attack scoring ----------------------------------------
    @staticmethod
    def _prize_value(pokemon: PokemonState, db: CardDB) -> int:
        """Prize count for KO'ing this Pokemon: 3 for Mega ex, 2 for ex, 1 otherwise."""
        info = db.get(pokemon.id)
        if info is None:
            return 1
        if info.is_mega_ex:
            return 3
        if info.is_ex:
            return 2
        return 1

    def _find_lethal_attack_prize_scored(self, state: GameState) -> OptionInfo | None:
        """Find lethal attacks, preferring higher-prize targets.

        If multiple attacks can KO (e.g. different targets via gust effects
        or multiple attacks on the active), pick the one that yields the most
        prizes.  If the active is the only target, this reduces to the
        standard lethal check.
        """
        opp = state.opp_active
        me_act = state.my_active
        if opp is None or me_act is None:
            return None
        db = self._db()
        atk_info = db.get(me_act.id)
        atk_type = atk_info.energy_type if atk_info else None
        opp_weak = self._weakness_of(opp, db)
        best, best_score = None, -1
        for o in state.attack_options():
            atk = db.attack(o.attack_id) if o.attack_id is not None else None
            if atk is None or atk.damage <= 0:
                continue
            if not can_pay_cost(me_act.energies, atk.energies):
                continue
            dmg = effective_damage(atk.damage, atk_type, opp_weak)
            if dmg >= opp.hp:
                score = self._prize_value(opp, db)
                if score > best_score:
                    best, best_score = o, score
        return best

    def _find_best_prize_attack(self, state: GameState) -> OptionInfo | None:
        """Score attacks by (prize value, damage), not just damage alone."""
        me_act = state.my_active
        if me_act is None:
            return None
        db = self._db()
        atk_info = db.get(me_act.id)
        atk_type = atk_info.energy_type if atk_info else None
        opp = state.opp_active
        opp_weak = self._weakness_of(opp, db) if opp else None
        best, best_score = None, (-1, -1)
        for o in state.attack_options():
            atk = db.attack(o.attack_id) if o.attack_id is not None else None
            if atk is None or atk.damage <= 0:
                continue
            if not can_pay_cost(me_act.energies, atk.energies):
                continue
            dmg = effective_damage(atk.damage, atk_type, opp_weak)
            pv = self._prize_value(opp, db) if opp else 1
            # Score: primary = can KO? (prize value if KO, else 0), secondary = damage
            score = (pv if dmg >= opp.hp else 0, dmg) if opp else (0, dmg)
            if score > best_score:
                best, best_score = o, score
        return best

    # --- 2HKO awareness (exp007 will enhance this) -------------------------
    def _would_die_to_counter(self, state: GameState) -> bool:
        """Check if opponent's active can KO our active right now."""
        me_act = state.my_active
        opp_act = state.opp_active
        if me_act is None or opp_act is None:
            return False
        db = self._db()
        opp_info = db.get(opp_act.id)
        if opp_info is None:
            return False
        opp_type = opp_info.energy_type
        my_info = db.get(me_act.id)
        my_weakness = my_info.weakness if my_info else None
        for atk_energies in (m.energies for m in opp_info.moves):
            if can_pay_cost(opp_act.energies, atk_energies):
                # We know their max affordable attack damage
                opp_max_atk = max(
                    (
                        m.damage
                        for m in opp_info.moves
                        if can_pay_cost(opp_act.energies, m.energies)
                    ),
                    default=0,
                )
                return effective_damage(opp_max_atk, opp_type, my_weakness) >= me_act.hp
        return False

    # --- phase-gated MAIN --------------------------------------------------
    def _decide_main(self, state: GameState) -> list[int]:
        phase = self._detect_phase(state)
        me = state.me

        # 1. Lethal attack (any phase) — prize-scored version.
        lethal = self._find_lethal_attack_prize_scored(state)
        if lethal is not None:
            return [lethal.index]

        # 2. Retreat if active is threatened.
        if me.active is not None and me.active.hp <= self.RETREAT_HP_THRESHOLD:
            retreat = self._retreat_option(state)
            if retreat is not None:
                return [retreat.index]

        # 3. Evolve (high priority in early/mid).
        if phase != "late":
            ev = self._best_evolve_option(state)
            if ev is not None:
                return [ev.index]

        # 4. Attach energy.
        if not state.energy_attached:
            attach = self._attach_option(state)
            if attach is not None:
                return [attach.index]

        # 5. Play cards (evolutions / supporters / items / basics).
        play = self._best_play_option(state)
        if play is not None:
            return [play.index]

        # 6. Attack (with prize awareness).
        if me.active is not None:
            # In late phase: check 2HKO before committing
            if phase == "late" and self._would_die_to_counter(state):
                # Find a safer play or end turn
                pass  # Fall through to end
            best_atk = self._find_best_prize_attack(state)
            if best_atk is not None:
                return [best_atk.index]

        # 7. End turn.
        end = state.end_options()
        if end:
            return [end[0].index]
        return self._safe_default(state)


# --- Kaggle entry point -------------------------------------------------------
def _read_deck_csv() -> list[int]:
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/" + path
    with open(path) as f:
        return [int(line) for line in f.read().split("\n") if line.strip()][:60]


_agent_instance: PrizePhaseAgent | None = None


def agent(obs_dict: dict) -> list[int]:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = PrizePhaseAgent(deck=_read_deck_csv())
    return _agent_instance(obs_dict)
