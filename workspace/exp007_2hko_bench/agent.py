"""exp007 — 2HKO awareness + bench protection.

Extends exp006 PrizePhaseAgent with:

1. Full 2HKO calculation — before attacking in mid/late, check if the
   opponent's active can KO us back.  If yes, prefer retreat, evolve, or
   end turn over a suicidal attack.

2. Bench protection — avoid benching Pokemon with HP below known opponent
   spread thresholds (Dragapult Phantom Dive = 60 to bench, Abomasnow
   Hammer-lanche = spread).  Prefer to keep high-HP basics on bench.
"""

from __future__ import annotations

import importlib.util as _ilu
import logging
import os
from pathlib import Path as _Path

from pokemon.card_db import CardDB
from pokemon.state import (
    GameState,
    OptionInfo,
    can_pay_cost,
    effective_damage,
)

_exp006_path = _Path("workspace/exp006_prize_phase/agent.py")
_spec = _ilu.spec_from_file_location("_exp006_prize", str(_exp006_path))
_exp006_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_exp006_mod)
_PrizePhaseAgent = _exp006_mod.PrizePhaseAgent

logger = logging.getLogger(__name__)


# Known spread damage from sample deck archetypes.
# Key: archetype name -> (direct damage, bench spread damage)
_KNOWN_SPREAD: dict[str, tuple[int, int]] = {
    "dragapult_ex": (200, 60),  # Phantom Dive
    "mega_abomasnow_ex": (200, 30),  # Hammer-lanche (estimated 30 to bench)
}


class TwoHKOBenchAgent(_PrizePhaseAgent):
    """Heuristic agent with 2HKO awareness and bench protection."""

    SPREAD_HP_THRESHOLD = 70  # Bench Pokemon below this HP are at risk

    def __init__(
        self, deck: list[int] | None = None, random_seed: int = 42, card_db: CardDB | None = None
    ):
        super().__init__(deck=deck, random_seed=random_seed, card_db=card_db)
        self._known_archetype: str | None = None

    # --- 2HKO: enhanced counter-attack simulation --------------------------
    def _opponent_max_damage(self, state: GameState) -> int:
        """Maximum damage opponent's active can deal to us right now."""
        opp_act = state.opp_active
        me_act = state.my_active
        if opp_act is None or me_act is None:
            return 0
        db = self._db()
        opp_info = db.get(opp_act.id)
        if opp_info is None:
            return 0
        opp_type = opp_info.energy_type
        my_info = db.get(me_act.id)
        my_weakness = my_info.weakness if my_info else None
        best_dmg = 0
        for m in opp_info.moves:
            if m.damage <= 0:
                continue
            if can_pay_cost(opp_act.energies, m.energies):
                dmg = effective_damage(m.damage, opp_type, my_weakness)
                if dmg > best_dmg:
                    best_dmg = dmg
        return best_dmg

    def _would_die_to_counter(self, state: GameState) -> bool:
        """True if opponent's active can KO our active right now."""
        me_act = state.my_active
        if me_act is None:
            return False
        return self._opponent_max_damage(state) >= me_act.hp

    def _would_2hko_us(self, state: GameState) -> bool:
        """True if opponent can 2-shot our active (dangerous to stay in)."""
        me_act = state.my_active
        if me_act is None:
            return False
        opp_dmg = self._opponent_max_damage(state)
        return opp_dmg > 0 and opp_dmg >= me_act.hp / 2

    def _safe_attack_available(self, state: GameState) -> OptionInfo | None:
        """Find an attack that doesn't leave us dead to counter.

        Returns the attack option if we survive, or None if all attacks
        are suicidal.
        """
        if not self._would_die_to_counter(state):
            # We survive — okay to attack
            return self._find_best_prize_attack(state)
        return None

    # --- bench protection ---------------------------------------------------
    def _bench_hp_safe(self, state: GameState, hp: int) -> bool:
        """Check if a Pokemon with this HP is safe on bench vs known threats.

        In mid/late game, spread damage can KO weak benched Pokemon.
        """
        if hp <= 30:
            return False  # Dies to any spread
        if hp <= self.SPREAD_HP_THRESHOLD and state.turn >= 4:
            return False  # Fragile in mid/late
        return True

    def _pick_bench_setup(self, state: GameState, opts: list[OptionInfo]) -> list[int]:
        """Override bench setup: prefer high-HP basics, avoid fragile ones."""
        db = self._db()
        scored: list[tuple[int, float, OptionInfo]] = []
        for o in opts:
            cid = self._option_hand_card_id(state, o)
            info = db.get(cid) if cid is not None else None
            hp = info.hp if info and info.is_pokemon else 0
            safe_bonus = 1.0 if self._bench_hp_safe(state, hp) else 0.0
            scored.append((hp, safe_bonus, o))
        # Sort by (safe_bonus desc, hp desc)
        scored.sort(key=lambda x: (-x[1], -x[0]))
        n = max(state.select.min_count, min(state.select.max_count, len(scored)))
        return [o.index for _, _, o in scored[:n]]

    # --- phase-gated MAIN with 2HKO ----------------------------------------
    def _decide_main(self, state: GameState) -> list[int]:
        phase = self._detect_phase(state)
        me = state.me

        # 1. Lethal attack (always take it).
        lethal = self._find_lethal_attack_prize_scored(state)
        if lethal is not None:
            return [lethal.index]

        # 2. Retreat if active is threatened (low HP OR would die to counter).
        needs_retreat = False
        if me.active is not None:
            if me.active.hp <= self.RETREAT_HP_THRESHOLD:
                needs_retreat = True
            elif phase in ("mid", "late") and self._would_die_to_counter(state):
                needs_retreat = True
        if needs_retreat:
            retreat = self._retreat_option(state)
            if retreat is not None:
                return [retreat.index]

        # 3. Evolve (early/mid only).
        if phase != "late":
            ev = self._best_evolve_option(state)
            if ev is not None:
                return [ev.index]

        # 4. Attach energy.
        if not state.energy_attached:
            attach = self._attach_option(state)
            if attach is not None:
                return [attach.index]

        # 5. Play cards.
        play = self._best_play_option(state)
        if play is not None:
            return [play.index]

        # 6. Attack — only if safe (mid/late) or any attack (early).
        if me.active is not None:
            if phase == "early":
                best_atk = self._find_best_prize_attack(state)
                if best_atk is not None:
                    return [best_atk.index]
            else:
                safe = self._safe_attack_available(state)
                if safe is not None:
                    return [safe.index]
                # If no safe attack, try to retreat instead of dying
                retreat = self._retreat_option(state)
                if retreat is not None:
                    return [retreat.index]

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


_agent_instance: TwoHKOBenchAgent | None = None


def agent(obs_dict: dict) -> list[int]:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = TwoHKOBenchAgent(deck=_read_deck_csv())
    return _agent_instance(obs_dict)
