"""Enhanced heuristic agents: prize-phase + 2HKO/bench-protection awareness.

Consolidates the exp006 (PrizePhaseAgent) and exp007 (TwoHKOBenchAgent)
enhancements into the bundled ``pokemon`` package so Kaggle submissions are
self-contained (no workspace file-path imports).  These subclass
:class:`pokemon.heuristic.LucarioHeuristicAgent`:

- :class:`PrizePhaseAgent` — phase detection (early/mid/late) + prize-aware
  attack scoring (prefer KOs that yield more prizes: Mega ex = 3, ex = 2).
- :class:`TwoHKOBenchAgent` — adds 2HKO awareness (don't attack if the opponent
  can KO us back; retreat instead) and bench protection (avoid benching
  low-HP Pokemon vulnerable to spread damage).

Both remain deck-agnostic enough to pilot any deck, but are tuned for the
Fighting/Mega-evolution playstyle of the Lucario family.
"""

from __future__ import annotations

import logging

from pokemon.card_db import CardDB
from pokemon.heuristic import LucarioHeuristicAgent
from pokemon.state import (
    GameState,
    OptionInfo,
    PokemonState,
    can_pay_cost,
    effective_damage,
)

logger = logging.getLogger(__name__)


class PrizePhaseAgent(LucarioHeuristicAgent):
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
        """Find lethal attacks, preferring higher-prize targets."""
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
        """Score attacks by (prize value if KO, damage), not just damage alone."""
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
            score = (pv if dmg >= opp.hp else 0, dmg) if opp else (0, dmg)
            if score > best_score:
                best, best_score = o, score
        return best

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

        lethal = self._find_lethal_attack_prize_scored(state)
        if lethal is not None:
            return [lethal.index]

        if me.active is not None and me.active.hp <= self.RETREAT_HP_THRESHOLD:
            retreat = self._retreat_option(state)
            if retreat is not None:
                return [retreat.index]

        if phase != "late":
            ev = self._best_evolve_option(state)
            if ev is not None:
                return [ev.index]

        if not state.energy_attached:
            attach = self._attach_option(state)
            if attach is not None:
                return [attach.index]

        play = self._best_play_option(state)
        if play is not None:
            return [play.index]

        if me.active is not None:
            if phase == "late" and self._would_die_to_counter(state):
                pass  # Fall through to end — avoid a suicidal attack.
            else:
                best_atk = self._find_best_prize_attack(state)
                if best_atk is not None:
                    return [best_atk.index]

        end = state.end_options()
        if end:
            return [end[0].index]
        return self._safe_default(state)


class TwoHKOBenchAgent(PrizePhaseAgent):
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
        """Find an attack that doesn't leave us dead to counter."""
        if not self._would_die_to_counter(state):
            return self._find_best_prize_attack(state)
        return None

    # --- bench protection ---------------------------------------------------
    def _bench_hp_safe(self, state: GameState, hp: int) -> bool:
        """Check if a Pokemon with this HP is safe on bench vs known threats."""
        if hp <= 30:
            return False
        if hp <= self.SPREAD_HP_THRESHOLD and state.turn >= 4:
            return False
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
        scored.sort(key=lambda x: (-x[1], -x[0]))
        n = max(state.select.min_count, min(state.select.max_count, len(scored)))
        return [o.index for _, _, o in scored[:n]]

    # --- phase-gated MAIN with 2HKO ----------------------------------------
    def _decide_main(self, state: GameState) -> list[int]:
        phase = self._detect_phase(state)
        me = state.me

        lethal = self._find_lethal_attack_prize_scored(state)
        if lethal is not None:
            return [lethal.index]

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

        if phase != "late":
            ev = self._best_evolve_option(state)
            if ev is not None:
                return [ev.index]

        if not state.energy_attached:
            attach = self._attach_option(state)
            if attach is not None:
                return [attach.index]

        play = self._best_play_option(state)
        if play is not None:
            return [play.index]

        if me.active is not None:
            if phase == "early":
                best_atk = self._find_best_prize_attack(state)
                if best_atk is not None:
                    return [best_atk.index]
            else:
                safe = self._safe_attack_available(state)
                if safe is not None:
                    return [safe.index]
                retreat = self._retreat_option(state)
                if retreat is not None:
                    return [retreat.index]

        end = state.end_options()
        if end:
            return [end[0].index]
        return self._safe_default(state)
