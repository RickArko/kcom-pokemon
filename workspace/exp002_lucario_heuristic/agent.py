"""exp002 — Lucario heuristic agent.

A state-aware rule-based agent for the Mega Lucario ex deck.  It parses the full
``cg`` observation via :mod:`pokemon.state`, looks up card/attack data via
:mod:`pokemon.card_db`, and selects actions with these priorities (from the
experimentation strategy):

    1. Attack — lethal check -> damage max -> type advantage (weakness x2)
    2. Retreat — if active HP <= 60 and a benched attacker is ready
    3. Energy — attach to active first, then bench setup
    4. Setup — evolve (toward Mega), play supporters for draw, bench basics

Setup actions are taken **before** attacking because attacking ends the turn.
The agent handles every select type (MAIN, YES_NO, CARD, ENERGY, ATTACK, COUNT,
EVOLVE) and falls back to a safe valid pick on any unexpected state, so it never
forfeits by raising an exception.

Deck: Mega Lucario ex (Fighting).  See ``deck.csv``.
"""

from __future__ import annotations

import logging
import os

from pokemon.agent import RuleBasedAgent
from pokemon.card_db import (
    CT_ITEM,
    CT_STADIUM,
    CT_SUPPORTER,
    CT_TOOL,
    CardDB,
    get_card_db,
)
from pokemon.state import (
    CTX_ACTIVATE,
    CTX_ATTACH_FROM,
    CTX_COIN_HEAD,
    CTX_DAMAGE,
    CTX_DAMAGE_COUNTER,
    CTX_DAMAGE_COUNTER_ANY,
    CTX_FIRST_EFFECT,
    CTX_HEAL,
    CTX_IS_FIRST,
    CTX_MORE_DEVOLVE,
    CTX_MULLIGAN,
    CTX_REMOVE_DAMAGE_COUNTER,
    CTX_SETUP_ACTIVE,
    CTX_SETUP_BENCH,
    CTX_SWITCH,
    CTX_TO_ACTIVE,
    GameState,
    OptionInfo,
    PokemonState,
    can_pay_cost,
    effective_damage,
    parse_obs,
)

logger = logging.getLogger(__name__)

# AreaType ints (mirror cg.api.AreaType) used to resolve option targets.
_AREA_ACTIVE = 4
_AREA_BENCH = 5
_AREA_HAND = 2


class LucarioHeuristicAgent(RuleBasedAgent):
    """Heuristic agent tuned for the Mega Lucario ex deck."""

    RETREAT_HP_THRESHOLD = 60
    LOW_HAND_THRESHOLD = 3
    ACTIVE_ENERGY_SATURATION = 2  # attach to active until this many energies

    def __init__(
        self, deck: list[int] | None = None, random_seed: int = 42, card_db: CardDB | None = None
    ):
        super().__init__(deck=deck, random_seed=random_seed)
        self._card_db: CardDB | None = card_db

    # --- card db ----------------------------------------------------------
    def _db(self) -> CardDB:
        if self._card_db is None:
            self._card_db = get_card_db()
        return self._card_db

    # --- main dispatch ----------------------------------------------------
    def _act(self, obs: dict) -> list[int]:
        try:
            state = parse_obs(obs)
            if state is None:
                return self._fallback_raw(obs)
            sel = state.select
            if sel.is_yes_no:
                return self._decide_yes_no(state)
            if sel.is_count:
                return self._decide_count(state)
            if sel.is_attack_select:
                return self._decide_attack(state)
            if sel.is_energy_select:
                return self._decide_energy(state)
            if sel.is_evolve_select:
                return self._decide_evolve(state)
            if sel.is_card_select:
                return self._decide_card(state)
            if sel.is_main:
                return self._decide_main(state)
            return self._safe_default(state)
        except Exception as e:  # noqa: BLE001 - never forfeit on a heuristic error
            logger.debug("heuristic error: %s; falling back", e)
            return self._fallback_raw(obs)

    # --- safe fallbacks ---------------------------------------------------
    def _safe_default(self, state: GameState) -> list[int]:
        opts = state.select.options
        if not opts:
            return []
        if state.select.min_count <= 1:
            for o in opts:
                if o.is_end:
                    return [o.index]
        n = (
            max(1, state.select.min_count)
            if state.select.min_count > 0
            else min(1, state.select.max_count)
        )
        n = min(n, len(opts))
        return [opts[i].index for i in range(n)]

    def _fallback_raw(self, obs: dict) -> list[int]:
        sel = obs.get("select") or {}
        raw_opts = sel.get("option") or []
        if not raw_opts:
            flat = obs.get("options") or []
            if flat:
                n = max(1, int(obs.get("minCount", 1) or 1))
                return [int(x) for x in flat[: min(n, len(flat))]]
            return []
        min_count = int(sel.get("minCount", 1) or 1)
        n = min(max(1, min_count), len(raw_opts))
        for i, o in enumerate(raw_opts[:n]):
            if isinstance(o, dict) and o.get("type") == 14:  # OPT_END
                return [i]
        return list(range(n))

    # --- YES_NO -----------------------------------------------------------
    def _decide_yes_no(self, state: GameState) -> list[int]:
        ctx = state.select.context
        yes = state.yes_options()
        no = state.no_options()
        pick_yes = [yes[0].index] if yes else None
        pick_no = [no[0].index] if no else None
        if ctx == CTX_IS_FIRST:
            return pick_yes or self._safe_default(state)  # go first
        if ctx == CTX_MULLIGAN:
            return pick_yes or self._safe_default(state)  # redraw for a basic
        if ctx == CTX_COIN_HEAD:
            return pick_yes or self._safe_default(state)  # choose heads
        if ctx == CTX_MORE_DEVOLVE:
            return pick_no or pick_yes or self._safe_default(state)  # stop devolving
        if ctx in (CTX_ACTIVATE, CTX_FIRST_EFFECT):
            return pick_yes or self._safe_default(state)  # activate beneficial effects
        return pick_yes or pick_no or self._safe_default(state)

    # --- COUNT ------------------------------------------------------------
    def _decide_count(self, state: GameState) -> list[int]:
        opts = state.select.options
        if not opts:
            return []
        # draw max / place max damage counters / remove max -> highest number
        best = max(opts, key=lambda o: o.count if o.count is not None else -1)
        return [best.index]

    # --- ATTACK (select.type == ATTACK) -----------------------------------
    def _decide_attack(self, state: GameState) -> list[int]:
        best = self._pick_best_attack(state)
        if best is not None:
            return [best.index]
        opts = state.select.options
        return [opts[0].index] if opts else []

    # --- ENERGY -----------------------------------------------------------
    def _decide_energy(self, state: GameState) -> list[int]:
        sel = state.select
        opts = sel.options
        if not opts:
            return []
        need = max(sel.min_count, sel.remain_energy_cost)
        need = max(1, need)
        need = min(need, sel.max_count, len(opts))
        return [o.index for o in opts[:need]]

    # --- EVOLVE -----------------------------------------------------------
    def _decide_evolve(self, state: GameState) -> list[int]:
        opts = state.select.options
        if not opts:
            return []
        db = self._db()
        best, best_score = opts[0], -1
        for o in opts:
            cid = o.card_id or self._option_hand_card_id(state, o)
            info = db.get(cid) if cid is not None else None
            score = 0
            if info:
                if info.is_mega_ex:
                    score += 100
                if info.is_stage2:
                    score += 50
                if info.is_ex:
                    score += 20
                score += info.hp
            if score > best_score:
                best, best_score = o, score
        return [best.index]

    # --- CARD targeting ---------------------------------------------------
    def _decide_card(self, state: GameState) -> list[int]:
        ctx = state.select.context
        opts = state.select.options
        if not opts:
            return []
        if ctx == CTX_SETUP_ACTIVE:
            return [self._pick_highest_hp_hand_card(state, opts)]
        if ctx == CTX_SETUP_BENCH:
            return self._pick_bench_setup(state, opts)
        if ctx in (CTX_SWITCH, CTX_TO_ACTIVE):
            return [self._pick_switch_target(state, opts)]
        if ctx == CTX_ATTACH_FROM:
            return [self._pick_attach_target(state, opts)]
        if ctx in (CTX_DAMAGE, CTX_DAMAGE_COUNTER, CTX_DAMAGE_COUNTER_ANY):
            return [self._pick_damage_target(state, opts)]
        if ctx in (CTX_REMOVE_DAMAGE_COUNTER, CTX_HEAL):
            return [self._pick_heal_target(state, opts)]
        return self._safe_default(state)

    # --- MAIN -------------------------------------------------------------
    def _decide_main(self, state: GameState) -> list[int]:
        # 1. Lethal attack wins now — take it before anything else.
        lethal = self._find_lethal_attack(state)
        if lethal is not None:
            return [lethal.index]

        me = state.me
        # 2. Retreat if active is in danger and a benched attacker is ready.
        if me.active is not None and me.active.hp <= self.RETREAT_HP_THRESHOLD:
            retreat = self._retreat_option(state)
            if retreat is not None:
                return [retreat.index]

        # 3. Evolve toward Mega / Stage 2 — high-value setup.
        ev = self._best_evolve_option(state)
        if ev is not None:
            return [ev.index]

        # 4. Attach energy for the turn (to active first, then bench setup).
        if not state.energy_attached:
            attach = self._attach_option(state)
            if attach is not None:
                return [attach.index]

        # 5. Play a supporter / item / basic to develop the board.
        play = self._best_play_option(state)
        if play is not None:
            return [play.index]

        # 6. Attack with the highest affordable damaging attack.
        best_atk = self._find_best_damage_attack(state)
        if best_atk is not None:
            return [best_atk.index]

        # 7. End the turn.
        end = state.end_options()
        if end:
            return [end[0].index]
        return self._safe_default(state)

    # --- attack evaluation ------------------------------------------------
    def _pick_best_attack(self, state: GameState) -> OptionInfo | None:
        lethal = self._find_lethal_attack(state)
        if lethal is not None:
            return lethal
        return self._find_best_damage_attack(state)

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
            if atk is None or atk.damage <= 0:
                continue
            if not can_pay_cost(me_act.energies, atk.energies):
                continue
            if effective_damage(atk.damage, atk_type, opp_weak) >= opp.hp:
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
            if atk is None or atk.damage <= 0:
                continue
            if not can_pay_cost(me_act.energies, atk.energies):
                continue
            dmg = effective_damage(atk.damage, atk_type, opp_weak)
            if dmg > best_dmg:
                best, best_dmg = o, dmg
        return best

    def _weakness_of(self, pokemon: PokemonState, db: CardDB) -> int | None:
        info = db.get(pokemon.id)
        return info.weakness if info else None

    # --- MAIN option finders ----------------------------------------------
    def _retreat_option(self, state: GameState) -> OptionInfo | None:
        ropts = state.retreat_options()
        if not ropts:
            return None
        me_act = state.my_active
        if me_act is None:
            return None
        db = self._db()
        info = db.get(me_act.id)
        retreat_cost = info.retreat_cost if info else 1
        if me_act.energy_count < retreat_cost:
            return None
        if me_act.energy_count == 0:
            return None
        if not state.me.has_benched:
            return None
        if state.me.benched_attacker(min_hp_ratio=0.5) is None:
            return None
        return ropts[0]

    def _best_evolve_option(self, state: GameState) -> OptionInfo | None:
        eopts = state.evolve_options()
        if not eopts:
            return None
        db = self._db()
        best, best_score = None, -1
        for o in eopts:
            cid = o.card_id or self._option_hand_card_id(state, o)
            info = db.get(cid) if cid is not None else None
            score = 0
            if info:
                if info.is_mega_ex:
                    score += 100
                if info.is_stage2:
                    score += 50
                if info.is_ex:
                    score += 20
                score += info.hp
            if score > best_score:
                best, best_score = o, score
        return best

    def _attach_option(self, state: GameState) -> OptionInfo | None:
        aopts = state.attach_options()
        return aopts[0] if aopts else None

    def _best_play_option(self, state: GameState) -> OptionInfo | None:
        popts = state.play_options()
        if not popts:
            return None
        db = self._db()
        me = state.me
        supporters, items, tools, basics, evolutions = [], [], [], [], []
        for o in popts:
            cid = self._play_card_id(state, o)
            info = db.get(cid) if cid is not None else None
            if info is None:
                continue
            if info.card_type == CT_SUPPORTER:
                supporters.append(o)
            elif info.card_type in (CT_ITEM, CT_STADIUM):
                items.append(o)
            elif info.card_type == CT_TOOL:
                tools.append(o)
            elif info.is_pokemon:
                if info.is_basic:
                    basics.append(o)
                else:
                    evolutions.append(o)
        if evolutions:
            return evolutions[0]
        if supporters and me.hand_count <= self.LOW_HAND_THRESHOLD:
            return supporters[0]
        if items:
            return items[0]
        if tools and me.active is not None:
            return tools[0]
        if basics and not me.bench_full:
            return basics[0]
        if supporters:
            return supporters[0]
        if basics:
            return basics[0]
        return None

    # --- CARD target pickers ----------------------------------------------
    def _pick_highest_hp_hand_card(self, state: GameState, opts: list[OptionInfo]) -> int:
        db = self._db()
        best, best_hp = opts[0], -1
        for o in opts:
            cid = self._option_hand_card_id(state, o)
            info = db.get(cid) if cid is not None else None
            hp = info.hp if info and info.is_pokemon else 0
            if hp > best_hp:
                best, best_hp = o, hp
        return best.index

    def _pick_bench_setup(self, state: GameState, opts: list[OptionInfo]) -> list[int]:
        db = self._db()
        scored: list[tuple[int, OptionInfo]] = []
        for o in opts:
            cid = self._option_hand_card_id(state, o)
            info = db.get(cid) if cid is not None else None
            hp = info.hp if info and info.is_pokemon else 0
            scored.append((hp, o))
        scored.sort(key=lambda x: -x[0])
        n = max(state.select.min_count, min(state.select.max_count, len(scored)))
        return [o.index for _, o in scored[:n]]

    def _pick_switch_target(self, state: GameState, opts: list[OptionInfo]) -> int:
        best, best_score = opts[0], (-1, -1)
        for o in opts:
            p = self._resolve_pokemon(state, o)
            if p is None:
                continue
            score = (p.energy_count, p.hp)
            if score > best_score:
                best, best_score = o, score
        return best.index

    def _pick_attach_target(self, state: GameState, opts: list[OptionInfo]) -> int:
        me = state.me
        if me.active is not None and me.active.energy_count < self.ACTIVE_ENERGY_SATURATION:
            for o in opts:
                p = self._resolve_pokemon(state, o)
                if p is not None and p is me.active:
                    return o.index
        benched = [(o, self._resolve_pokemon(state, o)) for o in opts]
        benched = [(o, p) for o, p in benched if p is not None and p is not me.active]
        if benched:
            best = max(benched, key=lambda x: (x[1].energy_count, x[1].hp))
            return best[0].index
        return opts[0].index

    def _pick_damage_target(self, state: GameState, opts: list[OptionInfo]) -> int:
        best = None
        for o in opts:
            if o.player_index is None or o.player_index == state.your_index:
                continue
            p = self._resolve_pokemon(state, o)
            if p is None:
                continue
            if best is None or p.hp < best[1].hp:
                best = (o, p)
        return best[0].index if best else opts[0].index

    def _pick_heal_target(self, state: GameState, opts: list[OptionInfo]) -> int:
        best = None
        for o in opts:
            if o.player_index is None or o.player_index != state.your_index:
                continue
            p = self._resolve_pokemon(state, o)
            if p is None:
                continue
            if best is None or p.damage > best[1].damage:
                best = (o, p)
        return best[0].index if best else opts[0].index

    # --- option -> pokemon / card resolution ------------------------------
    def _resolve_pokemon(self, state: GameState, opt: OptionInfo) -> PokemonState | None:
        if opt.area is None or opt.player_index is None:
            return None
        side = state.me if opt.player_index == state.your_index else state.opp
        if opt.area == _AREA_ACTIVE:
            return side.active
        if opt.area == _AREA_BENCH:
            idx = opt.area_index
            if idx is not None and 0 <= idx < len(side.bench):
                return side.bench[idx]
        return None

    def _option_hand_card_id(self, state: GameState, opt: OptionInfo) -> int | None:
        if opt.area == _AREA_HAND:
            idx = opt.area_index
            is_mine = opt.player_index == state.your_index if opt.player_index is not None else True
            side = state.me if is_mine else state.opp
            if idx is not None and 0 <= idx < len(side.hand):
                return side.hand[idx]
        if opt.card_id is not None:
            return opt.card_id
        return None

    def _play_card_id(self, state: GameState, opt: OptionInfo) -> int | None:
        if opt.hand_index is not None and 0 <= opt.hand_index < len(state.me.hand):
            return state.me.hand[opt.hand_index]
        return opt.card_id


# --- Kaggle submission entry point ------------------------------------------
# The local gauntlet instantiates ``LucarioHeuristicAgent`` directly (it discovers
# the Agent subclass).  Kaggle instead calls the module-level ``agent`` function
# defined here.  ``make build-submit`` copies this file to ``main.py``.

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
