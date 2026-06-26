"""Game state parser for Pokemon TCG observations.

Converts the raw ``cg`` observation dict (the nested structure returned by
``cg.game.battle_start`` / ``battle_select``) into convenient dataclass
objects oriented relative to the acting player: ``me`` is always the player
making the selection (``players[yourIndex]``) and ``opp`` is the other side.

The parser does **not** require the ``cg`` engine — it works on plain dicts, so
it is fully testable with mock observations.

Example
-------
>>> state = parse_obs(obs_dict)
>>> state.me.active.hp, state.opp.active.hp
(90, 150)
>>> [o.attack_id for o in state.attack_options()]
[1042]
"""

from __future__ import annotations

from dataclasses import dataclass

from pokemon.card_db import (
    COLORLESS,
    DARKNESS,
    PSYCHIC,
    RAINBOW,
    TEAM_ROCKET,
)

# --- SelectType / SelectContext / OptionType int constants -------------------
# Mirror cg.api enums (kept as ints to avoid importing the engine here).

# SelectType
SEL_MAIN = 0
SEL_CARD = 1
SEL_ATTACHED_CARD = 2
SEL_CARD_OR_ATTACHED = 3
SEL_ENERGY = 4
SEL_SKILL = 5
SEL_ATTACK = 6
SEL_EVOLVE = 7
SEL_COUNT = 8
SEL_YES_NO = 9
SEL_SPECIAL_CONDITION = 10

# SelectContext
CTX_MAIN = 0
CTX_SETUP_ACTIVE = 1
CTX_SETUP_BENCH = 2
CTX_SWITCH = 3
CTX_TO_ACTIVE = 4
CTX_TO_BENCH = 5
CTX_TO_FIELD = 6
CTX_TO_HAND = 7
CTX_DISCARD = 8
CTX_TO_DECK = 9
CTX_TO_DECK_BOTTOM = 10
CTX_TO_PRIZE = 11
CTX_NOT_MOVE = 12
CTX_DAMAGE_COUNTER = 13
CTX_DAMAGE_COUNTER_ANY = 14
CTX_DAMAGE = 15
CTX_REMOVE_DAMAGE_COUNTER = 16
CTX_HEAL = 17
CTX_EVOLVES_FROM = 18
CTX_EVOLVES_TO = 19
CTX_DEVOLVE = 20
CTX_ATTACH_FROM = 21
CTX_ATTACH_TO = 22
CTX_DETACH_FROM = 23
CTX_LOOK = 24
CTX_EFFECT_TARGET = 25
CTX_DISCARD_ENERGY_CARD = 26
CTX_DISCARD_TOOL_CARD = 27
CTX_SWITCH_ENERGY_CARD = 28
CTX_DISCARD_CARD_OR_ATTACHED = 29
CTX_DISCARD_ENERGY = 30
CTX_TO_HAND_ENERGY = 31
CTX_TO_DECK_ENERGY = 32
CTX_SWITCH_ENERGY = 33
CTX_SKILL_ORDER = 34
CTX_ATTACK = 35
CTX_DISABLE_ATTACK = 36
CTX_EVOLVE = 37
CTX_DRAW_COUNT = 38
CTX_DAMAGE_COUNTER_COUNT = 39
CTX_REMOVE_DAMAGE_COUNTER_COUNT = 40
CTX_IS_FIRST = 41
CTX_MULLIGAN = 42
CTX_ACTIVATE = 43
CTX_FIRST_EFFECT = 44
CTX_MORE_DEVOLVE = 45
CTX_COIN_HEAD = 46
CTX_AFFECT_SPECIAL_CONDITION = 47
CTX_RECOVER_SPECIAL_CONDITION = 48

# OptionType
OPT_NUMBER = 0
OPT_YES = 1
OPT_NO = 2
OPT_CARD = 3
OPT_TOOL_CARD = 4
OPT_ENERGY_CARD = 5
OPT_ENERGY = 6
OPT_PLAY = 7
OPT_ATTACH = 8
OPT_EVOLVE = 9
OPT_ABILITY = 10
OPT_DISCARD = 11
OPT_RETREAT = 12
OPT_ATTACK = 13
OPT_END = 14
OPT_SKILL = 15
OPT_SPECIAL_CONDITION = 16


@dataclass
class PokemonState:
    """A Pokemon in play (active or benched)."""

    id: int
    serial: int
    player_index: int
    hp: int
    max_hp: int
    energies: list[int]
    energy_cards: list[int]
    tools: list[int]
    pre_evolution: list[int]
    appear_this_turn: bool

    @property
    def energy_count(self) -> int:
        return len(self.energies)

    @property
    def damage(self) -> int:
        return max(0, self.max_hp - self.hp)

    @property
    def is_knocked_out(self) -> bool:
        return self.hp <= 0

    @property
    def hp_ratio(self) -> float:
        return self.hp / self.max_hp if self.max_hp else 0.0


@dataclass
class PlayerSide:
    """One player's visible board state (oriented as me/opp)."""

    active: PokemonState | None
    bench: list[PokemonState]
    hand: list[int]
    hand_count: int
    deck_count: int
    prize_count: int
    prizes_taken: int
    discard: list[int]
    bench_max: int
    poisoned: bool
    burned: bool
    asleep: bool
    paralyzed: bool
    confused: bool

    @property
    def has_benched(self) -> bool:
        return len(self.bench) > 0

    @property
    def bench_full(self) -> bool:
        return len(self.bench) >= self.bench_max

    @property
    def all_pokemon(self) -> list[PokemonState]:
        out: list[PokemonState] = []
        if self.active is not None:
            out.append(self.active)
        out.extend(self.bench)
        return out

    def benched_attacker(self, min_hp_ratio: float = 0.5) -> PokemonState | None:
        """Healthiest benched Pokemon above ``min_hp_ratio`` (prefers energy ready)."""
        candidates = [p for p in self.bench if p.hp_ratio >= min_hp_ratio]
        if not candidates:
            return None
        # Prefer one with the most energy (closest to attacking), then most HP.
        return max(candidates, key=lambda p: (p.energy_count, p.hp))


@dataclass
class OptionInfo:
    """A single selectable option, normalised from the raw Option dict."""

    index: int
    raw: dict
    type: int
    attack_id: int | None
    card_id: int | None
    hand_index: int | None
    area: int | None
    area_index: int | None
    player_index: int | None
    tool_index: int | None
    energy_index: int | None
    count: int | None
    in_play_area: int | None
    in_play_index: int | None
    serial: int | None
    special_condition_type: int | None

    is_attack: bool
    is_play: bool
    is_retreat: bool
    is_end: bool
    is_ability: bool
    is_attach: bool
    is_evolve: bool
    is_discard: bool
    is_yes: bool
    is_no: bool
    is_card: bool
    is_tool_card: bool
    is_energy_card: bool
    is_energy: bool
    is_number: bool


@dataclass
class SelectInfo:
    """The current selection prompt."""

    type: int
    context: int
    min_count: int
    max_count: int
    remain_damage_counter: int
    remain_energy_cost: int
    options: list[OptionInfo]
    deck: list[int] | None
    context_card_id: int | None
    effect_card_id: int | None

    @property
    def is_main(self) -> bool:
        return self.type == SEL_MAIN

    @property
    def is_yes_no(self) -> bool:
        return self.type == SEL_YES_NO

    @property
    def is_attack_select(self) -> bool:
        return self.type == SEL_ATTACK

    @property
    def is_card_select(self) -> bool:
        return self.type in (SEL_CARD, SEL_ATTACHED_CARD, SEL_CARD_OR_ATTACHED)

    @property
    def is_energy_select(self) -> bool:
        return self.type == SEL_ENERGY

    @property
    def is_count(self) -> bool:
        return self.type == SEL_COUNT

    @property
    def is_evolve_select(self) -> bool:
        return self.type == SEL_EVOLVE

    def by_type(self, opt_type: int) -> list[OptionInfo]:
        return [o for o in self.options if o.type == opt_type]


@dataclass
class GameState:
    """Full parsed game state, oriented to the acting player."""

    turn: int
    turn_action_count: int
    your_index: int
    first_player: int
    supporter_played: bool
    stadium_played: bool
    energy_attached: bool
    retreated: bool
    result: int
    stadium: list[int]
    me: PlayerSide
    opp: PlayerSide
    select: SelectInfo
    logs: list[dict]
    raw: dict
    search_begin_input: str | None

    @property
    def is_done(self) -> bool:
        return self.result != -1

    @property
    def prize_lead(self) -> int:
        """Positive = we are ahead on prizes taken."""
        return self.me.prizes_taken - self.opp.prizes_taken

    @property
    def my_active(self) -> PokemonState | None:
        return self.me.active

    @property
    def opp_active(self) -> PokemonState | None:
        return self.opp.active

    # --- option accessors ---
    def attack_options(self) -> list[OptionInfo]:
        return [o for o in self.select.options if o.is_attack]

    def play_options(self) -> list[OptionInfo]:
        return [o for o in self.select.options if o.is_play]

    def retreat_options(self) -> list[OptionInfo]:
        return [o for o in self.select.options if o.is_retreat]

    def attach_options(self) -> list[OptionInfo]:
        return [o for o in self.select.options if o.is_attach]

    def ability_options(self) -> list[OptionInfo]:
        return [o for o in self.select.options if o.is_ability]

    def evolve_options(self) -> list[OptionInfo]:
        return [o for o in self.select.options if o.is_evolve]

    def end_options(self) -> list[OptionInfo]:
        return [o for o in self.select.options if o.is_end]

    def yes_options(self) -> list[OptionInfo]:
        return [o for o in self.select.options if o.is_yes]

    def no_options(self) -> list[OptionInfo]:
        return [o for o in self.select.options if o.is_no]

    def card_options(self) -> list[OptionInfo]:
        return [o for o in self.select.options if o.is_card]


# --- parsing -----------------------------------------------------------------


def parse_obs(obs: dict | None) -> GameState | None:
    """Parse a raw cg observation dict into a :class:`GameState`.

    Returns ``None`` for the deck-selection observation (``select is None``).
    """
    if obs is None:
        return None
    select_raw = obs.get("select")
    if select_raw is None:
        return None

    current = obs.get("current") or {}
    your = int(current.get("yourIndex", 0))
    players = current.get("players") or [None, None]
    me_raw = players[your] if your < len(players) and players[your] is not None else {}
    opp_raw = players[1 - your] if len(players) > 1 and players[1 - your] is not None else {}

    return GameState(
        turn=int(current.get("turn", 0)),
        turn_action_count=int(current.get("turnActionCount", 0)),
        your_index=your,
        first_player=int(current.get("firstPlayer", -1)),
        supporter_played=bool(current.get("supporterPlayed", False)),
        stadium_played=bool(current.get("stadiumPlayed", False)),
        energy_attached=bool(current.get("energyAttached", False)),
        retreated=bool(current.get("retreated", False)),
        result=int(current.get("result", -1)),
        stadium=[c.get("id") for c in (current.get("stadium") or []) if c],
        me=_parse_side(me_raw),
        opp=_parse_side(opp_raw),
        select=_parse_select(select_raw),
        logs=list(obs.get("logs") or []),
        raw=obs,
        search_begin_input=obs.get("search_begin_input"),
    )


def _parse_pokemon(p: dict | None) -> PokemonState | None:
    if p is None:
        return None
    return PokemonState(
        id=int(p.get("id", 0)),
        serial=int(p.get("serial", 0)),
        player_index=int(p.get("playerIndex", 0)),
        hp=int(p.get("hp", 0)),
        max_hp=int(p.get("maxHp", 0)),
        energies=[int(e) for e in (p.get("energies") or [])],
        energy_cards=[c.get("id") for c in (p.get("energyCards") or []) if c],
        tools=[c.get("id") for c in (p.get("tools") or []) if c],
        pre_evolution=[c.get("id") for c in (p.get("preEvolution") or []) if c],
        appear_this_turn=bool(p.get("appearThisTurn", False)),
    )


def _parse_side(s: dict) -> PlayerSide:
    s = s or {}
    active_list = s.get("active") or []
    active = _parse_pokemon(active_list[0]) if active_list else None
    bench = [_parse_pokemon(p) for p in (s.get("bench") or []) if p]
    bench = [b for b in bench if b is not None]
    hand_raw = s.get("hand")
    hand = [c.get("id") for c in hand_raw if c] if hand_raw else []
    prize_list = s.get("prize") or []
    prize_count = len(prize_list)
    return PlayerSide(
        active=active,
        bench=bench,
        hand=hand,
        hand_count=int(s.get("handCount", len(hand))),
        deck_count=int(s.get("deckCount", 0)),
        prize_count=prize_count,
        prizes_taken=max(0, 6 - prize_count),
        discard=[c.get("id") for c in (s.get("discard") or []) if c],
        bench_max=int(s.get("benchMax", 5)),
        poisoned=bool(s.get("poisoned", False)),
        burned=bool(s.get("burned", False)),
        asleep=bool(s.get("asleep", False)),
        paralyzed=bool(s.get("paralyzed", False)),
        confused=bool(s.get("confused", False)),
    )


def _parse_option(i: int, o: dict) -> OptionInfo:
    t = int(o.get("type", -1))
    is_play = t == OPT_PLAY
    return OptionInfo(
        index=i,
        raw=o,
        type=t,
        attack_id=o.get("attackId"),
        card_id=o.get("cardId"),
        hand_index=int(o.get("index")) if is_play and o.get("index") is not None else None,
        area=o.get("area"),
        area_index=int(o.get("index")) if not is_play and o.get("index") is not None else None,
        player_index=o.get("playerIndex"),
        tool_index=o.get("toolIndex"),
        energy_index=o.get("energyIndex"),
        count=o.get("count") if o.get("count") is not None else o.get("number"),
        in_play_area=o.get("inPlayArea"),
        in_play_index=o.get("inPlayIndex"),
        serial=o.get("serial"),
        special_condition_type=o.get("specialConditionType"),
        is_attack=(t == OPT_ATTACK),
        is_play=is_play,
        is_retreat=(t == OPT_RETREAT),
        is_end=(t == OPT_END),
        is_ability=(t == OPT_ABILITY),
        is_attach=(t == OPT_ATTACH),
        is_evolve=(t == OPT_EVOLVE),
        is_discard=(t == OPT_DISCARD),
        is_yes=(t == OPT_YES),
        is_no=(t == OPT_NO),
        is_card=(t == OPT_CARD),
        is_tool_card=(t == OPT_TOOL_CARD),
        is_energy_card=(t == OPT_ENERGY_CARD),
        is_energy=(t == OPT_ENERGY),
        is_number=(t == OPT_NUMBER),
    )


def _parse_select(s: dict) -> SelectInfo:
    options = [_parse_option(i, o) for i, o in enumerate(s.get("option") or [])]
    deck_raw = s.get("deck")
    deck = [c.get("id") for c in deck_raw if c] if deck_raw else None
    ctx_card = s.get("contextCard")
    effect = s.get("effect")
    return SelectInfo(
        type=int(s.get("type", 0)),
        context=int(s.get("context", 0)),
        min_count=int(s.get("minCount", 0)),
        max_count=int(s.get("maxCount", 0)),
        remain_damage_counter=int(s.get("remainDamageCounter", 0)),
        remain_energy_cost=int(s.get("remainEnergyCost", 0)),
        options=options,
        deck=deck,
        context_card_id=ctx_card.get("id") if ctx_card else None,
        effect_card_id=effect.get("id") if effect else None,
    )


# --- combat helpers ----------------------------------------------------------


def can_pay_cost(available: list[int], cost: list[int]) -> bool:
    """Return True if ``available`` energies cover the attack ``cost``.

    Colorless (0) is paid by any energy.  Rainbow (10) counts as any type and
    Team Rocket (11) counts as Psychic or Darkness.
    """
    avail = list(available)
    for e in cost:
        if e == COLORLESS:
            continue
        if e in avail:
            avail.remove(e)
        elif RAINBOW in avail:
            avail.remove(RAINBOW)
        elif e in (PSYCHIC, DARKNESS) and TEAM_ROCKET in avail:
            avail.remove(TEAM_ROCKET)
        else:
            return False
    colorless_needed = sum(1 for e in cost if e == COLORLESS)
    return len(avail) >= colorless_needed


def weakness_multiplier(attacker_energy_type: int | None, defender_weakness: int | None) -> int:
    """Modern PTCG weakness is x2 when the attacker's type matches."""
    if attacker_energy_type is not None and defender_weakness is not None:
        if attacker_energy_type == defender_weakness:
            return 2
    return 1


def effective_damage(damage: int, attacker_type: int | None, defender_weakness: int | None) -> int:
    """Printed damage scaled by weakness."""
    return damage * weakness_multiplier(attacker_type, defender_weakness)
