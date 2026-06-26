"""Tests for the exp002 LucarioHeuristicAgent (mock CardDB, no engine)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from pokemon.card_db import (
    CT_BASIC_ENERGY,
    CT_ITEM,
    CT_POKEMON,
    CT_SUPPORTER,
    CT_TOOL,
    FIGHTING,
    METAL,
    WATER,
    AttackInfo,
    CardDB,
    CardInfo,
)
from pokemon.state import (
    CTX_DAMAGE,
    CTX_IS_FIRST,
    CTX_MAIN,
    CTX_SETUP_ACTIVE,
    OPT_ATTACH,
    OPT_ATTACK,
    OPT_CARD,
    OPT_END,
    OPT_PLAY,
    OPT_RETREAT,
    SEL_ATTACK,
    SEL_CARD,
    SEL_COUNT,
    SEL_MAIN,
    SEL_YES_NO,
)

# --- load the agent class from the workspace experiment ----------------------
_AGENT_PATH = Path("workspace/exp002_lucario_heuristic/agent.py")


def _load_agent_cls():
    spec = importlib.util.spec_from_file_location("_exp002_agent", str(_AGENT_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # find the Agent subclass
    from pokemon.agent import RuleBasedAgent

    for _, cls in vars(mod).items():
        if isinstance(cls, type) and issubclass(cls, RuleBasedAgent) and cls is not RuleBasedAgent:
            return cls
    raise RuntimeError("LucarioHeuristicAgent not found")


LucarioHeuristicAgent = _load_agent_cls()


# --- mock card database ------------------------------------------------------
def _info(cid, name, **kw):
    defaults = dict(
        id=cid,
        name=name,
        card_type=CT_POKEMON,
        card_type_name="pokemon",
        hp=0,
        energy_type=None,
        weakness=None,
        resistance=None,
        retreat_cost=1,
        is_basic=False,
        is_stage1=False,
        is_stage2=False,
        is_ex=False,
        is_mega_ex=False,
        is_tera=False,
        is_ace_spec=False,
        evolves_from=None,
        attack_ids=[],
    )
    defaults.update(kw)
    return CardInfo(**defaults)


def _mock_db() -> CardDB:
    cards = {
        677: _info(
            677,
            "Riolu",
            hp=80,
            energy_type=FIGHTING,
            weakness=5,
            retreat_cost=2,
            is_basic=True,
            attack_ids=[981],
        ),
        678: _info(
            678,
            "Mega Lucario ex",
            hp=340,
            energy_type=FIGHTING,
            weakness=5,
            retreat_cost=2,
            is_stage1=True,
            is_mega_ex=True,
            is_ex=True,
            attack_ids=[982, 983],
        ),
        722: _info(
            722,
            "Snover",
            hp=90,
            energy_type=WATER,
            weakness=METAL,
            retreat_cost=3,
            is_basic=True,
            attack_ids=[1044, 1045],
        ),
        723: _info(
            723,
            "Mega Abomasnow ex",
            hp=350,
            energy_type=WATER,
            weakness=METAL,
            retreat_cost=4,
            is_stage1=True,
            is_mega_ex=True,
            is_ex=True,
            attack_ids=[1046, 1047],
        ),
        999: _info(
            999,
            "BigBasic",
            hp=120,
            energy_type=FIGHTING,
            weakness=5,
            retreat_cost=1,
            is_basic=True,
            attack_ids=[],
        ),
        1145: _info(1145, "Mega Signal", card_type=CT_ITEM, card_type_name="item"),
        1158: _info(
            1158, "Maximum Belt", card_type=CT_TOOL, card_type_name="tool", is_ace_spec=True
        ),
        1205: _info(1205, "Cyrano", card_type=CT_SUPPORTER, card_type_name="supporter"),
        1227: _info(
            1227, "Lillie's Determination", card_type=CT_SUPPORTER, card_type_name="supporter"
        ),
        1235: _info(1235, "Waitress", card_type=CT_SUPPORTER, card_type_name="supporter"),
        6: _info(
            6,
            "Basic F Energy",
            card_type=CT_BASIC_ENERGY,
            card_type_name="basic_energy",
            energy_type=FIGHTING,
        ),
        3: _info(
            3,
            "Basic W Energy",
            card_type=CT_BASIC_ENERGY,
            card_type_name="basic_energy",
            energy_type=WATER,
        ),
    }
    attacks = {
        981: AttackInfo(981, "Accelerating Stab", "x", 30, [FIGHTING]),
        982: AttackInfo(982, "Aura Jab", "x", 130, [FIGHTING]),
        983: AttackInfo(983, "Mega Brave", "x", 270, [FIGHTING, FIGHTING]),
        1044: AttackInfo(1044, "Beat", "x", 10, [WATER]),
        1045: AttackInfo(1045, "Icy Snow", "x", 30, [WATER, WATER]),
        1047: AttackInfo(1047, "Frost Barrier", "x", 200, [WATER, WATER, WATER]),
    }
    return CardDB(cards=cards, attacks=attacks, source="mock")


def _agent(deck=None):
    return LucarioHeuristicAgent(deck=deck or [677] * 4 + [678] * 4 + [6] * 52, card_db=_mock_db())


# --- mock observation builders -----------------------------------------------
def _pokemon(id=678, hp=340, max_hp=340, energies=None, player_index=0):
    return {
        "id": id,
        "serial": 1,
        "playerIndex": player_index,
        "hp": hp,
        "maxHp": max_hp,
        "appearThisTurn": False,
        "energies": energies or [],
        "energyCards": [],
        "tools": [],
        "preEvolution": [],
    }


def _player(
    active=None,
    bench=None,
    hand=None,
    hand_count=None,
    deck_count=40,
    prize=6,
    bench_max=5,
    player_index=0,
):
    return {
        "active": [active] if active is not None else [],
        "bench": bench or [],
        "benchMax": bench_max,
        "deckCount": deck_count,
        "discard": [],
        "prize": [None] * prize,
        "handCount": hand_count if hand_count is not None else len(hand or []),
        "hand": [
            {"id": c, "serial": i, "playerIndex": player_index} for i, c in enumerate(hand or [])
        ]
        if hand is not None
        else None,
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
    }


def _obs(your_index=0, me=None, opp=None, select=None, energy_attached=False, result=-1):
    me = me or _player()
    opp = opp or _player(player_index=1)
    players = [me, opp] if your_index == 0 else [opp, me]
    return {
        "select": select,
        "logs": [],
        "current": {
            "turn": 3,
            "turnActionCount": 0,
            "yourIndex": your_index,
            "firstPlayer": 0,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": energy_attached,
            "retreated": False,
            "result": result,
            "stadium": [],
            "looking": None,
            "players": players,
        },
        "search_begin_input": "abc",
    }


def _sel(options, type=SEL_MAIN, context=CTX_MAIN, min_count=1, max_count=1, remain_energy_cost=0):
    return {
        "type": type,
        "context": context,
        "minCount": min_count,
        "maxCount": max_count,
        "remainDamageCounter": 0,
        "remainEnergyCost": remain_energy_cost,
        "option": options,
        "deck": None,
        "contextCard": None,
        "effect": None,
    }


# --- tests -------------------------------------------------------------------
class TestAgentInterface:
    def test_deck_selection(self):
        agent = _agent(deck=list(range(1, 61)))
        assert agent({"select": None}) == list(range(1, 61))

    def test_returns_empty_on_empty_options(self):
        agent = _agent()
        obs = _obs(select=_sel([], min_count=0, max_count=0))
        assert agent(obs) == []


class TestYesNo:
    def test_is_first_picks_yes(self):
        agent = _agent()
        sel = _sel([{"type": 1}, {"type": 2}], type=SEL_YES_NO, context=CTX_IS_FIRST)
        assert agent(_obs(select=sel)) == [0]  # YES


class TestCount:
    def test_picks_highest_number(self):
        agent = _agent()
        opts = [{"type": 0, "number": 1}, {"type": 0, "number": 5}, {"type": 0, "number": 3}]
        sel = _sel(opts, type=SEL_COUNT, min_count=1, max_count=1)
        assert agent(_obs(select=sel)) == [1]  # 5


class TestMainAttack:
    def test_lethal_attack_chosen_over_end(self):
        agent = _agent()
        me = _player(active=_pokemon(678, hp=340, energies=[FIGHTING, FIGHTING]))
        opp = _player(active=_pokemon(722, hp=90, player_index=1), player_index=1)
        opts = [{"type": OPT_ATTACK, "attackId": 982}, {"type": OPT_END}]
        obs = _obs(me=me, opp=opp, select=_sel(opts))
        # Aura Jab 130 vs Snover 90 -> lethal
        assert agent(obs) == [0]

    def test_max_damage_when_not_lethal(self):
        agent = _agent()
        me = _player(active=_pokemon(678, hp=340, energies=[FIGHTING, FIGHTING]))
        opp = _player(active=_pokemon(723, hp=350, player_index=1), player_index=1)
        opts = [
            {"type": OPT_ATTACK, "attackId": 982},
            {"type": OPT_ATTACK, "attackId": 983},
            {"type": OPT_END},
        ]
        obs = _obs(me=me, opp=opp, select=_sel(opts))
        # 130 and 270 both < 350; pick Mega Brave (270) at index 1
        assert agent(obs) == [1]

    def test_end_turn_when_no_attack(self):
        agent = _agent()
        me = _player(active=_pokemon(678, hp=340, energies=[]))
        opp = _player(active=_pokemon(723, hp=350, player_index=1), player_index=1)
        obs = _obs(me=me, opp=opp, select=_sel([{"type": OPT_END}]), energy_attached=True)
        assert agent(obs) == [0]


class TestMainAttach:
    def test_attaches_when_energy_unused(self):
        agent = _agent()
        me = _player(active=_pokemon(678, hp=340, energies=[]), hand=[6])
        opp = _player(active=_pokemon(723, hp=350, player_index=1), player_index=1)
        opts = [{"type": OPT_ATTACH}, {"type": OPT_END}]
        obs = _obs(me=me, opp=opp, select=_sel(opts), energy_attached=False)
        assert agent(obs) == [0]  # ATTACH


class TestMainRetreat:
    def test_retreats_when_active_low(self):
        agent = _agent()
        me = _player(
            active=_pokemon(677, hp=20, max_hp=80, energies=[FIGHTING, FIGHTING]),
            bench=[_pokemon(678, hp=340, energies=[FIGHTING], player_index=0)],
        )
        opp = _player(active=_pokemon(723, hp=350, player_index=1), player_index=1)
        opts = [{"type": OPT_RETREAT}, {"type": OPT_END}]
        obs = _obs(me=me, opp=opp, select=_sel(opts), energy_attached=True)
        assert agent(obs) == [0]  # RETREAT

    def test_no_retreat_when_no_bench(self):
        agent = _agent()
        me = _player(active=_pokemon(677, hp=20, max_hp=80, energies=[FIGHTING, FIGHTING]))
        opp = _player(active=_pokemon(723, hp=350, player_index=1), player_index=1)
        opts = [{"type": OPT_RETREAT}, {"type": OPT_END}]
        obs = _obs(me=me, opp=opp, select=_sel(opts), energy_attached=True)
        # no benched attacker -> should not retreat; falls through to END
        assert agent(obs) == [1]


class TestMainPlay:
    def test_plays_supporter_when_hand_low(self):
        agent = _agent()
        me = _player(
            active=_pokemon(678, hp=340, energies=[FIGHTING, FIGHTING]), hand=[1227], hand_count=1
        )
        opp = _player(active=_pokemon(723, hp=350, player_index=1), player_index=1)
        opts = [{"type": OPT_PLAY, "index": 0}, {"type": OPT_END}]
        obs = _obs(me=me, opp=opp, select=_sel(opts), energy_attached=True)
        # not lethal (270<350), supporter with hand_count=1 -> play it
        assert agent(obs) == [0]


class TestAttackSelect:
    def test_picks_lethal_attack(self):
        agent = _agent()
        me = _player(active=_pokemon(678, hp=340, energies=[FIGHTING, FIGHTING]))
        opp = _player(active=_pokemon(722, hp=90, player_index=1), player_index=1)
        opts = [{"type": OPT_ATTACK, "attackId": 983}, {"type": OPT_ATTACK, "attackId": 982}]
        sel = _sel(opts, type=SEL_ATTACK, context=CTX_MAIN)
        obs = _obs(me=me, opp=opp, select=sel)
        # both lethal; lethal scan returns first lethal (983 at index 0)
        assert agent(obs) == [0]


class TestCardSelect:
    def test_setup_active_picks_highest_hp(self):
        agent = _agent()
        me = _player(active=None, hand=[677, 999])
        opp = _player(player_index=1)
        opts = [
            {"type": OPT_CARD, "area": 2, "index": 0, "playerIndex": 0},  # Riolu hp 80
            {"type": OPT_CARD, "area": 2, "index": 1, "playerIndex": 0},  # BigBasic hp 120
        ]
        sel = _sel(opts, type=SEL_CARD, context=CTX_SETUP_ACTIVE)
        obs = _obs(me=me, opp=opp, select=sel)
        assert agent(obs) == [1]  # higher HP

    def test_damage_targets_lowest_hp_opponent(self):
        agent = _agent()
        me = _player(active=_pokemon(678, hp=340))
        opp = _player(
            active=_pokemon(723, hp=350, player_index=1),
            bench=[_pokemon(722, hp=90, player_index=1)],
            player_index=1,
        )
        opts = [
            {"type": OPT_CARD, "area": 4, "index": 0, "playerIndex": 1},  # opp active hp 350
            {"type": OPT_CARD, "area": 5, "index": 0, "playerIndex": 1},  # opp bench hp 90
        ]
        sel = _sel(opts, type=SEL_CARD, context=CTX_DAMAGE, min_count=1, max_count=1)
        obs = _obs(me=me, opp=opp, select=sel)
        assert agent(obs) == [1]  # lowest HP target


class TestRobustness:
    def test_unknown_select_type_falls_back_safely(self):
        agent = _agent()
        opts = [{"type": 200, "number": 1}, {"type": 200, "number": 2}]
        sel = _sel(opts, type=99, context=99, min_count=1, max_count=1)
        obs = _obs(select=sel)
        result = agent(obs)
        assert len(result) == 1 and 0 <= result[0] < len(opts)

    def test_never_raises_on_malformed(self):
        agent = _agent()
        # missing current / select fields -> should fall back, not raise
        assert agent({"select": {"option": [], "minCount": 0, "maxCount": 0}, "current": {}}) == []
