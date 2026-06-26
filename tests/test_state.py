"""Tests for pokemon.state (observation parsing + combat helpers)."""

from __future__ import annotations

import pytest

from pokemon.card_db import (
    COLORLESS,
    DARKNESS,
    FIGHTING,
    FIRE,
    PSYCHIC,
    RAINBOW,
    TEAM_ROCKET,
    WATER,
)
from pokemon.state import (
    CTX_IS_FIRST,
    CTX_MAIN,
    OPT_ATTACK,
    OPT_END,
    OPT_NO,
    OPT_PLAY,
    OPT_YES,
    SEL_COUNT,
    SEL_MAIN,
    SEL_YES_NO,
    GameState,
    can_pay_cost,
    effective_damage,
    parse_obs,
    weakness_multiplier,
)


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
    active=None, bench=None, hand=None, hand_count=None, deck_count=40, prize=6, bench_max=5
):
    return {
        "active": [active] if active is not None else [],
        "bench": bench or [],
        "benchMax": bench_max,
        "deckCount": deck_count,
        "discard": [],
        "prize": [None] * prize,
        "handCount": hand_count if hand_count is not None else len(hand or []),
        "hand": [{"id": c, "serial": i, "playerIndex": 0} for i, c in enumerate(hand or [])]
        if hand is not None
        else None,
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
    }


def _state(your_index=0, me=None, opp=None, select=None, turn=3, result=-1, **state_kw):
    me = me or {}
    opp = opp or {}
    if "hand" in me and me.get("hand") is not None:
        me = {**me, "handCount": me.get("handCount", len(me["hand"]))}
    players = [me, opp] if your_index == 0 else [opp, me]
    return {
        "select": select or _main_select(),
        "logs": [],
        "current": {
            "turn": turn,
            "turnActionCount": 0,
            "yourIndex": your_index,
            "firstPlayer": 0,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "result": result,
            "stadium": [],
            "looking": None,
            "players": players,
        },
        "search_begin_input": "abc",
    }


def _main_select(options=None, min_count=1, max_count=1):
    return {
        "type": SEL_MAIN,
        "context": CTX_MAIN,
        "minCount": min_count,
        "maxCount": max_count,
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "option": options or [{"type": OPT_END}],
        "deck": None,
        "contextCard": None,
        "effect": None,
    }


class TestParseObs:
    def test_none_select_returns_none(self):
        assert parse_obs({"select": None}) is None
        assert parse_obs(None) is None

    def test_me_opp_orientation(self):
        me = _player(
            active=_pokemon(678, hp=340, energies=[FIGHTING]), hand=[677, 6], deck_count=40
        )
        opp = _player(active=_pokemon(722, hp=90, energies=[WATER], player_index=1), hand_count=6)
        obs = _state(your_index=0, me=me, opp=opp)
        st = parse_obs(obs)
        assert isinstance(st, GameState)
        assert st.your_index == 0
        assert st.me.active.id == 678
        assert st.me.active.energy_count == 1
        assert st.opp.active.id == 722
        assert st.me.hand == [677, 6]
        assert st.opp.hand == []  # opponent hand hidden
        assert st.opp.hand_count == 6

    def test_orientation_swaps_when_your_index_is_1(self):
        me = _player(active=_pokemon(678), hand=[677])
        opp = _player(active=_pokemon(722), hand_count=5)
        obs = _state(your_index=1, me=me, opp=opp)
        st = parse_obs(obs)
        assert st.your_index == 1
        assert st.me.active.id == 678
        assert st.opp.active.id == 722

    def test_prize_tracking(self):
        me = _player(prize=4)  # 2 prizes taken
        opp = _player(prize=5)  # 1 prize taken
        obs = _state(me=me, opp=opp)
        st = parse_obs(obs)
        assert st.me.prize_count == 4
        assert st.me.prizes_taken == 2
        assert st.opp.prizes_taken == 1
        assert st.prize_lead == 1

    def test_pokemon_damage(self):
        p = _pokemon(678, hp=200, max_hp=340)
        obs = _state(me=_player(active=p), opp=_player())
        st = parse_obs(obs)
        assert st.me.active.damage == 140
        assert st.me.active.hp_ratio == pytest.approx(200 / 340)
        assert not st.me.active.is_knocked_out

    def test_done_when_result_set(self):
        obs = _state(result=0)
        st = parse_obs(obs)
        assert st.is_done
        assert st.result == 0

    def test_option_parsing_attack_play_end(self):
        opts = [
            {"type": OPT_PLAY, "index": 2},
            {"type": OPT_ATTACK, "attackId": 983},
            {"type": OPT_END},
        ]
        obs = _state(select=_main_select(opts, min_count=1, max_count=1))
        st = parse_obs(obs)
        plays = st.play_options()
        atks = st.attack_options()
        ends = st.end_options()
        assert len(plays) == 1 and plays[0].hand_index == 2
        assert len(atks) == 1 and atks[0].attack_id == 983 and atks[0].is_attack
        assert len(ends) == 1 and ends[0].is_end

    def test_yes_no_select(self):
        sel = {
            "type": SEL_YES_NO,
            "context": CTX_IS_FIRST,
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "option": [{"type": OPT_YES}, {"type": OPT_NO}],
            "deck": None,
            "contextCard": None,
            "effect": None,
        }
        obs = _state(select=sel)
        st = parse_obs(obs)
        assert st.select.is_yes_no
        assert len(st.yes_options()) == 1
        assert len(st.no_options()) == 1

    def test_count_select(self):
        sel = {
            "type": SEL_COUNT,
            "context": CTX_MAIN,
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "option": [
                {"type": 0, "number": 1},
                {"type": 0, "number": 5},
                {"type": 0, "number": 3},
            ],
            "deck": None,
            "contextCard": None,
            "effect": None,
        }
        st = parse_obs(_state(select=sel))
        assert st.select.is_count
        assert [o.count for o in st.select.options] == [1, 5, 3]

    def test_bench_max(self):
        obs = _state(me=_player(bench_max=5), opp=_player())
        st = parse_obs(obs)
        assert st.me.bench_max == 5
        assert not st.me.bench_full

    def test_facedown_active_is_none(self):
        me = _player(active=None)
        opp = _player(active=None)
        obs = _state(me=me, opp=opp)
        st = parse_obs(obs)
        assert st.me.active is None
        assert st.opp.active is None


class TestCombatHelpers:
    def test_can_pay_exact(self):
        assert can_pay_cost([FIGHTING], [FIGHTING])
        assert can_pay_cost([FIGHTING, FIGHTING], [FIGHTING, FIGHTING])

    def test_can_pay_colorless_any(self):
        assert can_pay_cost([WATER], [COLORLESS])
        assert can_pay_cost([WATER, FIRE], [COLORLESS, COLORLESS])

    def test_can_pay_mixed(self):
        # Mega Brave {F}{F} with 2 fighting
        assert can_pay_cost([FIGHTING, FIGHTING], [FIGHTING, FIGHTING])
        # {F} + colorless with F + W
        assert can_pay_cost([FIGHTING, WATER], [FIGHTING, COLORLESS])

    def test_cannot_pay_missing_type(self):
        assert not can_pay_cost([WATER], [FIGHTING])
        assert not can_pay_cost([FIGHTING], [FIGHTING, FIGHTING])

    def test_rainbow_wildcard(self):
        assert can_pay_cost([RAINBOW], [FIGHTING])
        assert can_pay_cost([RAINBOW, RAINBOW], [FIRE, FIRE])
        assert can_pay_cost([RAINBOW, WATER], [FIRE, COLORLESS])

    def test_team_rocket_pays_psychic_or_darkness(self):
        assert can_pay_cost([TEAM_ROCKET], [PSYCHIC])
        assert can_pay_cost([TEAM_ROCKET], [DARKNESS])
        assert not can_pay_cost([TEAM_ROCKET], [FIGHTING])

    def test_weakness_multiplier(self):
        assert weakness_multiplier(FIGHTING, PSYCHIC) == 1  # no match
        assert weakness_multiplier(FIGHTING, FIGHTING) == 2  # match
        assert weakness_multiplier(None, FIGHTING) == 1
        assert weakness_multiplier(FIGHTING, None) == 1

    def test_effective_damage(self):
        assert effective_damage(130, FIGHTING, PSYCHIC) == 130  # no weakness
        assert effective_damage(130, FIGHTING, FIGHTING) == 260  # weakness x2
