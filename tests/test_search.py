"""Tests for pokemon.search (UCB1, leaf value, predictions — no engine)."""

from __future__ import annotations

import numpy as np
import pytest

from pokemon.card_db import CT_POKEMON, CardDB, CardInfo
from pokemon.search import (
    MCTSResult,
    _leaf_value,
    _Node,
    _prizes_taken,
    _resolve_action,
    _sample_pool,
    _ucb1,
    build_predictions,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


def _pokemon_dict(id=678, hp=340, max_hp=340):
    return {
        "id": id,
        "serial": 1,
        "playerIndex": 0,
        "hp": hp,
        "maxHp": max_hp,
        "appearThisTurn": False,
        "energies": [],
        "energyCards": [],
        "tools": [],
        "preEvolution": [],
    }


def _side(active=None, prize=6, bench=None, player_index=0):
    return {
        "active": [active] if active is not None else [],
        "bench": bench or [],
        "prize": [None] * prize,
        "handCount": 0,
        "hand": None,
        "deckCount": 40,
        "discard": [],
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
    }


def _obs(your_index=0, me=None, opp=None, result=-1):
    me = me or _side()
    opp = opp or _side(player_index=1)
    players = [me, opp] if your_index == 0 else [opp, me]
    return {
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 14}],
            "deck": None,
            "contextCard": None,
            "effect": None,
        },
        "logs": [],
        "current": {
            "turn": 3,
            "yourIndex": your_index,
            "firstPlayer": 0,
            "result": result,
            "players": players,
            "stadium": [],
            "looking": None,
        },
        "search_begin_input": "abc",
    }


class TestSamplePool:
    def test_deals_without_replacement(self):
        out = _sample_pool([1, 2, 3, 4, 5], {"a": 3, "b": 2}, _rng())
        assert len(out["a"]) == 3 and len(out["b"]) == 2
        # no overlap between the two deals
        assert set(out["a"]).isdisjoint(set(out["b"]))
        assert set(out["a"]) | set(out["b"]) == {1, 2, 3, 4, 5}

    def test_zero_count_returns_empty(self):
        out = _sample_pool([1, 2, 3], {"a": 0}, _rng())
        assert out["a"] == []

    def test_pool_too_small_falls_back_with_replacement(self):
        out = _sample_pool([1], {"a": 3}, _rng())
        assert len(out["a"]) == 3
        assert all(x == 1 for x in out["a"])


class TestLeafValue:
    def test_terminal_win_for_root(self):
        obs = _obs(result=0)  # root player is 0
        assert _leaf_value(obs, root_player=0) == 1.0

    def test_terminal_loss_for_root(self):
        obs = _obs(result=1)  # player 1 wins -> root (0) loses
        assert _leaf_value(obs, root_player=0) == -1.0

    def test_terminal_draw(self):
        obs = _obs(result=2)
        assert _leaf_value(obs, root_player=0) == 0.0

    def test_prize_lead_dominates(self):
        me = _side(prize=2)  # 4 prizes taken
        opp = _side(prize=6, player_index=1)  # 0 taken
        obs = _obs(me=me, opp=opp)
        # prize_lead = 4 - 0 = 4 -> 4*0.15 = 0.6, clipped
        v = _leaf_value(obs, root_player=0)
        assert v == pytest.approx(0.6)

    def test_hp_lead_contributes(self):
        me = _side(active=_pokemon_dict(hp=340, max_hp=340))
        opp = _side(active=_pokemon_dict(hp=100, max_hp=300), player_index=1)
        obs = _obs(me=me, opp=opp)
        # prize_lead=0, hp_lead = 1.0 - 0.333 = 0.667 -> 0.667*0.5 = 0.333
        v = _leaf_value(obs, root_player=0)
        assert v == pytest.approx((340 / 340 - 100 / 300) * 0.5, rel=1e-6)

    def test_clipped_to_range(self):
        me = _side(prize=1, active=_pokemon_dict(hp=340, max_hp=340), bench=[_pokemon_dict()] * 5)
        opp = _side(prize=6, player_index=1)
        obs = _obs(me=me, opp=opp)
        assert _leaf_value(obs, root_player=0) == 1.0  # large positive clips to 1

    def test_facedown_active_counts_as_zero_hp(self):
        me = _side(active=None)
        opp = _side(active=_pokemon_dict(hp=300, max_hp=300), player_index=1)
        obs = _obs(me=me, opp=opp)
        assert _leaf_value(obs, root_player=0) == pytest.approx(-0.5)

    def test_prizes_taken_helper(self):
        assert _prizes_taken(_side(prize=6)) == 0
        assert _prizes_taken(_side(prize=3)) == 3
        assert _prizes_taken(_side(prize=0)) == 6


class TestUCB1:
    def test_unvisited_returns_inf(self):
        n = _Node(search_id=0, parent=None, action=[], option_index=0, obs_dict={})
        assert _ucb1(n, parent_visits=10, c=1.4) == float("inf")

    def test_exploitation_term(self):
        n = _Node(
            search_id=0,
            parent=None,
            action=[],
            option_index=0,
            obs_dict={},
            visits=10,
            total_value=5.0,
        )
        assert _ucb1(n, 10, 0.0) == 0.5  # pure exploitation, c=0

    def test_exploration_increases_with_parent_visits(self):
        n1 = _Node(
            search_id=0,
            parent=None,
            action=[],
            option_index=0,
            obs_dict={},
            visits=4,
            total_value=2.0,
        )
        v_low = _ucb1(n1, parent_visits=5, c=1.4)
        v_high = _ucb1(n1, parent_visits=50, c=1.4)
        assert v_high > v_low


class TestResolveAction:
    def test_single_option_main(self):
        sel = {"minCount": 1, "maxCount": 1, "option": [{"type": 14}]}
        node = _Node(search_id=0, parent=None, action=[], option_index=-1, obs_dict={"select": sel})
        assert _resolve_action(node, 2, sel) == [2]

    def test_multi_count_takes_first_n(self):
        sel = {"minCount": 3, "maxCount": 3, "option": [{"type": 3}] * 6}
        node = _Node(search_id=0, parent=None, action=[], option_index=-1, obs_dict={"select": sel})
        # option_index is the chosen one, but maxCount=3 -> action is first 3 indices
        assert _resolve_action(node, 0, sel) == [0, 1, 2]


class TestBuildPredictions:
    def _db(self):
        cards = {
            677: CardInfo(
                id=677,
                name="Riolu",
                card_type=CT_POKEMON,
                card_type_name="pokemon",
                hp=80,
                energy_type=6,
                weakness=5,
                resistance=None,
                retreat_cost=2,
                is_basic=True,
                is_stage1=False,
                is_stage2=False,
                is_ex=False,
                is_mega_ex=False,
                is_tera=False,
                is_ace_spec=False,
                evolves_from=None,
            ),
            678: CardInfo(
                id=678,
                name="Mega Lucario ex",
                card_type=CT_POKEMON,
                card_type_name="pokemon",
                hp=340,
                energy_type=6,
                weakness=5,
                resistance=None,
                retreat_cost=2,
                is_basic=False,
                is_stage1=True,
                is_stage2=False,
                is_ex=True,
                is_mega_ex=True,
                is_tera=False,
                is_ace_spec=False,
                evolves_from="Riolu",
            ),
            6: CardInfo(
                id=6,
                name="Basic F Energy",
                card_type=5,
                card_type_name="basic_energy",
                hp=0,
                energy_type=6,
                weakness=None,
                resistance=None,
                retreat_cost=0,
                is_basic=False,
                is_stage1=False,
                is_stage2=False,
                is_ex=False,
                is_mega_ex=False,
                is_tera=False,
                is_ace_spec=False,
                evolves_from=None,
            ),
        }
        return CardDB(cards=cards, attacks={}, source="mock")

    def test_prediction_counts_match_state(self):
        me = _side(active=_pokemon_dict(678), prize=4, player_index=0)
        me["handCount"] = 5
        me["hand"] = [{"id": 677, "serial": 1, "playerIndex": 0}] * 5
        me["deckCount"] = 40
        opp = _side(active=_pokemon_dict(677), prize=5, player_index=1)
        opp["handCount"] = 6
        opp["deckCount"] = 38
        obs = _obs(me=me, opp=opp)
        my_deck = [677, 678, 6] * 20  # 60 cards
        preds = build_predictions(obs, my_deck=my_deck, card_db=self._db(), rng=_rng())
        assert len(preds.your_deck) == 40
        assert len(preds.your_prize) == 4
        assert len(preds.opponent_deck) == 38
        assert len(preds.opponent_prize) == 5
        assert len(preds.opponent_hand) == 6
        assert preds.opponent_active == []  # opp active is face-up

    def test_facedown_opponent_active_gets_basic_id(self):
        opp = _side(
            active=None, player_index=1
        )  # active=[] means no active; test facedown separately
        # facedown active is represented as [None] in the cg obs
        opp["active"] = [None]
        obs = _obs(opp=opp)
        my_deck = [677, 678, 6] * 20
        preds = build_predictions(obs, my_deck=my_deck, card_db=self._db(), rng=_rng())
        assert len(preds.opponent_active) == 1
        assert preds.opponent_active[0] == 677  # first basic in deck

    def test_mirror_prior_when_no_hint(self):
        me = _side(prize=6, player_index=0)
        me["deckCount"] = 55
        opp = _side(prize=6, player_index=1)
        opp["deckCount"] = 55
        opp["handCount"] = 7
        obs = _obs(me=me, opp=opp)
        my_deck = [677, 678, 6] * 20
        preds = build_predictions(obs, my_deck=my_deck, card_db=self._db(), rng=_rng())
        # opponent deck sampled from mirror (our deck); all ids are from our deck
        assert all(x in my_deck for x in preds.opponent_deck)


class TestMCTSResult:
    def test_dataclass_fields(self):
        r = MCTSResult(action=[0], visits={0: 5}, win_rates={0: 0.6}, simulations=5, elapsed=0.1)
        assert r.action == [0]
        assert r.simulations == 5
        assert not r.fell_back
