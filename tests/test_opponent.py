"""Tests for pokemon.opponent (archetype classifier + counter strategy)."""

from __future__ import annotations

from pokemon.opponent import (
    Classification,
    OpponentClassifier,
    archetype_deck_list,
    counter_strategy,
)
from pokemon.state import parse_obs


def _obs_with_logs(your_index: int, logs: list[dict]) -> dict:
    """Build a minimal obs dict with the given logs."""
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
        "logs": logs,
        "current": {
            "turn": 3,
            "yourIndex": your_index,
            "firstPlayer": 0,
            "result": -1,
            "stadium": [],
            "looking": None,
            "players": [
                {
                    "active": [],
                    "bench": [],
                    "benchMax": 5,
                    "deckCount": 40,
                    "discard": [],
                    "prize": [None] * 6,
                    "handCount": 0,
                    "hand": None,
                    "poisoned": False,
                    "burned": False,
                    "asleep": False,
                    "paralyzed": False,
                    "confused": False,
                },
                {
                    "active": [],
                    "bench": [],
                    "benchMax": 5,
                    "deckCount": 40,
                    "discard": [],
                    "prize": [None] * 6,
                    "handCount": 0,
                    "hand": None,
                    "poisoned": False,
                    "burned": False,
                    "asleep": False,
                    "paralyzed": False,
                    "confused": False,
                },
            ],
        },
        "search_begin_input": "abc",
    }


def _play_log(player: int, card_id: int) -> dict:
    return {"type": 10, "playerIndex": player, "cardId": card_id, "serial": 1}


def _evolve_log(player: int, card_id: int, target_id: int = 0) -> dict:
    return {
        "type": 12,
        "playerIndex": player,
        "cardId": card_id,
        "serial": 1,
        "cardIdTarget": target_id,
        "serialTarget": 2,
    }


class TestOpponentClassifier:
    def test_empty_classifier_returns_none(self):
        clf = OpponentClassifier(your_index=0)
        result = clf.classify()
        assert result.archetype is None
        assert result.confidence == 0.0

    def test_ignores_own_card_plays(self):
        clf = OpponentClassifier(your_index=0)
        state = parse_obs(_obs_with_logs(0, [_play_log(0, 678)]))
        clf.update(state)
        assert clf.seen_cards == set()

    def test_ignores_basic_energy(self):
        clf = OpponentClassifier(your_index=0)
        state = parse_obs(_obs_with_logs(0, [_play_log(1, 3)]))
        clf.update(state)
        assert clf.seen_cards == set()

    def test_identifies_lucario_from_key_pokemon(self):
        clf = OpponentClassifier(your_index=0)
        logs = [_play_log(1, 677), _evolve_log(1, 678, 677)]
        state = parse_obs(_obs_with_logs(0, logs))
        clf.update(state)
        result = clf.classify()
        assert result.archetype == "mega_lucario_ex"
        assert 678 in result.matched_cards
        assert result.confidence > 0.5

    def test_identifies_iono_from_basics(self):
        clf = OpponentClassifier(your_index=0)
        logs = [_play_log(1, 268), _play_log(1, 270)]
        state = parse_obs(_obs_with_logs(0, logs))
        clf.update(state)
        result = clf.classify()
        assert result.archetype == "ionos_deck"

    def test_identifies_dragapult(self):
        clf = OpponentClassifier(your_index=0)
        logs = [_play_log(1, 119), _evolve_log(1, 120, 119), _evolve_log(1, 121, 120)]
        state = parse_obs(_obs_with_logs(0, logs))
        clf.update(state)
        result = clf.classify()
        assert result.archetype == "dragapult_ex"
        assert result.identified

    def test_identifies_abomasnow(self):
        clf = OpponentClassifier(your_index=0)
        logs = [_play_log(1, 722), _evolve_log(1, 723, 722)]
        state = parse_obs(_obs_with_logs(0, logs))
        clf.update(state)
        result = clf.classify()
        assert result.archetype == "mega_abomasnow_ex"

    def test_accumulates_across_calls(self):
        clf = OpponentClassifier(your_index=0)
        # First batch: only a basic
        state1 = parse_obs(_obs_with_logs(0, [_play_log(1, 677)]))
        clf.update(state1)
        assert clf.classify().confidence < 0.8
        # Second batch: the key Pokemon
        state2 = parse_obs(_obs_with_logs(0, [_evolve_log(1, 678, 677)]))
        clf.update(state2)
        assert clf.classify().identified

    def test_classification_properties(self):
        c = Classification(
            archetype="ionos_deck", confidence=0.9, seen_cards={268}, matched_cards={268}
        )
        assert c.identified
        c2 = Classification(archetype=None, confidence=0.0, seen_cards=set(), matched_cards=set())
        assert not c2.identified


class TestCounterStrategy:
    def test_iono_strategy_has_type_advantage(self):
        s = counter_strategy("ionos_deck")
        assert s["type_advantage"] is True
        assert s["aggression"] > 1.0

    def test_dragapult_strategy_protects_bench(self):
        s = counter_strategy("dragapult_ex")
        assert s["protect_bench"] is True

    def test_abomasnow_strategy_protects_bench(self):
        s = counter_strategy("mega_abomasnow_ex")
        assert s["protect_bench"] is True

    def test_unknown_strategy_is_balanced(self):
        s = counter_strategy(None)
        assert s["aggression"] == 1.0
        assert s["type_advantage"] is False

    def test_lucario_mirror_is_tempo_race(self):
        s = counter_strategy("mega_lucario_ex")
        assert s["aggression"] > 1.0


class TestArchetypeDeckList:
    def test_known_archetype_returns_deck(self):
        deck = archetype_deck_list("mega_lucario_ex")
        assert deck is not None
        assert len(deck) == 60

    def test_unknown_archetype_returns_none(self):
        assert archetype_deck_list(None) is None
        assert archetype_deck_list("nonexistent") is None
