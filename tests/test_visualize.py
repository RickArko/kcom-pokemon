"""Tests for the visualization module.

Uses mock observations (no ``cg`` engine required).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pokemon.state import OPT_ATTACK, OPT_END, OPT_PLAY
from pokemon.visualize import Replay, ReplayFrame, ReplayPlayer, render

# ── test helpers (mirror tests/test_state.py) ─────────────────────────


CTX_MAIN = 0
SEL_MAIN = 0


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


# ── tests ─────────────────────────────────────────────────────────────


class TestRender:
    """Test that render produces output for known states."""

    def test_render_returns_string(self):
        me = _player(active=_pokemon(678, hp=340, energies=[6]), hand=[677, 6])
        opp = _player(active=_pokemon(722, hp=90, energies=[3], player_index=1))
        obs = _state(me=me, opp=opp, turn=3)
        text = render(obs)
        assert isinstance(text, str)
        assert len(text) > 100
        assert "340" in text
        assert "Turn 3" in text

    def test_render_deck_selection(self):
        text = render({"select": None})
        assert "deck selection" in text

    def test_render_with_actions(self):
        me = _player(active=_pokemon(678), hand=[677])
        opp = _player(active=_pokemon(722, player_index=1))
        opts = [{"type": OPT_PLAY, "index": 0}, {"type": OPT_END}]
        obs = _state(me=me, opp=opp, select=_main_select(opts))
        text = render(obs, actions=[0])
        assert isinstance(text, str)

    def test_render_game_over(self):
        me = _player(active=None, prize=0)
        opp = _player(active=None, prize=6)
        obs = _state(me=me, opp=opp, result=0)
        text = render(obs)
        assert "Result" in text

    def test_render_bench(self):
        bench_poke = [_pokemon(677, hp=60, max_hp=60)]
        me = _player(active=_pokemon(678), bench=bench_poke, hand=[])
        opp = _player(active=_pokemon(722, player_index=1))
        obs = _state(me=me, opp=opp)
        text = render(obs)
        assert "Bench" in text


class TestReplayRoundtrip:
    """Test saving and loading replay files."""

    def test_save_and_load(self, tmp_path):
        frames = [
            ReplayFrame(obs={"select": None}, actions=[], player=0, turn=0, turn_action=0),
            ReplayFrame(
                obs={"select": {"option": [{"type": 14}]}, "current": {"turn": 1}},
                actions=[0],
                player=0,
                turn=1,
                turn_action=0,
            ),
        ]
        replay = Replay(
            frames=frames,
            deck0=[1] * 60,
            deck1=[2] * 60,
            agents=("agent0", "agent1"),
            result={"winner": 0, "turns": 1, "score0": 1, "score1": 0, "error": None},
            metadata={"date": "2025-01-01"},
        )
        path = tmp_path / "test_replay.json"
        saved = replay.save(str(path))
        assert saved.exists()

        loaded = Replay.load(str(path))
        assert loaded.n_frames == 2
        assert loaded.winner == 0
        assert loaded.total_turns == 1
        assert loaded.agents == ("agent0", "agent1")
        assert loaded.deck0 == [1] * 60
        assert loaded.deck1 == [2] * 60

    def test_empty_replay(self):
        frames = [
            ReplayFrame(obs={"select": None}, actions=[], player=0, turn=0, turn_action=0),
        ]
        replay = Replay(
            frames=frames,
            deck0=[],
            deck1=[],
            agents=("a", "b"),
            result={"winner": -1, "turns": 0},
        )
        assert replay.n_frames == 1
        assert replay.winner == -1

    def test_replay_with_visualize(self):
        frames = [
            ReplayFrame(
                obs={"select": None},
                actions=[],
                player=0,
                turn=0,
                turn_action=0,
                visualize_json=json.dumps({"test": "data"}),
            ),
        ]
        replay = Replay(
            frames=frames,
            deck0=[],
            deck1=[],
            agents=("a", "b"),
            result={"winner": 0},
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            replay.save(f.name)
            path = f.name
        loaded = Replay.load(path)
        assert loaded.frames[0].visualize_json == json.dumps({"test": "data"})
        Path(path).unlink(missing_ok=True)


class TestReplayPlayer:
    """Test interactive replay navigation."""

    def test_navigation(self):
        frames = [
            ReplayFrame(obs={"turn": i}, actions=[], player=i % 2, turn=i, turn_action=0)
            for i in range(10)
        ]
        replay = Replay(
            frames=frames,
            deck0=[],
            deck1=[],
            agents=("a", "b"),
            result={"winner": 0},
        )
        player = ReplayPlayer(replay)

        assert player.index == 0
        assert player.n_frames == 10

        player.next()
        assert player.index == 1

        player.next(5)
        assert player.index == 6

        player.prev()
        assert player.index == 5

        player.jump(9)
        assert player.index == 9

        player.next()  # clamped to last
        assert player.index == 9

        player.jump(-5)  # clamped to 0
        assert player.index == 0

    def test_render_current(self):
        frame = ReplayFrame(
            obs={"select": None},
            actions=[],
            player=0,
            turn=0,
            turn_action=0,
        )
        replay = Replay(
            frames=[frame],
            deck0=[],
            deck1=[],
            agents=("a", "b"),
            result={"winner": -1},
        )
        player = ReplayPlayer(replay)
        text = player.render_current()
        assert isinstance(text, str)
        assert len(text) > 0

        # Advance past last frame should stay at last frame
        player.next()
        text = player.render_current()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_summary(self):
        frames = [
            ReplayFrame(obs={}, actions=[], player=0, turn=i, turn_action=0) for i in range(5)
        ]
        replay = Replay(
            frames=frames,
            deck0=[1] * 60,
            deck1=[2] * 60,
            agents=("agent_a", "agent_b"),
            result={"winner": 0, "turns": 5, "score0": 1, "score1": 0},
        )
        player = ReplayPlayer(replay)
        summary = player.summary()
        assert "agent_a" in summary
        assert "agent_b" in summary
        assert "Winner" in summary
        assert "5" in summary


class TestRenderWithEngine:
    """Integration-style tests using full observation shapes."""

    def test_full_game_state_render(self):
        me = _player(
            active=_pokemon(678, hp=200, max_hp=340, energies=[6, 6, 6, 6]),
            bench=[_pokemon(677, hp=60, max_hp=60)],
            hand=[677, 677, 1145, 1205, 6, 6],
            deck_count=34,
            prize=5,
        )
        opp = _player(
            active=_pokemon(722, hp=90, max_hp=340, energies=[3, 3, 3], player_index=1),
            bench=[_pokemon(721, hp=90, max_hp=90, player_index=1)],
            hand_count=4,
            deck_count=40,
            prize=6,
        )
        opts = [
            {"type": OPT_PLAY, "index": 0},
            {"type": OPT_PLAY, "index": 1},
            {"type": OPT_PLAY, "index": 2},
            {"type": OPT_ATTACK, "attackId": 983},
            {"type": OPT_END},
        ]
        obs = _state(me=me, opp=opp, select=_main_select(opts), turn=5, result=-1)
        text = render(obs)

        assert isinstance(text, str)
        assert "Turn 5" in text
        assert "200/340" in text
        assert "90/340" in text
        assert "Options" in text
        assert "Attack" in text or "attack" in text


class TestRenderEdgeCases:
    """Edge cases: empty board, face-down active, etc."""

    def test_both_active_none(self):
        me = _player(active=None, bench=[], hand=[])
        opp = _player(active=None, bench=[], hand_count=0)
        obs = _state(me=me, opp=opp)
        text = render(obs)
        assert isinstance(text, str)

    def test_special_conditions(self):
        me = _player(active=_pokemon(678))
        opp = _player(active=_pokemon(722, player_index=1))
        obs = _state(me=me, opp=opp)
        obs["current"]["players"][0]["poisoned"] = True
        obs["current"]["players"][0]["burned"] = True
        text = render(obs)
        assert isinstance(text, str)

    def test_stadium_present(self):
        me = _player(active=_pokemon(678))
        opp = _player(active=_pokemon(722, player_index=1))
        obs = _state(me=me, opp=opp)
        obs["current"]["stadium"] = [{"id": 1145, "serial": 1, "playerIndex": 0}]
        text = render(obs)
        assert isinstance(text, str)
