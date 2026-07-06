"""Game state visualiser and replay recorder for Pokemon TCG battles.

Provides three capabilities:

1. **Record** — capture every observation + action during a match into a
   replay file (JSON), optionally including ``visualize_data()`` god-view data
   from the engine.

2. **Render** — draw the current board, hand, options and logs to the terminal
   with ANSI colour, readable without a browser.

3. **Replay** — step through a saved replay file interactively, or dump a
   single frame as formatted text for sharing.

Usage (CLI)
-----------
See ``scripts/visualize.py`` or run::

    uv run python -m pokemon.visualize --help

Usage (library)
---------------
    from pokemon.visualize import render, record_match

    # Render a single observation to a string
    text = render(obs_dict, card_db=my_db)
    print(text)

    # Record a full match to a replay file
    result = record_match(
        agent0, agent1, output="replay.json",
        include_visualize=True,
    )
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re as _re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from pokemon.card_db import (
    ENERGY_NAME,
    CardDB,
    get_card_db,
)
from pokemon.state import GameState, OptionInfo, parse_obs

logger = logging.getLogger(__name__)

# ── ANSI helpers ──────────────────────────────────────────────────────

_CLEAR = "\033[H\033[J" if os.name != "nt" else "\n" * 80
_BOLD = "\033[1m" if os.name != "nt" else ""
_DIM = "\033[2m" if os.name != "nt" else ""
_RED = "\033[91m" if os.name != "nt" else ""
_GREEN = "\033[92m" if os.name != "nt" else ""
_YELLOW = "\033[93m" if os.name != "nt" else ""
_BLUE = "\033[94m" if os.name != "nt" else ""
_MAGENTA = "\033[95m" if os.name != "nt" else ""
_CYAN = "\033[96m" if os.name != "nt" else ""
_RESET = "\033[0m" if os.name != "nt" else ""

_SYMBOLS = {
    "poisoned": "☠",
    "burned": "🔥",
    "asleep": "💤",
    "paralyzed": "⚡",
    "confused": "🌀",
}


# ── Replay data structures ────────────────────────────────────────────


@dataclass
class ReplayFrame:
    """A single decision point in a recorded game."""

    obs: dict
    actions: list[int]
    player: int
    turn: int
    turn_action: int
    visualize_json: str | None = None


@dataclass
class Replay:
    """Full recording of a single match."""

    frames: list[ReplayFrame]
    deck0: list[int]
    deck1: list[int]
    agents: tuple[str, str]
    result: dict
    metadata: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        """Write replay to a JSON file."""
        path = Path(path)
        data = {
            "format_version": 1,
            "agents": list(self.agents),
            "deck0": self.deck0,
            "deck1": self.deck1,
            "result": self.result,
            "metadata": self.metadata,
            "frames": [
                {
                    "obs": f.obs,
                    "actions": f.actions,
                    "player": f.player,
                    "turn": f.turn,
                    "turn_action": f.turn_action,
                    "visualize_json": f.visualize_json,
                }
                for f in self.frames
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Replay:
        """Load replay from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        agents = tuple(data.get("agents", ["?", "?"]))
        result = data.get("result", {})
        deck0 = data.get("deck0", [])
        deck1 = data.get("deck1", [])
        frames = [
            ReplayFrame(
                obs=f["obs"],
                actions=f.get("actions", []),
                player=f.get("player", 0),
                turn=f.get("turn", 0),
                turn_action=f.get("turn_action", 0),
                visualize_json=f.get("visualize_json"),
            )
            for f in data.get("frames", [])
        ]
        return cls(
            frames=frames,
            deck0=deck0,
            deck1=deck1,
            agents=agents,
            result=result,
            metadata=data.get("metadata", {}),
        )

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def winner(self) -> int:
        return self.result.get("winner", -1)

    @property
    def total_turns(self) -> int:
        if not self.frames:
            return 0
        return self.frames[-1].turn


# ── Recording ─────────────────────────────────────────────────────────


class ReplayRecorder:
    """Captures observations + actions during a match for later replay."""

    def __init__(self, include_visualize: bool = False):
        self.include_visualize = include_visualize
        self.frames: list[ReplayFrame] = []
        self._deck0: list[int] = []
        self._deck1: list[int] = []

    def record_deck(self, player: int, deck: list[int]) -> None:
        if player == 0:
            self._deck0 = list(deck)
        else:
            self._deck1 = list(deck)

    def record_step(self, obs: dict, actions: list[int]) -> None:
        current = obs.get("current") or {}
        player = int(current.get("yourIndex", 0))
        turn = int(current.get("turn", 0))
        turn_action = int(current.get("turnActionCount", 0))
        vis = None
        if self.include_visualize:
            try:
                import cg.game as _cg_game

                vis = _cg_game.visualize_data()
            except Exception:
                pass
        self.frames.append(
            ReplayFrame(
                obs=obs,
                actions=list(actions),
                player=player,
                turn=turn,
                turn_action=turn_action,
                visualize_json=vis,
            )
        )

    def build_replay(
        self,
        agents: tuple[str, str],
        result: dict,
        **metadata,
    ) -> Replay:
        return Replay(
            frames=list(self.frames),
            deck0=list(self._deck0),
            deck1=list(self._deck1),
            agents=agents,
            result=result,
            metadata=metadata,
        )


def record_match(
    agent0: Callable,
    agent1: Callable,
    output: str | Path | None = None,
    agent_names: tuple[str, str] = ("agent0", "agent1"),
    include_visualize: bool = False,
    max_turns: int = 100,
) -> tuple[Replay | None, dict]:
    """Run a match between two agents and return a ``Replay`` and result.

    Parameters
    ----------
    agent0, agent1:
        Callable agents obeying the ``agent(obs_dict) -> list[int]`` contract.
    output:
        Optional path to write the replay JSON.  If ``None`` the replay is not
        saved to disk.
    agent_names:
        Human-readable names stored in the replay metadata.
    include_visualize:
        If ``True`` call ``visualize_data()`` after each step (god-view data).
        Requires the ``cg`` engine.
    max_turns:
        Abort if the match exceeds this many turns.

    Returns
    -------
    ``(replay, result)`` where *result* has keys *winner*, *turns*, *score0*,
    *score1*, *error*.
    """
    from pokemon.harness import _flatten_obs

    try:
        import cg.game as _cg_game
    except ImportError:
        raise RuntimeError(
            "Game engine not found. Download the simulation SDK:\n  make sim-download"
        )

    recorder = ReplayRecorder(include_visualize=include_visualize)

    deck_obs = {"select": None}
    try:
        deck0 = agent0(deck_obs)
        deck1 = agent1(deck_obs)
    except Exception as e:
        logger.warning("Deck selection error: %s", e)
        return (
            None,
            {"winner": -1, "turns": 0, "score0": 0, "score1": 0, "error": str(e)},
        )

    recorder.record_deck(0, deck0)
    recorder.record_deck(1, deck1)

    try:
        obs, _ = _cg_game.battle_start(deck0, deck1)
    except Exception as e:
        logger.warning("Battle start error: %s", e)
        return (
            None,
            {"winner": -1, "turns": 0, "score0": 0, "score1": 0, "error": str(e)},
        )

    flat_obs = _flatten_obs(obs)
    turn = 0
    try:
        while not flat_obs.get("done", False) and turn < max_turns:
            current_player = flat_obs["current_player"]
            agent = agent0 if current_player == 0 else agent1
            actions = agent(flat_obs)
            recorder.record_step(obs, actions)
            obs = _cg_game.battle_select(actions)
            flat_obs = _flatten_obs(obs)
            turn += 1
    except Exception as e:
        logger.warning("Match error at turn %d: %s", turn, e)
        _cg_game.battle_finish()
        result = {
            "winner": -1,
            "turns": turn,
            "score0": 0,
            "score1": 0,
            "error": str(e),
        }
        replay = recorder.build_replay(
            agents=agent_names,
            result=result,
            max_turns=max_turns,
            date=datetime.now().isoformat(),
        )
        if output:
            replay.save(output)
        return (replay, result)

    _cg_game.battle_finish()

    raw_result = obs.get("current", {}).get("result", -1) if obs else -1
    if raw_result == 0:
        winner = 0
    elif raw_result == 1:
        winner = 1
    else:
        winner = -1

    result = {
        "winner": winner,
        "turns": turn,
        "score0": 1 if winner == 0 else 0,
        "score1": 1 if winner == 1 else 0,
        "error": None,
    }
    replay = recorder.build_replay(
        agents=agent_names,
        result=result,
        max_turns=max_turns,
        date=datetime.now().isoformat(),
    )
    if output:
        replay.save(output)
    return (replay, result)


# ── Terminal rendering ────────────────────────────────────────────────


class GameRenderer:
    """Renders a ``GameState`` to a terminal-friendly formatted string."""

    def __init__(self, card_db: CardDB | None = None):
        self._db = card_db

    def db(self) -> CardDB:
        if self._db is None:
            self._db = get_card_db()
        return self._db

    def render(self, state: GameState, actions: list[int] | None = None) -> str:
        """Return a full-screen terminal rendering of the game state."""
        lines: list[str] = []
        lines.append(self._header(state, actions))
        lines.append("")
        me_col = self._side_column(state.me, state.your_index, is_me=True, state=state)
        opp_col = self._side_column(state.opp, 1 - state.your_index, is_me=False, state=state)
        mid_col = self._middle_column(state)
        max_h = max(len(me_col), len(opp_col), len(mid_col))
        me_col += [""] * (max_h - len(me_col))
        opp_col += [""] * (max_h - len(opp_col))
        mid_col += [""] * (max_h - len(mid_col))
        for m, c, o in zip(me_col, mid_col, opp_col):
            lines.append(f"  {m:<50}{c:<30}{o:<50}")
        lines.append("")
        lines.append(self._options_block(state))
        lines.append("")
        lines.append(self._log_block(state))
        return "\n".join(lines)

    def render_frame(
        self,
        obs: dict,
        actions: list[int] | None = None,
        visualize_json: str | None = None,
    ) -> str:
        """Render a raw observation dict (same API as ``render`` but takes raw dict)."""
        if visualize_json:
            try:
                vis = json.loads(visualize_json)
                # visualize_data() returns a JSON array of snapshots (one
                # per decision).  Take the last element as the current frame.
                if isinstance(vis, list):
                    vis = vis[-1] if vis else None
            except Exception:
                vis = None
            if vis:
                return self._render_god_view(vis, actions)
        state = parse_obs(obs)
        if state is None:
            return _DIM + "  (deck selection phase — no board state yet)" + _RESET
        return self.render(state, actions)

    def _render_god_view(self, vis: dict, actions: list[int] | None = None) -> str:
        """Render a god-view snapshot from ``visualize_data()``."""
        lines: list[str] = []
        current = vis.get("current") or {}
        turn = current.get("turn", 0)
        your_idx = current.get("yourIndex", 0)
        result = current.get("result", -1)
        result_str = f"  Result: P{result}" if result != -1 else ""
        turn_str = f"Turn {turn}"
        lines.append(
            f"{_BOLD}{_CYAN}  ═══ {turn_str}  │  P{your_idx} selecting  {result_str} ═══{_RESET}"
        )
        lines.append("")
        select = vis.get("select") or {}
        players = current.get("players") or [None, None]
        labels = [
            (0, "P0 (You)" if your_idx == 0 else "P0"),
            (1, "P1 (You)" if your_idx == 1 else "P1"),
        ]
        for idx, label in labels:
            p = players[idx] or {}
            lines.append(f"{_BOLD}  {label}:{_RESET}")
            actives = p.get("active") or []
            if actives and actives[0] is not None:
                a = actives[0]
                name = self._card_name(a.get("id"))
                hp_str = f"HP: {a.get('hp')}/{a.get('maxHp')}"
                lines.append(f"    Active: {_GREEN}{name}{_RESET}  {hp_str}")
                energies = a.get("energies") or []
                if energies:
                    e_str = " ".join(ENERGY_NAME.get(e, f"?{e}")[0] for e in energies)
                    lines.append(f"    Energy: {e_str} ({len(energies)})")
            bench = p.get("bench") or []
            for i, b in enumerate(bench):
                if b is None:
                    continue
                name = self._card_name(b.get("id"))
                lines.append(f"    Bench[{i}]: {name}  HP: {b.get('hp')}/{b.get('maxHp')}")
            hand = p.get("hand") or []
            hand_str = " ".join(self._card_name(c.get("id")) for c in hand if c) if hand else "?"
            lines.append(f"    Hand: {hand_str}  ({p.get('handCount', '?')})")
            lines.append(f"    Deck: {p.get('deckCount', 0)}  Prize: {len(p.get('prize') or [])}")
            lines.append("")
        if select:
            opts = select.get("option") or []
            lines.append(f"{_YELLOW}  Options ({len(opts)}):{_RESET}")
            for i, o in enumerate(opts):
                arrow = f"{_GREEN}◀ {_RESET}" if actions and i in actions else "  "
                type_name = self._opt_type_name(o.get("type"))
                detail = self._option_detail(o)
                lines.append(f"    [{i}] {arrow}{type_name}{detail}")
        logs = vis.get("logs") or []
        if logs:
            lines.append(f"\n{_DIM}  Logs:{_RESET}")
            for log in logs[-6:]:
                lines.append(f"    {_DIM}{self._log_text(log)}{_RESET}")
        return "\n".join(lines)

    # ── helpers ────────────────────────────────────────────────────

    def _header(self, state: GameState, actions: list[int] | None) -> str:
        turn = state.turn
        sel_type = self._sel_type_name(state.select.type)
        sel_ctx = self._sel_ctx_name(state.select.context)
        result_str = ""
        if state.is_done:
            result_str = f"  │  {_BOLD}{_RED}Result: P{state.result}{_RESET}"
        me_flag = f"{_GREEN}◀ YOU{_RESET}" if state.your_index == 0 else ""
        opp_flag = f"{_MAGENTA}◀ YOU{_RESET}" if state.your_index == 1 else ""
        return (
            f"{_BOLD}{_CYAN}  ═══ Turn {turn}  │  {sel_type} / {sel_ctx}  │  "
            f"P0 {me_flag}  vs  P1 {opp_flag}{result_str}  ═══{_RESET}"
        )

    def _side_column(self, side, player_idx: int, is_me: bool, state: GameState) -> list[str]:
        db = self.db()
        label_color = _GREEN if is_me else _MAGENTA
        label = f"{label_color}{'─── YOU ───' if is_me else '─── OPPONENT ───'}{_RESET}"
        lines: list[str] = [f"{_BOLD}{label}{_RESET}"]

        # Active
        active = side.active
        if active is not None:
            info = db.get(active.id)
            name = info.name if info else f"Card#{active.id}"
            sc = self._special_condition_text(side)
            hp_color = (
                _GREEN if active.hp_ratio > 0.5 else _YELLOW if active.hp_ratio > 0.2 else _RED
            )
            lines.append(f"  {_BOLD}Active:{_RESET} {name} {sc}")
            lines.append(f"    {hp_color}HP: {active.hp}/{active.max_hp}{_RESET}")
            if active.energies:
                e_str = " ".join(ENERGY_NAME.get(e, f"?{e}")[0] for e in active.energies)
                lines.append(f"    Energy: {e_str} ({len(active.energies)})")
            if active.tools:
                tool_names = [db.get(t) for t in active.tools]
                tool_str = ", ".join(t.name if t else f"#{t}" for t in tool_names)
                lines.append(f"    Tool: {tool_str}")
        else:
            lines.append(f"  {_DIM}Active: ??? (face down){_RESET}")

        # Bench
        if side.bench:
            lines.append(f"  {_BOLD}Bench:{_RESET}")
            for i, p in enumerate(side.bench):
                info = db.get(p.id)
                name = info.name if info else f"Card#{p.id}"
                hp_color = _GREEN if p.hp_ratio > 0.5 else _YELLOW
                lines.append(f"    [{i}] {name}  {hp_color}{p.hp}/{p.max_hp}{_RESET}")
        else:
            lines.append(f"  {_DIM}Bench: empty{_RESET}")

        # Hand
        if is_me:
            hand_ids = side.hand
            if hand_ids:
                hand_names = []
                for cid in hand_ids[:10]:
                    info = db.get(cid)
                    hand_names.append(info.name[:18] if info else f"#{cid}")
                if len(hand_ids) > 10:
                    hand_names.append("…")
                lines.append(f"  {_BOLD}Hand ({len(hand_ids)}):{_RESET}")
                lines.append(f"    {'  '.join(hand_names)}")
            else:
                lines.append(f"  {_DIM}Hand: empty{_RESET}")
        else:
            lines.append(f"  {_BOLD}Hand:{_RESET} {side.hand_count} cards")

        # Deck
        lines.append(f"  {_BOLD}Deck:{_RESET} {side.deck_count}")

        # Prize
        lines.append(
            f"  {_BOLD}Prize:{_RESET} {side.prize_count} remaining ({side.prizes_taken} taken)"
        )

        # Discard
        if side.discard:
            discard_names = []
            for cid in side.discard[-5:]:
                info = db.get(cid)
                discard_names.append(info.name[:12] if info else f"#{cid}")
            d_str = ", ".join(discard_names)
            if len(side.discard) > 5:
                d_str += " …"
            lines.append(f"  {_BOLD}Discard ({len(side.discard)}):{_RESET}")
            lines.append(f"    {d_str}")
        else:
            lines.append(f"  {_DIM}Discard: empty{_RESET}")

        return lines

    def _middle_column(self, state: GameState) -> list[str]:
        lines: list[str] = []
        lines.append(f"{_DIM}─── STATUS ───{_RESET}")
        me = state.me
        opp = state.opp
        if me.poisoned:
            lines.append(f"  {_RED}☠ You poisoned{_RESET}")
        if me.burned:
            lines.append(f"  {_RED}🔥 You burned{_RESET}")
        if me.asleep:
            lines.append(f"  {_YELLOW}💤 You asleep{_RESET}")
        if me.paralyzed:
            lines.append(f"  {_YELLOW}⚡ You paralyzed{_RESET}")
        if me.confused:
            lines.append(f"  {_YELLOW}🌀 You confused{_RESET}")
        if opp.poisoned:
            lines.append(f"  {_MAGENTA}☠ Opp poisoned{_RESET}")
        if opp.burned:
            lines.append(f"  {_MAGENTA}🔥 Opp burned{_RESET}")
        stadium = state.stadium
        if stadium:
            db = self.db()
            s_names = [db.get(s).name if db.get(s) else f"#{s}" for s in stadium]
            lines.append(f"  Stadium: {s_names[0]}")
        if state.supporter_played:
            lines.append(f"  {_DIM}Supporter used{_RESET}")
        if state.energy_attached:
            lines.append(f"  {_DIM}Energy attached{_RESET}")
        if state.retreated:
            lines.append(f"  {_DIM}Retreated{_RESET}")
        lines.append("")
        lines.append(f"  {_BOLD}Prize lead:{_RESET} {state.prize_lead:+d}")
        if state.is_done:
            w = "P0" if state.result == 0 else ("P1" if state.result == 1 else "Draw")
            lines.append(f"  {_BOLD}{_RED}Result: {w}{_RESET}")
        return lines

    def _options_block(self, state: GameState) -> str:
        opts = state.select.options
        if not opts:
            return f"  {_DIM}(no options){_RESET}"
        lines: list[str] = [f"{_BOLD}{_YELLOW}  ─── Options ({len(opts)}) ───{_RESET}"]
        for o in opts:
            type_name = self._opt_type_name(o.type)
            detail = self._option_detail(o)
            sc = state.select
            show_range = sc.min_count != 1 or sc.max_count != 1
            range_str = f"  [{sc.minCount}..{sc.maxCount}]" if show_range else ""
            lines.append(
                f"    [{o.index}] {_BOLD}{type_name}{_RESET}{detail}{_DIM}{range_str}{_RESET}"
            )
        return "\n".join(lines)

    def _log_block(self, state: GameState) -> str:
        logs = state.logs
        if not logs:
            return ""
        lines: list[str] = [f"{_DIM}  ─── Recent Logs ───{_RESET}"]
        for log in logs[-8:]:
            lines.append(f"    {_DIM}{self._log_text(log)}{_RESET}")
        return "\n".join(lines)

    # ── card / option / log description helpers ────────────────────

    def _card_name(self, card_id: int | None) -> str:
        if card_id is None:
            return "???"
        db = self.db()
        info = db.get(card_id)
        return info.name if info else f"#{card_id}"

    def _special_condition_text(self, side) -> str:
        parts = []
        if side.poisoned:
            parts.append(f"{_RED}☠{_RESET}")
        if side.burned:
            parts.append(f"{_RED}🔥{_RESET}")
        if side.asleep:
            parts.append(f"{_YELLOW}💤{_RESET}")
        if side.paralyzed:
            parts.append(f"{_YELLOW}⚡{_RESET}")
        if side.confused:
            parts.append(f"{_YELLOW}🌀{_RESET}")
        return " ".join(parts) if parts else ""

    @staticmethod
    def _sel_type_name(t: int | str) -> str:
        if isinstance(t, str):
            return t
        names = {
            0: "Main",
            1: "Card",
            2: "AttachedCard",
            3: "CardOrAttached",
            4: "Energy",
            5: "Skill",
            6: "Attack",
            7: "Evolve",
            8: "Count",
            9: "YesNo",
            10: "SpecialCondition",
        }
        return names.get(t, f"Type#{t}")

    @staticmethod
    def _sel_ctx_name(ctx: int | str) -> str:
        if isinstance(ctx, str):
            return ctx
        names = {
            0: "Main",
            1: "SetupActive",
            2: "SetupBench",
            3: "Switch",
            4: "ToActive",
            5: "ToBench",
            8: "Discard",
            15: "Damage",
            17: "Heal",
            21: "AttachFrom",
            22: "AttachTo",
            35: "Attack",
            37: "Evolve",
            38: "DrawCount",
            41: "IsFirst",
            42: "Mulligan",
            43: "Activate",
            46: "CoinHead",
            47: "AffectSC",
            48: "RecoverSC",
        }
        return names.get(ctx, f"Ctx#{ctx}")

    @staticmethod
    def _opt_type_name(t: int | str) -> str:
        if isinstance(t, str):
            return t
        names = {
            0: "Num",
            1: "Yes",
            2: "No",
            3: "Card",
            4: "ToolCard",
            5: "EnergyCard",
            6: "Energy",
            7: "Play",
            8: "Attach",
            9: "Evolve",
            10: "Ability",
            11: "Discard",
            12: "Retreat",
            13: "Attack",
            14: "End",
            15: "Skill",
            16: "SpecialCondition",
        }
        return names.get(t, f"Opt#{t}")

    def _option_detail(self, o: OptionInfo | dict) -> str:
        parts: list[str] = []
        if isinstance(o, dict):
            cid = o.get("cardId") or o.get("card_id")
            aid = o.get("attackId") or o.get("attack_id")
            hidx = o.get("index") if o.get("type") == 7 else None
            cnt = o.get("count") or o.get("number")
            is_ret = o.get("type") == 12
        else:
            cid = o.card_id
            aid = o.attack_id
            hidx = o.hand_index
            cnt = o.count
            is_ret = o.is_retreat
        if cid is not None and cid > 0:
            parts.append(f" {self._card_name(cid)}")
        if aid is not None:
            db = self.db()
            atk = db.attack(aid)
            if atk:
                parts.append(f" {atk.name} ({atk.damage}dmg)")
            else:
                parts.append(f" attack#{aid}")
        if hidx is not None:
            parts.append(f" (hand[{hidx}])")
        if cnt is not None:
            parts.append(f" ×{cnt}")
        if is_ret:
            parts.append(" (retreat)")
        return "".join(parts)

    @staticmethod
    def _log_type_name(t: int | str) -> str:
        """Map an int or string log type to a human-readable name.

        ``visualize_data()`` stringifies enums (e.g. ``"Draw"``), while the
        standard observation dict uses ints (e.g. ``4``).
        """
        if isinstance(t, str):
            return t
        _lt = {
            0: "Shuffle",
            2: "TurnStart",
            3: "TurnEnd",
            4: "Draw",
            5: "DrawReverse",
            6: "MoveCard",
            7: "MoveCardReverse",
            8: "Switch",
            9: "Change",
            10: "Play",
            11: "Attach",
            12: "Evolve",
            13: "Devolve",
            14: "MoveAttached",
            15: "Attack",
            16: "HPChange",
            17: "Poisoned",
            18: "Burned",
            19: "Asleep",
            20: "Paralyzed",
            21: "Confused",
            22: "Coin",
            23: "Result",
        }
        return _lt.get(t, f"L{t}")

    def _log_text(self, log: dict) -> str:
        raw_type = log.get("type", -1)
        pid = log.get("playerIndex", -1)
        prefix = f"P{pid}" if pid != -1 else "??"
        tname = self._log_type_name(raw_type)
        card_id = log.get("cardId")
        target_id = log.get("cardIdTarget")
        attack_id = log.get("attackId")
        value = log.get("value")
        text = f"{prefix} {tname}"
        if card_id is not None:
            text += f" [{self._card_name(card_id)}]"
        if target_id is not None:
            text += f" → {self._card_name(target_id)}"
        if attack_id is not None:
            db = self.db()
            atk = db.attack(attack_id)
            if atk:
                text += f" ({atk.name})"
        if value is not None:
            text += f" ({value:+d})"
        head = log.get("head")
        if head is not None:
            text += f" {'heads' if head else 'tails'}"
        result = log.get("result")
        if result is not None:
            reasons = {1: "Prizes", 2: "DeckOut", 3: "NoActive", 4: "Effect"}
            reason = log.get("reason", -1)
            rtext = reasons.get(reason, f"R{reason}")
            text += f" P{result} wins ({rtext})"
        return text


# ── Convenience render function ───────────────────────────────────────


def render(
    obs: dict,
    card_db: CardDB | None = None,
    actions: list[int] | None = None,
    visualize_json: str | None = None,
) -> str:
    """Render a single observation to a formatted terminal string.

    Parameters
    ----------
    obs:
        Raw observation dict from the ``cg`` engine.
    card_db:
        Optional card database for name lookups.  Loads default if not given.
    actions:
        Optional list of chosen action indices to highlight.
    visualize_json:
        Optional ``visualize_data()`` JSON string for god-view rendering.

    Returns
    -------
    str:
        Formatted terminal display string (with ANSI colour codes).
    """
    renderer = GameRenderer(card_db=card_db)
    return renderer.render_frame(obs, actions=actions, visualize_json=visualize_json)


# ── Interactive replay player ─────────────────────────────────────────


class ReplayPlayer:
    """Interactive step-through of a recorded :class:`Replay`."""

    def __init__(self, replay: Replay, card_db: CardDB | None = None):
        self.replay = replay
        self._index = 0
        self._renderer = GameRenderer(card_db=card_db)

    @property
    def index(self) -> int:
        return self._index

    @property
    def n_frames(self) -> int:
        return self.replay.n_frames

    def current_frame(self) -> ReplayFrame | None:
        if 0 <= self._index < self.n_frames:
            return self.replay.frames[self._index]
        return None

    def render_current(self) -> str:
        frame = self.current_frame()
        if frame is None:
            return _DIM + "  (end of replay)" + _RESET
        return self._renderer.render_frame(
            frame.obs,
            actions=frame.actions,
            visualize_json=frame.visualize_json,
        )

    def next(self, steps: int = 1) -> bool:
        new_idx = min(self._index + steps, self.n_frames - 1)
        changed = new_idx != self._index
        self._index = new_idx
        return changed

    def prev(self, steps: int = 1) -> bool:
        new_idx = max(self._index - steps, 0)
        changed = new_idx != self._index
        self._index = new_idx
        return changed

    def jump(self, idx: int) -> bool:
        idx = max(0, min(idx, self.n_frames - 1))
        changed = idx != self._index
        self._index = idx
        return changed

    def summary(self) -> str:
        r = self.replay
        n = r.n_frames
        w = r.winner
        agent_names = r.agents
        turns = r.total_turns
        winner_str = (
            f"{agent_names[0]} (P0)" if w == 0 else f"{agent_names[1]} (P1)" if w == 1 else "Draw"
        )
        error = r.result.get("error")
        error_str = f"  Error: {error}" if error else ""
        return (
            f"{_BOLD}Replay Summary{_RESET}\n"
            f"  Agents: {agent_names[0]} vs {agent_names[1]}\n"
            f"  Frames: {n}  Turns: {turns}\n"
            f"  Winner: {winner_str}{error_str}\n"
            f"  Frame:  {self._index + 1}/{n}"
        )


def run_interactive(
    agent0: Callable,
    agent1: Callable,
    card_db: CardDB | None = None,
    output: str | Path | None = None,
    agent_names: tuple[str, str] = ("agent0", "agent1"),
    max_turns: int = 100,
    step_delay: float = 0.3,
) -> None:
    """Run a match with live per-step terminal display.

    Each decision is rendered to the terminal before the agent acts.  A small
    ``step_delay`` (seconds) gives time to read the board before the next
    action fires.
    """
    from pokemon.harness import _flatten_obs

    try:
        import cg.game as _cg_game
    except ImportError:
        raise RuntimeError(
            "Game engine not found. Download the simulation SDK:\n  make sim-download"
        )

    recorder = ReplayRecorder(include_visualize=True)

    deck_obs = {"select": None}
    try:
        deck0 = agent0(deck_obs)
        deck1 = agent1(deck_obs)
    except Exception as e:
        print(f"Deck selection error: {e}")
        return

    recorder.record_deck(0, deck0)
    recorder.record_deck(1, deck1)

    try:
        obs, _ = _cg_game.battle_start(deck0, deck1)
    except Exception as e:
        print(f"Battle start error: {e}")
        return

    flat_obs = _flatten_obs(obs)
    turn = 0
    try:
        while not flat_obs.get("done", False) and turn < max_turns:
            current_player = flat_obs["current_player"]
            agent = agent0 if current_player == 0 else agent1

            # Render
            vis_str = _cg_game.visualize_data()
            rendered = render(obs, card_db=card_db, visualize_json=vis_str)
            sys.stdout.write(_CLEAR)
            sys.stdout.write(rendered)
            t = obs.get("current", {}).get("turn", "?")
            turn_info = f"Turn {t}  |  P{current_player}'s turn  |  Action {turn}"
            print(f"\n{_BOLD}{_CYAN}  {turn_info}{_RESET}")
            sys.stdout.flush()

            time.sleep(step_delay)

            actions = agent(flat_obs)
            recorder.record_step(obs, actions)
            obs = _cg_game.battle_select(actions)
            flat_obs = _flatten_obs(obs)
            turn += 1
    except Exception as e:
        logger.warning("Match error at turn %d: %s", turn, e)
        _cg_game.battle_finish()
        result = {
            "winner": -1,
            "turns": turn,
            "score0": 0,
            "score1": 0,
            "error": str(e),
        }
        replay = recorder.build_replay(
            agents=agent_names,
            result=result,
            max_turns=max_turns,
            interactive=True,
        )
        if output:
            replay.save(output)
        _show_final_result(render, obs, card_db, result)
        return

    _cg_game.battle_finish()

    raw_result = obs.get("current", {}).get("result", -1) if obs else -1
    if raw_result == 0:
        winner = 0
    elif raw_result == 1:
        winner = 1
    else:
        winner = -1

    result = {
        "winner": winner,
        "turns": turn,
        "score0": 1 if winner == 0 else 0,
        "score1": 1 if winner == 1 else 0,
        "error": None,
    }
    replay = recorder.build_replay(
        agents=agent_names,
        result=result,
        max_turns=max_turns,
        interactive=True,
    )
    if output:
        replay.save(output)
    _show_final_result(render, obs, card_db, result)


def _show_final_result(
    render_fn,
    obs: dict,
    card_db: CardDB | None,
    result: dict,
) -> None:
    """Display the final board + result."""
    try:
        import cg.game as _cg_game

        vis_str = _cg_game.visualize_data()
        rendered = render_fn(obs, card_db=card_db, visualize_json=vis_str)
    except Exception:
        rendered = render_fn(obs, card_db=card_db)
    sys.stdout.write(_CLEAR)
    sys.stdout.write(rendered)
    w = result.get("winner", -1)
    t = result.get("turns", 0)
    color = _GREEN if w != -1 else _YELLOW
    winner_str = f"P{w}" if w != -1 else "Draw"
    print(f"\n{_BOLD}{color}  Game Over — {winner_str} wins in {t} turns{_RESET}")
    if result.get("error"):
        print(f"  {_RED}Error: {result['error']}{_RESET}")
    sys.stdout.flush()


# ── CLI entry point (when run as module) ──────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pokemon TCG game visualizer and replay tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    run_p = sub.add_parser("run", help="Run a match with live terminal display")
    run_p.add_argument("--agent0", required=True, help="Path to agent0 Python file")
    run_p.add_argument("--deck0", required=True, help="Path to agent0 deck CSV")
    run_p.add_argument("--agent1", required=True, help="Path to agent1 Python file")
    run_p.add_argument("--deck1", required=True, help="Path to agent1 deck CSV")
    run_p.add_argument("--delay", type=float, default=0.3, help="Step delay in seconds")
    run_p.add_argument("--max-turns", type=int, default=100)
    run_p.add_argument("--output", help="Save replay to this JSON file")
    run_p.add_argument(
        "--names", nargs=2, default=["agent0", "agent1"], help="Display names for the agents"
    )

    # record
    rec_p = sub.add_parser("record", help="Record a match to a replay file (no display)")
    rec_p.add_argument("--agent0", required=True, help="Path to agent0 Python file")
    rec_p.add_argument("--deck0", required=True, help="Path to agent0 deck CSV")
    rec_p.add_argument("--agent1", required=True, help="Path to agent1 Python file")
    rec_p.add_argument("--deck1", required=True, help="Path to agent1 deck CSV")
    rec_p.add_argument("--output", default="replay.json", help="Output replay JSON file")
    rec_p.add_argument("--max-turns", type=int, default=100)
    rec_p.add_argument(
        "--names", nargs=2, default=["agent0", "agent1"], help="Display names for the agents"
    )

    # replay
    replay_p = sub.add_parser("replay", help="Step through a saved replay file")
    replay_p.add_argument("file", help="Replay JSON file")
    replay_p.add_argument("--frame", type=int, default=0, help="Jump to frame N")
    replay_p.add_argument("--dump", action="store_true", help="Print all frames as text and exit")
    replay_p.add_argument("--summary", action="store_true", help="Print replay summary and exit")

    # render
    render_p = sub.add_parser("render", help="Render a single observation JSON to text")
    render_p.add_argument("file", help="JSON file with an observation dict")
    render_p.add_argument("--actions", type=int, nargs="*", help="Action indices to highlight")

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``uv run python -m pokemon.visualize ...``"""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "record":
        _cmd_record(args)
    elif args.command == "replay":
        _cmd_replay(args)
    elif args.command == "render":
        _cmd_render(args)


def _load_agent_from_file(path: str, deck: list[int], seed: int = 42):
    """Dynamically import an agent from a Python file path."""
    import importlib.util
    import inspect

    from pokemon.agent import Agent, RandomAgent, RuleBasedAgent

    mod_name = f"_agent_{Path(path).stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for _, cls in inspect.getmembers(mod, inspect.isclass):
        if issubclass(cls, Agent) and cls not in (Agent, RandomAgent, RuleBasedAgent):
            return cls(deck=deck, random_seed=seed)

    raise ValueError(f"No Agent subclass found in {path}")


def _load_deck(path: str) -> list[int]:
    from pokemon.deck import Deck

    return Deck.from_csv(path).cards


def _cmd_run(args) -> None:
    deck0 = _load_deck(args.deck0)
    deck1 = _load_deck(args.deck1)
    agent0 = _load_agent_from_file(args.agent0, deck0)
    agent1 = _load_agent_from_file(args.agent1, deck1)
    run_interactive(
        agent0,
        agent1,
        output=args.output,
        agent_names=tuple(args.names),
        max_turns=args.max_turns,
        step_delay=args.delay,
    )


def _cmd_record(args) -> None:
    deck0 = _load_deck(args.deck0)
    deck1 = _load_deck(args.deck1)
    agent0 = _load_agent_from_file(args.agent0, deck0)
    agent1 = _load_agent_from_file(args.agent1, deck1)
    print(f"Recording match: {args.names[0]} vs {args.names[1]}...")
    replay, result = record_match(
        agent0,
        agent1,
        output=args.output,
        agent_names=tuple(args.names),
        include_visualize=True,
        max_turns=args.max_turns,
    )
    if replay:
        print(f"  Saved {replay.n_frames} frames to {args.output}")
    w = result.get("winner", -1)
    t = result.get("turns", 0)
    print(f"  Result: P{w} wins in {t} turns" if w != -1 else f"  Result: Draw in {t} turns")


def _cmd_replay(args) -> None:
    replay = Replay.load(args.file)
    player = ReplayPlayer(replay)
    if args.summary:
        print(player.summary())
        return
    if args.dump:
        for i in range(replay.n_frames):
            player.jump(i)
            rendered = player.render_current()
            clean = _re.sub(r"\033\[[0-9;]*m", "", rendered)
            print(f"\n{'=' * 72}")
            print(f"Frame {i + 1}/{replay.n_frames}")
            print(clean)
        return
    if args.frame:
        player.jump(args.frame)
        print(player.render_current())
        return
    _interactive_replay_loop(player)


def _cmd_render(args) -> None:
    obs = json.loads(Path(args.file).read_text())
    rendered = render(obs, actions=args.actions)
    clean = _re.sub(r"\033\[[0-9;]*m", "", rendered)
    print(clean)


def _interactive_replay_loop(player: ReplayPlayer) -> None:
    """Terminal UI for stepping through a replay."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)
        while True:
            sys.stdout.write(_CLEAR)
            sys.stdout.write(player.render_current())
            sys.stdout.write(f"\n  {_DIM}[n]ext  [p]rev  [<]jump  [s]ummary  [q]uit{_RESET} ")
            sys.stdout.flush()

            ch = sys.stdin.read(1)
            if ch == "q":
                break
            elif ch == "n":
                player.next()
            elif ch == "N":
                player.next(10)
            elif ch == "p":
                player.prev()
            elif ch == "P":
                player.prev(10)
            elif ch == "s":
                sys.stdout.write(_CLEAR)
                print(player.summary())
                print(f"\n  {_DIM}Press any key to continue...{_RESET}")
                sys.stdout.flush()
                sys.stdin.read(1)
            elif ch == "<":
                sys.stdout.write("\n  Jump to frame (0-based): ")
                sys.stdout.flush()
                line = ""
                while True:
                    c = sys.stdin.read(1)
                    if c == "\r" or c == "\n":
                        break
                    if c.isdigit():
                        line += c
                        sys.stdout.write(c)
                        sys.stdout.flush()
                if line:
                    player.jump(int(line))
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()


if __name__ == "__main__":
    main()
