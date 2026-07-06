# Visualization Tool: Replay Recording, Terminal Display & Playback

A terminal-based game state visualizer for Pokemon TCG AI battles. Record
matches, replay them step by step, or watch live as agents play — all from the
command line.

## Quick Start

```bash
# Record a match between two experiments
uv run python scripts/visualize.py record \
    --agent0 workspace/exp002_lucario_heuristic/agent.py \
    --deck0 workspace/exp002_lucario_heuristic/deck.csv \
    --agent1 workspace/exp003_lucario_mcts/agent.py \
    --deck1 workspace/exp003_lucario_mcts/deck.csv \
    --output data/replays/lucario_mirror.json

# Step through the replay interactively
uv run python scripts/visualize.py replay data/replays/lucario_mirror.json

# Run a match with live per-step display
uv run python scripts/visualize.py run \
    --agent0 workspace/exp002_lucario_heuristic/agent.py \
    --deck0 workspace/exp002_lucario_heuristic/deck.csv \
    --agent1 workspace/exp003_lucario_mcts/agent.py \
    --deck1 workspace/exp003_lucario_mcts/deck.csv \
    --delay 0.5
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  scripts/visualize.py          CLI entry point (argparse)        │
│       │                                                          │
│       v                                                          │
│  src/pokemon/visualize.py      Core module                       │
│       ├── ReplayRecorder       Captures observations + actions   │
│       ├── Replay               Save/load JSON replay files       │
│       ├── GameRenderer         Formats GameState → ANSI text     │
│       ├── ReplayPlayer         Interactive frame navigation      │
│       ├── render()             Single-frame convenience function │
│       ├── record_match()       One-shot record + save            │
│       └── run_interactive()    Live match with per-step display  │
│       │                                                          │
│       ├── pokemon.state        GameState, parse_obs()            │
│       └── pokemon.card_db      CardDB, name/attack lookups       │
└──────────────────────────────────────────────────────────────────┘
```

The tool works at two levels:

1. **Observation-level** — records the raw observation dicts the engine sends
   to agents at each decision point, plus the actions the agents chose.
2. **God-view** — optionally calls the engine's `visualize_data()` after every
   step to capture a full-revelation snapshot (both players' hands, deck
   contents, prize cards) for richer replay viewing.

## Happy Path: Record → Analyze → Iterate

### 1. Record a match

```bash
mkdir -p data/replays

# Record a mirror match between exp002 (heuristic) and exp005 (MCTS + opponent model)
uv run python scripts/visualize.py record \
    --agent0 workspace/exp002_lucario_heuristic/agent.py \
    --deck0 workspace/exp002_lucario_heuristic/deck.csv \
    --agent1 workspace/exp005_lucario_opp_mcts/agent.py \
    --deck1 workspace/exp005_lucario_opp_mcts/deck.csv \
    --names heuristic opp_mcts \
    --output data/replays/heuristic_vs_opp_mcts.json
```

Output:
```
Recording match: heuristic vs opp_mcts...
  Saved 142 frames to data/replays/heuristic_vs_opp_mcts.json
  Result: P0 wins in 12 turns
```

The replay JSON is a self-contained file::
- `deck0`, `deck1` — both decks
- `agents` — display names
- `result` — winner, turns, scores
- `frames` — array of observations + actions for every decision point

### 2. Explore the replay interactively

```bash
uv run python scripts/visualize.py replay data/replays/heuristic_vs_opp_mcts.json
```

This opens an interactive terminal viewer:

```
═══ Turn 3  │  Main / Main  │  P0 ◀ YOU  vs  P1  ═══

─── YOU ───                ─── STATUS ───        ─── OPPONENT ───
Active: Mega Lucario ex        Prize lead: +0    Active: Mega Abomasnow ex
  HP: 340/340                                      HP: 90/340
  Energy: F F F F (4)           Energy attached   Energy: W W (2)
  Tool: Maximum Belt
                                              Hand: 4 cards
Bench:                                            Deck: 40
  [0] Riolu  60/60             Supporter used    Prize: 6 remaining (0 taken)
                                                    Discard: empty
Hand (6):
  677  677  1145  1205  6  6
Deck: 34
Prize: 5 remaining (1 taken)
Discard: empty

─── Options (4) ───
  [0] Attack Mega Brave (270dmg)
  [1] Attack Mega Signal (130dmg)
  [2] Play Riolu (hand[0])
  [3] End

─── Recent Logs ───
  P0 TurnStart
  P0 Play [Mega Signal]
  P1 DrawReverse
  P1 Change [Mega Abomasnow ex] → [Snover]

[n]ext  [p]rev  [<]jump  [s]ummary  [q]uit
```

| Key | Action |
|---|---|
| `n` | Next frame |
| `N` | Skip 10 frames forward |
| `p` | Previous frame |
| `P` | Skip 10 frames back |
| `<` | Jump to a specific frame number |
| `s` | Show replay summary (agents, turns, winner) |
| `q` | Quit |

### 3. Dump a replay as plain text

```bash
# Print every frame as plain text (no ANSI colours) for sharing or grep
uv run python scripts/visualize.py replay data/replays/heuristic_vs_opp_mcts.json --dump

# Just the summary
uv run python scripts/visualize.py replay data/replays/heuristic_vs_opp_mcts.json --summary
```

### 4. Watch live as agents play

```bash
uv run python scripts/visualize.py run \
    --agent0 workspace/exp002_lucario_heuristic/agent.py \
    --deck0 workspace/exp002_lucario_heuristic/deck.csv \
    --agent1 workspace/exp003_lucario_mcts/agent.py \
    --deck1 workspace/exp003_lucario_mcts/deck.csv \
    --delay 0.3
```

This shows every decision as it happens, with a configurable delay between
steps. At the end it also saves a replay file if `--output` is given.

### 5. Render a single observation to text

Capture a single observation from a match dump, then render it:

```bash
uv run python scripts/visualize.py render obs.json
```

Highlight the chosen actions:

```bash
uv run python scripts/visualize.py render obs.json --actions 0 2
```

## Library API

### `render(obs, card_db=None, actions=None, visualize_json=None) -> str`

Render a single observation to an ANSI-coloured terminal string.

```python
from pokemon.visualize import render

text = render(obs_dict)
print(text)
```

Parameters:
- `obs` — raw observation dict from the `cg` engine
- `card_db` — optional `CardDB` instance (loads default if omitted)
- `actions` — optional list of chosen action indices (highlighted in output)
- `visualize_json` — optional `visualize_data()` JSON string for god-view

### `record_match(agent0, agent1, output=None, ...) -> (Replay | None, dict)`

Run a match between two agents and return a Replay + result dict.

```python
from pokemon.visualize import record_match

replay, result = record_match(
    my_agent, opponent_agent,
    agent_names=("exp005", "random"),
    output="data/replays/test.json",
    include_visualize=True,
)
print(f"Winner: P{result['winner']} in {result['turns']} turns")
```

### `Replay` — save/load

```python
from pokemon.visualize import Replay

# Save
replay.save("path/to/replay.json")

# Load
loaded = Replay.load("path/to/replay.json")
print(f"{loaded.n_frames} frames, winner P{loaded.winner}")
```

### `ReplayPlayer` — interactive navigation

```python
from pokemon.visualize import ReplayPlayer

player = ReplayPlayer(loaded)
player.next()       # advance one frame
player.next(10)     # skip 10
player.prev()       # go back
player.jump(50)     # go to frame 50
text = player.render_current()  # get formatted display
print(player.summary())
```

### `run_interactive(agent0, agent1, ...)` — live match display

```python
from pokemon.visualize import run_interactive

run_interactive(
    agent0, agent1,
    agent_names=("exp005", "exp003"),
    output="data/replays/live.json",
    step_delay=0.3,
)
```

## Data Format

The replay JSON format is versioned (`format_version: 1`) and self-contained.
Each frame contains:

| Field | Type | Description |
|---|---|---|
| `obs` | dict | Raw engine observation (decision point) |
| `actions` | list[int] | Option indices the agent chose |
| `player` | int | Which player's turn (0 or 1) |
| `turn` | int | Turn number |
| `turn_action` | int | Action number within this turn |
| `visualize_json` | str\|null | God-view snapshot (if recorded) |

The `visualize_json` field is only present when `include_visualize=True`. It
contains a JSON string with the engine's `visualize_data()` output, which
reveals both players' hands, deck contents, and prize cards.

## Comparison with Other Visualizers

| Tool | Platform | Features |
|---|---|---|
| **This tool** (`pokemon.visualize`) | CLI terminal | Record, live display, interactive replay, no browser needed |
| [PTCG_ABCS_Visualizer](https://github.com/hiro094/PTCG_ABCS_Visualizer) | Browser (HTML+JS) | Card images, drag-drop replay, EN/JP toggle, discard grid |
| [cabt-viewer](https://github.com/charlielockyer-rice/cabt-viewer) | Browser (Svelte) | Live play, replay, sample agents, card metadata |

All three tools consume the same underlying engine data. The terminal tool is
best for quick iteration and CI — no browser, no build step.

## Limitations

- **No card images** — terminal rendering shows card names and stats as text.
  For a graphical view use the browser-based visualizers linked above.
- **`run` and `record` require the `cg` engine** — needs `make sim-download`
  first. The `render` and `replay` commands work without the engine.
- **Interactive replay uses raw TTY mode** — requires a Unix-like terminal
  (Linux/macOS). Falls back to non-interactive on Windows.
- **God-view requires engine** — `visualize_data()` is only available during
  and immediately after an engine match, not from recorded replay files unless
  captured at record time.
