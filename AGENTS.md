# AGENTS.md — kcom-pokemon

Kaggle Simulation + Strategy competitions. Build a Pokemon TCG AI agent.

## Commands

All Python **must** be prefixed with `uv run` (`.venv` not on PATH).

| Command | What it does |
|---|---|
| `make install` | `uv sync --extra dev` + editable install + auth check |
| `make download` | Fetch card CSVs to `data/raw/` |
| `make sim-download` | Fetch `cg` game engine to `data/sim_sample/` (join competition first) |
| `make test` | `uv run pytest tests/ -v` |
| `make test ARGS="-k name -x"` | Focused test |
| `make lint` | `ruff check src/ scripts/ tests/` (no autofix) |
| `make format` | `ruff format --check` (no rewrite) |
| `make format-fix` | Apply ruff formatting |
| `make gauntlet` | Run agent tournament (requires SDK) |
| `make build-submit ARGS="--agent ... --deck ..."` | Package `.tar.gz` |
| `make submit` | Upload to Kaggle + show leaderboard |

Verification loop: `make lint && make format && make test`. No typechecker configured.

## Architecture

- **Package**: `src/pokemon/` — `agent.py` (ABC + baselines), `deck.py`, `data.py`, `harness.py`, `tracking.py`.
- **Agent interface**: `agent(obs_dict) -> list[int]`.
  - First call (`obs["select"] is None`): return 60-card deck (Card ID list).
  - Subsequent calls: return choice indices (`minCount ≤ len ≤ maxCount`, no duplicates).
- **Game engine**: `cg` package (compiled `.so`/`.dll`, ctypes wrapper) in `data/sim_sample/cg/`. gitignored.
- **Search API**: `search_begin`/`search_step`/`search_end` for forward lookahead (MCTS, alpha-beta).
- **Rating**: TrueSkill μ/σ (μ₀=600), win/loss only, latest 2 submissions count.
- **Experiments**: `workspace/expNNN_name/` with `agent.py`, `deck.csv`, `SESSION_NOTES.md`.

## Agent Development

```python
from pokemon.agent import RuleBasedAgent

class MyAgent(RuleBasedAgent):
    def __init__(self, deck, random_seed=42):
        super().__init__(deck=deck, random_seed=random_seed)

    def _act(self, obs: dict) -> list[int]:
        return [obs["options"][0]]
```

Subclass `RuleBasedAgent` or `Agent` directly. Agents must be registered in `scripts/gauntlet.py` to participate in tournaments.

## Key gotchas

- `from __future__ import annotations` used in all source files.
- Ruff: line-length 100, `target-version = "py311"`, rules E/F/I.
- Python 3.12 (`.python-version`), `requires-python = ">=3.11"`.
- Deck must be exactly **60 cards** (Kaggle enforces).
- `make build-submit` copies your `agent.py` → `main.py` in the tarball; the `--agent` path is the source file.
- `harness.run_gauntlet` does round-robin with swapped sides — each ordered pair plays `n_matches` games.
- Kaggle auth: `.kaggle/access_token` (chmod 600) or env var `KAGGLE_API_TOKEN`. Also supports legacy `~/.kaggle/kaggle.json`.
- You **must join** both competitions on Kaggle before `make download` / `make sim-download` (otherwise 403).
- No CI / GitHub Actions configured.

## Testing

- Tests use mock observations — no game engine required.
- `tests/test_agent.py` — agent interface, RandomAgent, RuleBasedAgent.
- `tests/test_deck.py` — deck builder and card data loading.
- `tests/test_integration.py` — full pipeline (agent + deck + tracking).

## Gitignored

`data/raw/*.csv`, `data/sim_sample/`, `submit/`, `workspace/results/`, `*.joblib`, `*.pkl`, `*.cbm`, `*.tar.gz`, `.kaggle/*` (except `*.example`), `.ai/*`, `.venv/`, caches.

## References

- **`Iteration.md`** — experiment log, suggested directions, workflow reference. Read before starting new experiments.
