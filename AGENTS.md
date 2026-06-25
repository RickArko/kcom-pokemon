# AGENTS.md — kcom-pokemon

Kaggle Competition: [Pokemon TCG AI Battle Challenge Strategy](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy)
Predict the winner of a Pokemon TCG battle from in-game state.
Metric: **accuracy**. Target column: `winner`.

## Commands

All Python invocations **must** be prefixed with `uv run` (uv manages the env;
`.venv` exists but is not on PATH — bare `python`/`pytest` will fail or hit the
wrong interpreter).

| Command | Purpose |
|---|---|
| `make install` | `uv sync --extra dev` + editable install + kaggle auth check |
| `make download` | Fetch/expand competition CSVs into `data/` (requires auth) |
| `make train CONFIG=config/foo.yaml RUN_NAME=bar` | Train ensemble & save a run |
| `make train ARGS="--flag x"` | Pass extra args to `scripts/train.py` |
| `make predict` | `uv run python scripts/predict.py $(ARGS)` |
| `make test` | `uv run pytest tests/ -v` |
| `make test ARGS="-k name -x"` | Focused test run |
| `make lint` | `ruff check src/ scripts/ tests/` (check only, no autofix) |
| `make format` | `ruff format ... --check` (check only — does NOT rewrite) |
| `make format-fix` | Apply ruff formatting |
| `make submit SUBMISSION_FILE=... SUBMISSION_MSG="..."` | Upload to Kaggle + show leaderboard |

Verification loop after changes: `make lint && make format && make test`.
No typechecker / mypy configured.

## Architecture

- **Package**: `src/pokemon/` — `data.py` (loaders), `features.py`
  (`StrategyFeatureEngineer`), `models.py` (`StackingEnsemble`), `tracking.py`
  (run logger). Installed editable via hatchling (`packages = ["src/pokemon"]`).
- **Ensemble**: LGBM + XGBoost + CatBoost base models → LogisticRegression
  meta-model on out-of-fold probabilities. Stratified k-fold CV.
- **Config-driven**: YAML in `config/` controls features, CV, and every model
  hyperparam. `config/config.yaml` = tuned default (5-fold, 1000 estimators);
  `config/baseline.yaml` = fast reference (3-fold, 250 estimators).
- **Experiments** live in `config/experiments/` — copy an existing config to
  start a new one.
- **Each `make train`** creates `outputs/runs/<timestamp>_<name>/` containing a
  frozen `config.yaml`, `metrics.json`, `models/ensemble.joblib`, and
  `submission.csv`. Also writes the canonical `outputs/submissions/submission.csv`.
- **Re-predict without retraining**: `uv run python scripts/predict.py --run-dir outputs/runs/<name>`
- **Compare runs**: `uv run python scripts/compare.py` (reads `outputs/runs/`).

## Feature Engineering — `StrategyFeatureEngineer`

The engineer performs three steps in order:

1. **Drop** low-signal ID/metadata columns (`drop_cols`).
2. **Diff pairs** (`diff_pairs`): for each `(col_a, col_b)`, create
   `col_a_col_b = col_a - col_b`.  Only pairs where both columns exist are
   computed; missing pairs are silently skipped.
3. **Interaction pairs** (`interaction_pairs`): for each `(col_a, col_b)`,
   create `col_a_x_col_b = col_a * col_b`.  Both operand columns must survive
   step 1 or be produced by step 2.  Invalid pairs warn and are skipped.
4. **Encode categoricals** (`cat_cols`, `encoding`):
   - `"ohe"` (default) — OneHotEncoder; new columns named `{col}_{value}`.
   - `"label"` — LabelEncoder → int32.
   - `"passthrough"` — no transformation.
   If `cat_cols=None`, all remaining object-dtype columns are auto-encoded.
   If `cat_cols=[]`, no encoding is applied (caller must ensure no string
   columns remain, or models will fail).

### Default diff pairs (applied when data has these columns)

| Pair | Resulting column | Meaning (positive = player advantage) |
|---|---|---|
| `player_active_pokemon_hp`, `opponent_active_pokemon_hp` | `player_active_pokemon_hp_opponent_active_pokemon_hp` | Player HP lead |
| `player_bench_count`, `opponent_bench_count` | `player_bench_count_opponent_bench_count` | Bench size lead |
| `player_hand_count`, `opponent_hand_count` | `player_hand_count_opponent_hand_count` | Hand size lead |
| `player_deck_count`, `opponent_deck_count` | `player_deck_count_opponent_deck_count` | Deck size lead |
| `opponent_prize_count`, `player_prize_count` | `opponent_prize_count_player_prize_count` | Prizes-taken lead (opp taken − player taken → positive = player has taken fewer → player is behind on prizes) |

## Conventions & gotchas

- `encoding` param: `"ohe"` (default), `"label"` (LabelEncoder → int32),
  `"passthrough"` (raw strings).  Set via `features.encoding` in config.
- `cat_cols=[]` explicitly disables encoding; ensure no object-dtype columns
  survive or sklearn models will raise `ValueError`.
- `catboost_info/` is written during training and is gitignored — safe to delete.
- `from __future__ import annotations` is used in all source files.
- Ruff: line-length 100, `target-version = "py311"`, rules E/F/I.
- Runtime: Python 3.12 (`.python-version`); `requires-python = ">=3.11"`.

## Testing

- Tests use **synthetic battle-state data** (the `synthetic_battle_data` fixture in
  `tests/test_integration.py`) — `make test` works offline, no Kaggle download needed.
- `tests/test_models.py` covers the feature transformer, ensemble, and
  save/load roundtrip; `tests/test_integration.py` covers the end-to-end
  pipeline + submission format.

## Kaggle auth

- Token file: `.kaggle/access_token` (chmod 600), or env var `KAGGLE_API_TOKEN`.
  `make download` / `make submit` read the file first, then fall back to the env var.
  Legacy `~/.kaggle/kaggle.json` also works.
- You must **join the competition** (Accept Rules) on the Kaggle web page before
  `make download` works — otherwise it 403s.

## Gitignored artifacts (won't appear in `git status`)

`data/*.csv` and `*.zip`, `outputs/submissions/*` (except `.gitkeep`),
`outputs/runs/`, `models/`, `*.joblib`/`*.pkl`/`*.cbm`, `catboost_info/`,
`.kaggle/*` (except `*.example`), `.ai/*`, `.venv/`, caches. Run data lives only
on disk — re-run `make download` on a fresh clone.

## Experiment state

No experiments have been run yet.  Start with:

```bash
make train CONFIG=config/baseline.yaml RUN_NAME=baseline
uv run python scripts/compare.py
```

Read **`Iteration.md`** before running new experiments to avoid repeating
completed work and to understand the current best direction.
