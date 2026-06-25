# Iteration Workflow

A structured workflow for running, tracking, and comparing model experiments for the
[Pokemon TCG AI Battle Challenge Strategy](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy) competition.

---

## Quick Start

```bash
make install          # dependencies + kaggle auth (one-time)
make download         # fetch competition data (one-time)

# Run your first experiment
make train CONFIG=config/baseline.yaml RUN_NAME=my_first_experiment

# Compare all experiments
uv run python scripts/compare.py
```

---

## How It Works

### 1. Config-driven experiments

Every experiment is defined by a single YAML file.  The config controls every
knob: which features to use, how many CV folds, every model hyperparameter.

```bash
make train CONFIG=config/experiments/v001_label_encoding.yaml RUN_NAME=v001
```

### 2. Automatic run tracking

Each `make train` creates a timestamped directory under `outputs/runs/`:

```
outputs/runs/
  20260625_131300_baseline/
    config.yaml          # frozen copy of the config used
    metrics.json         # OOF scores, params, wall time
    models/ensemble.joblib  # serialised ensemble (loadable)
    submission.csv       # competition submission
```

This makes every experiment **reproducible** — you can re-run from the saved
config, or re-predict from the saved model.

### 3. Compare experiments

```bash
uv run python scripts/compare.py

# Example output:
#                              run  overall_oof_score  ...  n_features
#    20260625_131519_v001_label_enc            0.7456  ...          22
#               20260625_131300_baseline       0.7412  ...          18
```

### 4. Re-predict from a saved model (no re-training)

```bash
uv run python scripts/predict.py --run-dir outputs/runs/20260625_131300_baseline
```

---

## Running the Baseline

The baseline is the "original" pipeline before any iteration:

```bash
make train CONFIG=config/baseline.yaml RUN_NAME=baseline
```

| Setting | Value |
|---|---|
| Features | Raw game-state columns + 5 advantage diffs + OHE deck/pokemon names |
| Models | LGBM + XGBoost + CatBoost (stacked with LogisticRegression) |
| CV | 3-fold stratified (for speed; use 5 for final) |
| Estimators per model | 250 |

---

## Iteration Log

*(No experiments have been run yet. Add entries here after each run.)*

---

## Workflow Reference

### Run an experiment

```bash
make train CONFIG=config/experiments/my_config.yaml RUN_NAME=my_experiment
```

### Compare results

```bash
uv run python scripts/compare.py
uv run python scripts/compare.py --sort-by elapsed_seconds
```

### Re-predict from a saved model

```bash
uv run python scripts/predict.py --run-dir outputs/runs/20260625_131300_baseline
```

### Submit to Kaggle

```bash
make submit
make submit SUBMISSION_FILE=outputs/runs/20260625_131300_baseline/submission.csv \
             SUBMISSION_MSG="baseline: advantage diffs + OHE deck names"
```

### Create a new experiment config

1. Copy an existing config: `cp config/baseline.yaml config/experiments/my_idea.yaml`
2. Edit the feature / model / CV sections
3. Run it: `make train CONFIG=config/experiments/my_idea.yaml`

---

## Suggested First Experiments

### v001 — Label encoding for deck/pokemon names

**Hypothesis:** OHE creates many sparse columns for deck and pokemon names.
Label encoding reduces dimensionality and may improve ensemble stacking.

**Change:** Set `encoding: label` in config.

### v002 — Interaction features

**Hypothesis:** `turns × prize_diff` captures late-game pressure; `hp_diff × bench_diff`
captures board dominance.

**Change:** Add `interaction_pairs` to config.

```yaml
features:
  interaction_pairs:
    - [turns, opponent_prize_count_player_prize_count]
    - [player_active_pokemon_hp_opponent_active_pokemon_hp, player_bench_count_opponent_bench_count]
```

### v003 — Tune hyperparameters

**Hypothesis:** Lower learning rate + more estimators improves generalisation.

**Change:** Increase `n_estimators` to 500–1000, lower `learning_rate` to 0.03.

### v004 — Simple average meta-model

**Hypothesis:** LogisticRegression meta-model may overfit OOF probabilities.
A simple average of base-model predictions might generalise better.

**Change:** Set `meta.model: simple_average` in config.

---

## Next Directions (to try)

- **More base-model diversity** — ExtraTrees, HistGBM for diversity in the ensemble.
- **Per-class threshold tuning** — optimize decision boundary for imbalanced classes.
- **Feature selection** — use LGBM feature importance to drop low-signal columns.
- **Adversarial validation** — check for train/test distribution shift.
