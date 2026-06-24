# Model Training Workflow

This document describes the script-based modelling workflow.

The aim is to move repeatable model execution out of notebooks and into reusable code.

## Model registry

Registered models live in:

```text
src/modeling/model_registry.py
```

Each model has:

- a model name
- a feature set
- a fresh estimator factory
- a short description

Current registered models:

| Model | Feature set | Purpose |
|---|---|---|
| `market_logistic` | `market_only` | Pinnacle market baseline |
| `form_logistic` | `form_only` | Public football signal without market prices |
| `all_logistic` | `all` | Market plus football features in a simple model |
| `all_lgbm_regularized` | `all` | Conservative LightGBM full-feature model |

## Walk-forward export

Run:

```bash
python scripts/train_walk_forward.py \
  --models market_logistic form_logistic all_logistic \
  --output outputs/predictions/oof_predictions.csv
```

The script loads `engineered_features`, joins the target from `matches`, runs expanding-window walk-forward validation and writes out-of-fold predictions.

## Output columns

The prediction export includes:

| Column | Meaning |
|---|---|
| `model` | Model name |
| `feature_set` | Feature set used by the model |
| `test_season` | Walk-forward test season |
| `y_true` | Settled target |
| `y_prob` | Model probability |
| `match_id` | Source match identifier |
| `season` | Match season |
| `kickoff_utc` | Kickoff timestamp |
| `over_25` | Settled Over 2.5 target |
| `pin_odds_over` | Pinnacle Over 2.5 odds, if present |
| `pin_implied_prob_over` | Pinnacle implied probability, if present |

## Backtest handoff

The exported CSV can be used by the backtest layer:

```bash
python scripts/run_backtest.py \
  --input outputs/predictions/oof_predictions.csv \
  --model-prob-col y_prob \
  --odds-col pin_odds_over \
  --target-col over_25 \
  --sort-col kickoff_utc \
  --group-col season
```

## Why this matters

Notebooks are useful for exploration, but the final project needs scripts that can be rerun reliably.

This workflow makes the model outputs reproducible and prepares the project for generated reports and dashboard views.
