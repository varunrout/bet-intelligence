# bet-intelligence

**A football betting-market intelligence system: converting bookmaker odds into fair probabilities and testing whether models can be correctly priced, not just accurate.**

Built on English Premier League data (2019/20–2023/24), this project treats the betting market as the benchmark every model must beat. Bookmaker odds are converted to implied probabilities, market margin is removed to produce fair-probability estimates, and every model is evaluated on calibration, log loss and Brier score against those market baselines — because in a market with a real price, being close is worthless if you are not correctly priced.

## What this project demonstrates

- Market-aware probability modelling: implied probabilities, margin removal, bookmaker agreement and market-efficiency analysis.
- Leakage-safe feature engineering: rolling form, contextual and market features built strictly from information available before kick-off, with automated leakage tests.
- Walk-forward validation: models are trained and evaluated season-by-season in chronological order, never on shuffled splits.
- Calibration-first evaluation: reliability curves, log loss and Brier score against market benchmarks, not raw accuracy.

## Architecture

```
data/raw (football-data.co.uk CSVs, Understat xG)
   │  src/ingestion    — fdco_loader, understat_loader
   ▼
DuckDB (data/db)       — src/utils/db, scripts/init_db.py
   │  src/transform    — odds_transformer (margin removal),
   │                     match_transformer, team_resolver
   ▼
src/features           — market_features, rolling_form,
   │                     context_features, pipeline
   ▼
src/modeling           — walk_forward, features, evaluate
   ▼
reports/figures        — ~60 generated analysis figures
```

## Pipeline

```bash
pip install -r requirements.txt
python scripts/init_db.py        # build the DuckDB database from raw data
python scripts/run_pipeline.py   # ingest + transform
python scripts/build_features.py # leakage-safe feature tables
python scripts/fetch_xg.py       # Understat xG enrichment
pytest                           # run the test suite
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_descriptive_eda.ipynb` | Market overview, outcome distributions, seasonal patterns |
| `02_odds_analysis.ipynb` | Margin analysis, bookmaker agreement, calibration of market prices, market efficiency by probability decile |
| `03_feature_engineering_review.ipynb` | Feature coverage, correlations, Elo features, rest days and congestion |
| `04_baseline_model.ipynb` | Baseline models under walk-forward validation, calibration and AUC by season |
| `04b_model_improvements.ipynb` | Model comparison, feature-shift analysis, calibration comparison |

## Testing

| Test | Guards against |
|---|---|
| `test_feature_leakage.py` | Any feature using information from after kick-off |
| `test_odds_transformer.py` | Margin-removal and implied-probability correctness |
| `test_rolling_form.py` | Rolling windows peeking at the current match |
| `test_team_resolver.py` | Team-name mismatches across data sources |

## Data

- Match results and closing odds: [football-data.co.uk](https://www.football-data.co.uk/) (Premier League, 2019/20–2023/24)
- Expected goals: [Understat](https://understat.com/)
- Storage: DuckDB (`data/db/bet_intelligence.duckdb`), rebuilt locally via `scripts/init_db.py`

## In development

- `src/evaluation` and `src/prescriptive` — the decision layer: position selection rules, stake sizing, realised return, drawdown and bankroll exposure, framed as uncertainty-aware decision support rather than a promise of guaranteed edge.
- Closing-line-value analysis against model probabilities.

## Honest limitations

- Single league, five seasons — findings may not generalise across competitions.
- Closing odds only; no intra-market price movement.
- Evaluation is held-out and chronological, not causal, and no live betting is performed or implied.
