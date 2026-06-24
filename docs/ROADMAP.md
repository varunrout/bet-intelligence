# Completion Roadmap

This roadmap turns Bet Intelligence from a research repo into a complete portfolio-grade betting intelligence system.

## Current state

Already implemented:

- DuckDB schema
- football-data.co.uk ingestion
- match, odds and team stats transformation
- team alias resolution
- leakage-safe feature engineering
- Understat xG enrichment
- walk-forward validation utilities
- baseline model notebooks
- leakage tests

Main gap:

- the project needs a betting intelligence layer, backtesting, reporting and reproducible project polish.

## Phase 5: Betting edge engine

Goal: convert model probabilities into value-betting signals.

Deliverables:

- `src/betting/edge.py`
- edge calculation
- expected value calculation
- bet flagging rules
- flat stake sizing
- fractional Kelly stake sizing
- unit tests for edge, EV and staking logic

Acceptance criteria:

- model probability and bookmaker odds produce a deterministic betting signal
- invalid odds and probabilities are handled safely
- no staking strategy can produce negative stakes
- tests cover profitable, non-profitable and invalid-input cases

## Phase 6: Backtesting framework

Goal: evaluate whether historical betting signals produce useful strategy performance.

Deliverables:

- `src/betting/backtest.py`
- `scripts/run_backtest.py`
- season-level and total performance summaries
- profit curve and drawdown calculations
- edge-threshold filtering

Acceptance criteria:

- historical predictions can be converted into bet-level results
- output includes bets, hit rate, profit, ROI, yield and max drawdown
- backtests can be grouped by season
- no future information is used in bet selection

## Phase 7: Model consolidation

Goal: move model execution from notebooks into reusable scripts and modules.

Deliverables:

- `src/modeling/train.py`
- `src/modeling/predict.py`
- `src/modeling/calibration.py`
- `scripts/train_models.py`
- `scripts/evaluate_models.py`

Acceptance criteria:

- models can be trained without opening notebooks
- model outputs can be saved and reused by the betting layer
- calibration method is explicit and tested

## Phase 8: Reporting layer

Goal: generate human-readable evidence for the project.

Deliverables:

- `reports/model_performance.md`
- `reports/backtest_results.md`
- generated plots for calibration, edge distribution, profit curve and drawdown
- project summary written in portfolio language

Acceptance criteria:

- a reader can understand the modelling result without opening notebooks
- charts are generated from saved outputs
- results distinguish between predictive performance and betting performance

## Phase 9: Optional dashboard

Goal: make the system interactive.

Deliverables:

- `app/streamlit_app.py`
- filters for season, team, odds range and edge threshold
- model performance view
- betting backtest view
- feature importance view

Acceptance criteria:

- app runs locally from a single command
- it does not require re-running the full pipeline on launch
- it uses saved outputs or local DB tables

## Phase 10: Final project polish

Goal: make the repo defensible for recruiters, reviewers and future development.

Deliverables:

- clean README
- methodology documentation
- data dictionary
- CI workflow
- Makefile
- final project report
- limitations section

Acceptance criteria:

- project can be cloned, installed and tested
- pipeline commands are documented
- limitations are honest and visible
- tests pass in CI

## Suggested PR sequence

1. Project documentation and reproducibility
2. Betting edge engine
3. Backtesting framework
4. Model training and prediction CLI
5. Reporting and final project polish

This order keeps the project reviewable and avoids one oversized PR.
