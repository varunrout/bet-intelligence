# Bet Intelligence

Bet Intelligence is a football betting research project for the English Premier League Over 2.5 goals market.

The project ingests historical match, odds and xG data, stores it in DuckDB, builds leakage-safe pre-match features, benchmarks predictive models against market odds, and prepares the foundation for value-betting backtests.

The core question is deliberately strict:

> Can public football data and engineered pre-match features add useful signal beyond the closing betting market?

The answer is not assumed. The project is designed to test it honestly.

## Current scope

- Competition: English Premier League
- Seasons: 2019/20 to 2023/24
- Market: Over 2.5 goals
- Main benchmark: Pinnacle closing implied probability
- Extra signal sources: rolling team form, shots, xG, rest days, congestion and Elo
- Validation: expanding-window walk-forward cross-validation

## Project structure

```text
bet-intelligence/
  config/                 League and team alias configuration
  data/                   Local data area, ignored by Git
  docs/                   Methodology, roadmap and data dictionary
  notebooks/              Exploratory analysis and research evidence
  reports/                Generated figures and written outputs
  scripts/                CLI entry points for ETL, features and xG ingestion
  src/                    Reusable project code
    analysis/             Descriptive analysis helpers
    features/             Feature engineering pipeline
    ingestion/            External data loaders
    modeling/             Model preparation, evaluation and validation
    transform/            Raw data normalisation
    utils/                Config, database and time utilities
  tests/                  Regression and leakage guard tests
```

## Why this project is different from a simple prediction model

A model that predicts football outcomes is not automatically useful for betting.

A betting model must answer three harder questions:

1. Is the model better than a strong market baseline?
2. Are its probabilities calibrated enough to support staking decisions?
3. Does the model create positive expected value after realistic betting rules?

This repo is being built around those three questions.

## Reproducible pipeline

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Initialise the local DuckDB database:

```bash
python scripts/init_db.py
```

Run the football-data.co.uk ingestion pipeline:

```bash
python scripts/run_pipeline.py
```

Fetch Understat xG and rebuild features:

```bash
python scripts/fetch_xg.py --all
python scripts/build_features.py --rebuild
```

Run tests:

```bash
pytest -q
```

A convenience `Makefile` is included for the most common commands:

```bash
make setup
make init-db
make ingest
make fetch-xg
make features
make test
```

## Modelling approach

The modelling layer uses an expanding-window walk-forward design:

```text
Fold 1: train 2019/20                          -> test 2020/21
Fold 2: train 2019/20, 2020/21                 -> test 2021/22
Fold 3: train 2019/20, 2020/21, 2021/22        -> test 2022/23
Fold 4: train 2019/20, 2020/21, 2021/22, 2022/23 -> test 2023/24
```

This avoids random train/test splits, which would be inappropriate for market and time-series data.

Core model groups:

- `market_only`: Pinnacle closing implied probability baseline
- `form_only`: rolling football performance features without market prices
- `all`: market, form, xG, context and Elo features together

## Leakage policy

No model feature should use match information that would not be available before kickoff.

The repo includes leakage tests for:

- rolling feature construction
- odds snapshot timing
- target column isolation
- temporal split integrity

Any leakage test failure should be treated as a blocking issue.

## Next build phases

The project is now moving from modelling research into a complete betting intelligence system.

Planned phases:

1. Betting edge engine
2. Backtesting framework
3. Model registry and training CLI
4. Generated reporting layer
5. Optional Streamlit dashboard

See `docs/ROADMAP.md` for the full completion plan.

## Important note

This project is for research, modelling and portfolio demonstration. It is not betting advice. Historical backtests do not guarantee future performance.
