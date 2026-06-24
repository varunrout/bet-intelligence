# Methodology

This document explains the modelling logic behind Bet Intelligence.

## Objective

The objective is not simply to predict whether a match finishes Over 2.5 goals.

The objective is to estimate whether a model can identify useful probability differences between:

- the model's estimated probability of Over 2.5 goals
- the bookmaker or market-implied probability for Over 2.5 goals

That difference becomes relevant only if it can be converted into positive expected value after backtesting.

## Data sources

### football-data.co.uk

Used for:

- match results
- goals
- Over 2.5 settlement
- bookmaker odds
- basic team match statistics

### Understat

Used for:

- match-level expected goals
- rolling xG for and against features

Understat xG is merged back into the local DuckDB match tables using season, home team and away team matching after team name resolution.

## Target

The binary target is:

```text
over_25 = 1 if total_goals > 2.5 else 0
```

The project currently focuses on the Over 2.5 goals market because it has a simple binary settlement and enough historical liquidity to compare against market prices.

## Feature families

### Market features

Market features are derived from bookmaker odds, especially Pinnacle closing odds where available.

Examples:

- Pinnacle implied probability Over 2.5
- Pinnacle Over and Under odds
- Bet365 implied probability Over 2.5
- market average probability
- market maximum probability
- bookmaker divergence

The market-only model is the key benchmark. Any model that cannot beat this baseline has no practical betting edge.

### Rolling form features

Rolling features use only previous matches for each team.

Examples:

- goals scored average over previous 5 matches
- goals conceded average over previous 5 matches
- shots average
- shots on target average
- wins in last 5
- clean sheets in last 5
- Over 2.5 rate in last 5

The current match is never included in its own rolling feature calculation.

### xG features

Understat xG is used to add shot-quality context.

Examples:

- home xG for average over previous 5 matches
- home xG against average over previous 5 matches
- away xG for average over previous 5 matches
- away xG against average over previous 5 matches

These features are intended to capture process quality rather than only realised goals.

### Context features

Context features capture match scheduling and team conditions.

Examples:

- home rest days
- away rest days
- rest differential
- matches played in the previous 14 days
- Elo rating
- Elo differential

## Validation strategy

The project uses expanding-window walk-forward validation.

This is chosen because betting data is temporal and market behaviour changes across seasons. Random splits would allow future data to influence model selection and would overstate real-world performance.

Current fold structure:

```text
Fold 1: train 2019/20                          -> test 2020/21
Fold 2: train 2019/20, 2020/21                 -> test 2021/22
Fold 3: train 2019/20, 2020/21, 2021/22        -> test 2022/23
Fold 4: train 2019/20, 2020/21, 2021/22, 2022/23 -> test 2023/24
```

## Evaluation metrics

### ROC-AUC

Measures rank-order discrimination. Useful for comparing model signal, but not sufficient for betting.

### Brier score

Measures probability accuracy. Lower is better.

### Brier Skill Score

Measures improvement versus a naive constant-rate forecast. Positive values indicate skill versus the base-rate baseline.

### Log loss

Penalises overconfident wrong predictions. Useful for identifying poorly calibrated models.

### Calibration

Calibration is essential because betting decisions depend on probability levels, not just ranking.

A model can have acceptable ROC-AUC and still be unusable for staking if its probabilities are miscalibrated.

## Betting evaluation layer

The next project layer converts model probabilities into betting decisions.

Core calculations:

```text
market_probability = 1 / decimal_odds
edge = model_probability - market_probability
expected_value = model_probability * decimal_odds - 1
```

A bet should only be considered when expected value is positive and the edge exceeds a chosen threshold.

## Backtesting philosophy

The project should report unflattering results clearly.

A useful backtest must include:

- total bets
- hit rate
- ROI
- yield
- drawdown
- season-level performance
- odds-band performance
- edge-bucket performance
- comparison with market baseline

The final aim is not to force a profitable result. The aim is to build an honest intelligence system that shows whether any edge exists.
