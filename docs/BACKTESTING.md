# Betting Backtesting

The backtesting layer evaluates whether model probabilities would have produced useful historical betting decisions.

It sits after modelling:

```text
features -> model probabilities -> edge engine -> settled bets -> backtest metrics
```

## Required inputs

A backtest input CSV needs at least three columns:

| Column type | Example | Meaning |
|---|---|---|
| Model probability | `y_prob` | Estimated probability of Over 2.5 |
| Decimal odds | `pin_odds_over` | Available decimal odds for Over 2.5 |
| Settled target | `over_25` | 1 if the match finished Over 2.5, else 0 |

Optional but recommended columns:

| Column | Use |
|---|---|
| `season` | Season-level reporting |
| `kickoff_utc` | Chronological ordering for profit curve |
| `match_id` | Traceability back to source data |
| `home_team`, `away_team` | Manual review and diagnostics |

## Core calculations

```text
market_prob = 1 / decimal_odds
edge = model_prob - market_prob
ev = model_prob * decimal_odds - 1
```

A bet is selected when:

```text
edge >= min_edge
and ev >= min_ev
and min_odds <= decimal_odds <= max_odds, if max_odds is supplied
```

## CLI example

```bash
python scripts/run_backtest.py \
  --input outputs/predictions/oof_predictions.csv \
  --model-prob-col y_prob \
  --odds-col pin_odds_over \
  --target-col over_25 \
  --sort-col kickoff_utc \
  --group-col season \
  --min-edge 0.02 \
  --min-ev 0.00 \
  --stake-strategy flat \
  --flat-stake 1.0 \
  --output-dir outputs/backtests
```

## Outputs

The script writes:

| File | Purpose |
|---|---|
| `backtest_bets.csv` | Row-level betting decisions and realised profit |
| `profit_curve.csv` | Bet-only equity curve data |
| `backtest_summary.json` | Headline metrics |
| `backtest_report.md` | Human-readable Markdown summary |
| `backtest_group_summary.csv` | Optional group summary if `--group-col` is provided |

## Metrics

| Metric | Meaning |
|---|---|
| bets | Number of selected bets |
| wins | Profitable settled bets |
| losses | Losing settled bets |
| stake | Total staked units |
| profit | Net units won or lost |
| roi | Profit divided by total stake |
| hit_rate | Wins divided by bets |
| avg_odds | Average odds of selected bets |
| avg_edge | Average model edge on selected bets |
| max_drawdown | Worst peak-to-trough loss in the realised profit curve |

## Interpretation rules

A positive backtest is not enough on its own.

The result should be checked for:

- enough bet count
- stable season-level performance
- sensible calibration
- limited drawdown
- no dependence on one extreme odds band
- no hidden leakage or future data

The aim is not to force profit. The aim is to determine whether the model's probability estimates are useful against the market.
