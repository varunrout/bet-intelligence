# Data Dictionary

This document describes the main entities used by Bet Intelligence.

The project stores data in a local DuckDB database. Table definitions live in `scripts/init_db.py`.

## competitions

Competition reference table.

Important fields:

| Column | Meaning |
|---|---|
| competition_id | Internal competition identifier |
| name | Full competition name |
| short_code | Project short code, for example `EPL` |
| country | Competition country |
| tier | League tier |

## teams

Canonical team reference table.

Important fields:

| Column | Meaning |
|---|---|
| team_id | Internal team identifier |
| name_canonical | Single project-wide team name |
| short_code | Display abbreviation |

Team aliases are controlled in `config/team_aliases.yaml`.

## matches

One row per football match.

Important fields:

| Column | Meaning |
|---|---|
| match_id | Stable match identifier |
| competition_id | Link to competition |
| season | Season label, for example `2023/24` |
| kickoff_utc | Match kickoff timestamp |
| home_team_id | Home team identifier |
| away_team_id | Away team identifier |
| goals_home | Home goals |
| goals_away | Away goals |
| total_goals | Total goals |
| over_25 | Binary target for Over 2.5 goals |
| result_ftr | Full-time result |

## team_match_stats

One row per team per match.

Important fields:

| Column | Meaning |
|---|---|
| stat_id | Internal row identifier |
| match_id | Match identifier |
| team_id | Team identifier |
| is_home | Whether the row is for the home team |
| shots | Total shots |
| shots_on_target | Shots on target |
| corners | Corners |
| fouls | Fouls |
| yellow_cards | Yellow cards |
| red_cards | Red cards |
| xg_for | Team expected goals |
| xg_against | Opponent expected goals |

`xg_for` and `xg_against` are populated after running the Understat xG ingestion script.

## bookmakers

Bookmaker reference table.

Important fields:

| Column | Meaning |
|---|---|
| bookmaker_id | Internal bookmaker identifier |
| name | Bookmaker name |
| short_code | Source short code, for example `P` or `B365` |
| is_sharp | Whether bookmaker is treated as sharp market reference |

## odds_snapshots

Historical bookmaker odds data.

Important fields:

| Column | Meaning |
|---|---|
| snapshot_id | Internal odds row identifier |
| match_id | Match identifier |
| bookmaker_id | Bookmaker identifier |
| market_type | Market name, currently `ou25` |
| odds_over | Decimal odds for Over 2.5 |
| odds_under | Decimal odds for Under 2.5 |
| implied_prob_over | Over implied probability |
| implied_prob_under | Under implied probability |
| margin | Bookmaker overround/margin |
| snapshot_type | Opening, closing or source-specific snapshot label |

## engineered_features

One row per match and feature version.

Important fields:

| Column | Meaning |
|---|---|
| feature_id | Internal row identifier |
| match_id | Match identifier |
| computed_at_utc | Timestamp used to represent feature availability |
| feature_version | Feature set version |
| home_goals_scored_avg5 | Home team previous 5 match goals scored average |
| home_goals_conceded_avg5 | Home team previous 5 match goals conceded average |
| away_goals_scored_avg5 | Away team previous 5 match goals scored average |
| away_goals_conceded_avg5 | Away team previous 5 match goals conceded average |
| home_xg_for_avg5 | Home team previous 5 match xG for average |
| home_xg_against_avg5 | Home team previous 5 match xG against average |
| away_xg_for_avg5 | Away team previous 5 match xG for average |
| away_xg_against_avg5 | Away team previous 5 match xG against average |
| home_rest_days | Home team rest days before kickoff |
| away_rest_days | Away team rest days before kickoff |
| rest_differential | Home rest days minus away rest days |
| home_elo | Home team Elo before match |
| away_elo | Away team Elo before match |
| elo_differential | Home Elo minus away Elo |
| pin_implied_prob_over | Pinnacle implied probability for Over 2.5 |
| b365_implied_prob_over | Bet365 implied probability for Over 2.5 |
| avg_implied_prob_over | Market average implied probability |
| pin_b365_divergence | Difference between Pinnacle and Bet365 probabilities |

## Target leakage rule

The following columns must never be used as model features:

- `goals_home`
- `goals_away`
- `total_goals`
- `over_25`
- `result_ftr`

These are outcomes, not pre-match inputs.
