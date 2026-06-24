"""
CLI entry point for historical betting backtests.

This script expects a CSV containing at least:

- a model probability column
- a decimal odds column
- a settled binary target column

Example
-------
python scripts/run_backtest.py \
  --input outputs/predictions/oof_predictions.csv \
  --model-prob-col y_prob \
  --odds-col pin_odds_over \
  --target-col over_25 \
  --group-col season \
  --output-dir outputs/backtests
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.betting.backtest import profit_curve, run_backtest, summarize_backtest, summarize_by_group
from src.betting.edge import EdgePolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def write_markdown_report(
    output_path: Path,
    summary: dict,
    group_summary: pd.DataFrame | None = None,
) -> None:
    """Write a compact Markdown report for a completed backtest."""
    lines = [
        "# Betting Backtest Results",
        "",
        "## Headline Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    for key, value in summary.items():
        lines.append(f"| {key} | {value} |")

    if group_summary is not None and not group_summary.empty:
        lines.extend(["", "## Group Summary", ""])
        lines.append(group_summary.to_markdown(index=False))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a historical betting backtest from a CSV.")
    parser.add_argument("--input", required=True, type=Path, help="Input CSV path.")
    parser.add_argument("--output-dir", default=Path("outputs/backtests"), type=Path)
    parser.add_argument("--model-prob-col", required=True, help="Model probability column.")
    parser.add_argument("--odds-col", required=True, help="Decimal odds column.")
    parser.add_argument("--target-col", required=True, help="Settled binary target column.")
    parser.add_argument("--sort-col", default=None, help="Optional chronological sort column.")
    parser.add_argument("--group-col", default=None, help="Optional group summary column, e.g. season.")
    parser.add_argument("--min-edge", type=float, default=0.02, help="Minimum probability edge.")
    parser.add_argument("--min-ev", type=float, default=0.0, help="Minimum expected value.")
    parser.add_argument("--min-odds", type=float, default=1.01, help="Minimum decimal odds.")
    parser.add_argument("--max-odds", type=float, default=None, help="Maximum decimal odds.")
    parser.add_argument(
        "--stake-strategy",
        choices=["flat", "kelly"],
        default="flat",
        help="Stake sizing strategy.",
    )
    parser.add_argument("--flat-stake", type=float, default=1.0, help="Flat stake amount.")
    parser.add_argument("--bankroll", type=float, default=100.0, help="Reference bankroll for Kelly.")
    parser.add_argument("--kelly-multiplier", type=float, default=0.25, help="Fraction of full Kelly.")
    parser.add_argument(
        "--max-bankroll-fraction",
        type=float,
        default=0.05,
        help="Kelly stake cap as fraction of bankroll.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading predictions from %s", args.input)
    df = pd.read_csv(args.input)

    policy = EdgePolicy(
        min_edge=args.min_edge,
        min_ev=args.min_ev,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
    )

    backtest_df = run_backtest(
        df,
        model_prob_col=args.model_prob_col,
        odds_col=args.odds_col,
        target_col=args.target_col,
        policy=policy,
        stake_strategy=args.stake_strategy,
        flat_stake_amount=args.flat_stake,
        bankroll=args.bankroll,
        kelly_multiplier=args.kelly_multiplier,
        max_bankroll_fraction=args.max_bankroll_fraction,
        sort_col=args.sort_col,
    )

    summary = summarize_backtest(backtest_df).as_dict()
    curve = profit_curve(backtest_df)
    group_summary = None
    if args.group_col:
        group_summary = summarize_by_group(backtest_df, args.group_col)

    backtest_path = args.output_dir / "backtest_bets.csv"
    summary_path = args.output_dir / "backtest_summary.json"
    curve_path = args.output_dir / "profit_curve.csv"
    report_path = args.output_dir / "backtest_report.md"

    backtest_df.to_csv(backtest_path, index=False)
    curve.to_csv(curve_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown_report(report_path, summary, group_summary)

    if group_summary is not None:
        group_summary.to_csv(args.output_dir / "backtest_group_summary.csv", index=False)

    log.info("Backtest complete.")
    log.info("  Bets placed : %s", summary["bets"])
    log.info("  Profit      : %s", summary["profit"])
    log.info("  ROI         : %s", summary["roi"])
    log.info("  Outputs     : %s", args.output_dir)


if __name__ == "__main__":
    main()
