"""CLI for walk-forward model training and prediction export."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.modeling.features import prepare
from src.modeling.model_registry import available_model_names, get_model_spec
from src.modeling.walk_forward import predictions_dataframe, walk_forward_cv
from src.utils.config import DB_PATH
from src.utils.db import run_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_modeling_frame(feature_version: str = "v1") -> pd.DataFrame:
    """Load target, metadata and engineered features from DuckDB."""
    sql = """
        SELECT
            m.match_id,
            m.season,
            m.kickoff_utc,
            m.over_25,
            ef.*
        FROM engineered_features ef
        JOIN matches m ON ef.match_id = m.match_id
        WHERE ef.feature_version = ?
        ORDER BY m.kickoff_utc
    """
    df = run_query(sql, params=[feature_version], db_path=DB_PATH)
    if df.empty:
        raise RuntimeError(
            "No engineered features found. Run scripts/build_features.py first."
        )

    duplicate_cols = [c for c in df.columns if c.endswith("_1")]
    if duplicate_cols:
        df = df.drop(columns=duplicate_cols)

    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    return df.reset_index(drop=True)


def export_predictions(df: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    """Run walk-forward CV for requested models and return OOF predictions."""
    frames: list[pd.DataFrame] = []
    metadata_cols = [
        "match_id",
        "season",
        "kickoff_utc",
        "over_25",
        "pin_odds_over",
        "pin_implied_prob_over",
        "b365_odds_over",
        "b365_implied_prob_over",
    ]
    metadata_cols = [c for c in metadata_cols if c in df.columns]

    for model_name in model_names:
        spec = get_model_spec(model_name)
        log.info("Training %s with feature_set=%s", spec.name, spec.feature_set)
        X, y = prepare(df, feature_set=spec.feature_set, drop_nan_rows=True)
        folds = walk_forward_cv(df, X, y, model_fn=spec.factory, label=spec.name)
        preds = predictions_dataframe(folds)

        if preds.empty:
            log.warning("No predictions produced for model %s", model_name)
            continue

        meta = df.loc[preds.index, metadata_cols].copy()
        preds = preds.join(meta)
        preds["feature_set"] = spec.feature_set
        frames.append(preds.reset_index(drop=True))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train registered models using walk-forward CV.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["market_logistic", "form_logistic", "all_logistic"],
        help=f"Model names. Available: {available_model_names()}",
    )
    parser.add_argument("--feature-version", default="v1")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/predictions/oof_predictions.csv"),
    )
    args = parser.parse_args()

    unknown = sorted(set(args.models) - set(available_model_names()))
    if unknown:
        log.error("Unknown model(s): %s", unknown)
        log.error("Available models: %s", available_model_names())
        sys.exit(1)

    df = load_modeling_frame(feature_version=args.feature_version)
    log.info("Loaded %d rows and %d columns", len(df), len(df.columns))

    predictions = export_predictions(df, model_names=args.models)
    if predictions.empty:
        log.error("No predictions were produced.")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    log.info("Saved %d prediction rows to %s", len(predictions), args.output)


if __name__ == "__main__":
    main()
