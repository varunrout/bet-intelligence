"""
Walk-forward cross-validation for Football Market Intelligence System — Phase 4.

Implements an expanding-window walk-forward CV protocol over EPL seasons.

Fold structure (4 folds, 5 seasons 2019/20 – 2023/24)
------------------------------------------------------
  Fold 1: train 2019/20                          → test 2020/21
  Fold 2: train 2019/20, 2020/21                 → test 2021/22
  Fold 3: train 2019/20, 2020/21, 2021/22        → test 2022/23
  Fold 4: train 2019/20, 2020/21, 2021/22, 2022/23 → test 2023/24

Design rules
------------
- Expanding window: every fold adds one more season to training.
  No rolling window is used — more data is always better for the prior.
- Chronological ordering of df is MANDATORY before calling walk_forward_cv().
- Random or shuffled splits are strictly forbidden for financial time-series.
- The df, X, and y arguments must share the same pandas index (subset allowed).

Public API
----------
  walk_forward_cv(df, X, y, model_fn, label)   -> list[dict]
  aggregate_cv_results(fold_results)            -> dict
  results_to_dataframe(fold_results)            -> pd.DataFrame
  predictions_dataframe(fold_results)           -> pd.DataFrame
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.modeling.evaluate import compute_metrics


# ── Walk-forward CV ────────────────────────────────────────────────────────────


def walk_forward_cv(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    model_fn: Callable,
    label: str = "model",
) -> list[dict]:
    """
    Expanding-window walk-forward cross-validation over EPL seasons.

    Parameters
    ----------
    df       : Master DataFrame.  Must contain a 'season' column.
               Must be index-aligned with X (X.index must be a subset of
               df.index).  Ordering by kickoff_utc is assumed.
    X        : Feature matrix (pd.DataFrame).  Index is a subset of df.index
               after NaN rows have been dropped by prepare().
    y        : Binary target Series (pd.Series).  Must be index-aligned with X.
               Typically obtained via df.loc[X.index, "over_25"].
    model_fn : Zero-argument callable returning a fresh, unfitted model that
               exposes sklearn-compatible .fit(X, y) and
               .predict_proba(X) -> array[:, 2] methods.
    label    : Human-readable name for the model (stored in each fold result).

    Returns
    -------
    List of fold-result dicts.  Each dict contains:
        test_season  : str   — the season used as the test set
        n_train      : int   — number of training rows
        n_test       : int   — number of test rows
        metrics      : dict  — from compute_metrics() for this fold
        y_true       : np.ndarray — ground-truth labels for test rows
        y_prob       : np.ndarray — predicted probabilities for test rows
        test_index   : pd.Index  — original df/X index for the test rows
        model        : fitted model instance
        label        : str   — same as the label argument

    Notes
    -----
    Seasons are sorted lexicographically, which matches the EPL season ID
    format 'YYYY/YY' (e.g. '2019/20' < '2020/21').  The first season is
    always used only as training; the loop starts at i=1.
    """
    season_col = df.loc[X.index, "season"]
    seasons    = sorted(season_col.unique())

    if len(seasons) < 2:
        raise ValueError(
            f"walk_forward_cv requires at least 2 seasons; found {seasons}."
        )

    fold_results: list[dict] = []

    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        test_season   = seasons[i]

        train_mask = season_col.isin(train_seasons)
        test_mask  = season_col == test_season

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_test  = X.loc[test_mask]
        y_test  = y.loc[test_mask]

        if len(X_test) == 0:
            continue

        model = model_fn()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics               = compute_metrics(y_test.values, y_prob, label=label)
        metrics["test_season"] = test_season

        fold_results.append(
            {
                "test_season": test_season,
                "n_train":     int(len(X_train)),
                "n_test":      int(len(X_test)),
                "metrics":     metrics,
                "y_true":      y_test.values.copy(),
                "y_prob":      y_prob.copy(),
                "test_index":  X_test.index,
                "model":       model,
                "label":       label,
            }
        )

    return fold_results


# ── Aggregation helpers ────────────────────────────────────────────────────────


def aggregate_cv_results(fold_results: list[dict]) -> dict:
    """
    Average metrics across all folds from walk_forward_cv().

    Parameters
    ----------
    fold_results : list returned by walk_forward_cv().

    Returns
    -------
    Dict with mean and std of each scalar metric across folds, plus metadata:
        label, n_folds, n_train_mean, n_test_total,
        roc_auc_mean, roc_auc_std,
        brier_mean,   brier_std,
        bss_mean,     bss_std,
        log_loss_mean, log_loss_std
    """
    if not fold_results:
        return {}

    metric_keys = ["roc_auc", "brier", "bss", "log_loss"]
    agg: dict = {
        "label":        fold_results[0]["label"],
        "n_folds":      len(fold_results),
        "n_train_mean": int(np.mean([r["n_train"] for r in fold_results])),
        "n_test_total": int(np.sum( [r["n_test"]  for r in fold_results])),
    }

    for key in metric_keys:
        vals = [r["metrics"][key] for r in fold_results if key in r["metrics"]]
        agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
        agg[f"{key}_std"]  = round(float(np.std(vals)),  4)

    return agg


def results_to_dataframe(fold_results: list[dict]) -> pd.DataFrame:
    """
    Flatten per-fold results into a single DataFrame for display or charting.

    Parameters
    ----------
    fold_results : list returned by walk_forward_cv().

    Returns
    -------
    pd.DataFrame with columns:
        model, test_season, n_train, n_test,
        roc_auc, brier, bss, log_loss, base_rate
    """
    rows = []
    for r in fold_results:
        m = r["metrics"]
        rows.append(
            {
                "model":       r["label"],
                "test_season": r["test_season"],
                "n_train":     r["n_train"],
                "n_test":      r["n_test"],
                "roc_auc":     m.get("roc_auc"),
                "brier":       m.get("brier"),
                "bss":         m.get("bss"),
                "log_loss":    m.get("log_loss"),
                "base_rate":   m.get("base_rate"),
            }
        )
    return pd.DataFrame(rows)


def predictions_dataframe(fold_results: list[dict]) -> pd.DataFrame:
    """
    Pool all out-of-fold predictions into a single DataFrame.

    Use this to compute overall calibration or pooled metrics across all folds
    for a single model.

    Parameters
    ----------
    fold_results : list returned by walk_forward_cv().

    Returns
    -------
    pd.DataFrame indexed by the original match index, with columns:
        test_season, y_true, y_prob, model
    """
    frames = []
    for r in fold_results:
        frame = pd.DataFrame(
            {
                "test_season": r["test_season"],
                "y_true":      r["y_true"],
                "y_prob":      r["y_prob"],
                "model":       r["label"],
            },
            index=r["test_index"],
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["test_season", "y_true", "y_prob", "model"])

    return pd.concat(frames, ignore_index=False)
