"""
Model evaluation utilities for Football Market Intelligence System — Phase 4.

All metrics use a probabilistic evaluation framework appropriate for binary
classification with well-calibrated probability outputs:

  ROC-AUC   — rank-order discrimination ability (threshold-independent).
  Brier     — mean squared error of probability forecasts (lower = better).
  BSS       — Brier Skill Score: improvement over the naive constant-rate
              baseline (higher = better; 0 = no skill; 1 = perfect).
  Log-Loss  — cross-entropy penalising confident wrong predictions.

Public API
----------
  compute_metrics(y_true, y_prob, label)  -> dict
  calibration_bins(y_true, y_prob, n_bins) -> pd.DataFrame
  metrics_table(metrics_list)             -> pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


# ── Core metric computation ────────────────────────────────────────────────────


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    label: str = "model",
) -> dict:
    """
    Compute probabilistic evaluation metrics for a binary classifier.

    Parameters
    ----------
    y_true : array-like of {0, 1} ground-truth binary labels.
    y_prob : array-like of predicted probabilities for the positive class.
    label  : human-readable identifier for the model / fold.

    Returns
    -------
    dict with keys:
        label, n, base_rate, roc_auc, brier, bss, log_loss

    Notes
    -----
    BSS = 1 - Brier / Brier_naive, where Brier_naive is the Brier score of
    always predicting the empirical base rate in y_true.  A model with
    BSS = 0 is no better than a constant prediction; BSS < 0 is worse.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Clip to a safe range to prevent log(0)
    y_prob_safe = np.clip(y_prob, 1e-7, 1.0 - 1e-7)

    base_rate  = float(y_true.mean())
    brier      = float(np.mean((y_prob - y_true) ** 2))
    brier_naive = float(np.mean((base_rate - y_true) ** 2))
    bss        = float(1.0 - brier / brier_naive) if brier_naive > 1e-12 else 0.0

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        # Raised when y_true has only one class
        auc = float("nan")

    ll = float(log_loss(y_true, y_prob_safe))

    return {
        "label":     label,
        "n":         int(len(y_true)),
        "base_rate": round(base_rate, 4),
        "roc_auc":   round(auc,       4),
        "brier":     round(brier,     4),
        "bss":       round(bss,       4),
        "log_loss":  round(ll,        4),
    }


# ── Calibration analysis ───────────────────────────────────────────────────────


def calibration_bins(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Group predictions into equal-width probability bins and compute the mean
    predicted probability vs actual positive rate in each bin.

    A perfectly calibrated model gives mean_predicted ≈ actual_rate in every
    bin, lying on the diagonal of a reliability diagram.

    Parameters
    ----------
    y_true : ground-truth binary labels.
    y_prob : predicted probabilities for the positive class.
    n_bins : number of equal-width bins spanning [0, 1].

    Returns
    -------
    pd.DataFrame with columns:
        bin_lower, bin_upper, bin_center, n,
        mean_predicted, actual_rate, calibration_error

    Empty bins (no predictions in range) are omitted.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict] = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Last bin is closed on both ends
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        n = int(mask.sum())
        if n == 0:
            continue

        rows.append(
            {
                "bin_lower":      round(float(lo),                          3),
                "bin_upper":      round(float(hi),                          3),
                "bin_center":     round(float((lo + hi) / 2.0),             3),
                "n":              n,
                "mean_predicted": round(float(y_prob[mask].mean()),          4),
                "actual_rate":    round(float(y_true[mask].mean()),          4),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["calibration_error"] = (df["mean_predicted"] - df["actual_rate"]).round(4)
    return df


# ── Metrics table formatter ────────────────────────────────────────────────────


def metrics_table(metrics_list: list[dict]) -> pd.DataFrame:
    """
    Convert a list of compute_metrics() dicts into a formatted display DataFrame.

    Parameters
    ----------
    metrics_list : list of dicts, each the return value of compute_metrics().

    Returns
    -------
    pd.DataFrame with columns ordered for display:
        label, n, base_rate, roc_auc, brier, bss, log_loss
    """
    if not metrics_list:
        return pd.DataFrame(
            columns=["label", "n", "base_rate", "roc_auc", "brier", "bss", "log_loss"]
        )

    df = pd.DataFrame(metrics_list)
    ordered_cols = [
        c for c in ["label", "n", "base_rate", "roc_auc", "brier", "bss", "log_loss"]
        if c in df.columns
    ]
    return df[ordered_cols].reset_index(drop=True)
