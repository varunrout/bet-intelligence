"""
Model registry for walk-forward experiments.

The registry keeps model definitions in code rather than notebooks. Each model
spec defines the feature set it expects and a factory that returns a fresh,
unfitted sklearn-compatible estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    """Definition for a model that can be evaluated walk-forward."""

    name: str
    feature_set: str
    factory: Callable[[], object]
    description: str


def logistic_regression_factory() -> Pipeline:
    """Return a robust logistic regression pipeline for probability baselines."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def regularized_lgbm_factory():
    """Return a conservative LightGBM classifier for small tabular datasets."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "LightGBM is required for the 'all_lgbm_regularized' model. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    return LGBMClassifier(
        objective="binary",
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.3,
        random_state=42,
        verbose=-1,
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    "market_logistic": ModelSpec(
        name="market_logistic",
        feature_set="market_only",
        factory=logistic_regression_factory,
        description="Pinnacle market-implied probability logistic baseline.",
    ),
    "form_logistic": ModelSpec(
        name="form_logistic",
        feature_set="form_only",
        factory=logistic_regression_factory,
        description="Rolling form, xG, context and Elo logistic model without market prices.",
    ),
    "all_logistic": ModelSpec(
        name="all_logistic",
        feature_set="all",
        factory=logistic_regression_factory,
        description="Logistic model using market, form, xG, context and Elo features.",
    ),
    "all_lgbm_regularized": ModelSpec(
        name="all_lgbm_regularized",
        feature_set="all",
        factory=regularized_lgbm_factory,
        description="Conservative LightGBM model for the full feature set.",
    ),
}


def available_model_names() -> list[str]:
    """Return all registered model names."""
    return sorted(MODEL_SPECS.keys())


def get_model_spec(name: str) -> ModelSpec:
    """Return a registered model spec by name."""
    try:
        return MODEL_SPECS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{name}'. Available models: {available_model_names()}"
        ) from exc


def list_model_specs() -> list[ModelSpec]:
    """Return registered model specs sorted by name."""
    return [MODEL_SPECS[name] for name in available_model_names()]
