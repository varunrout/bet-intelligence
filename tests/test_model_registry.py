from __future__ import annotations

import pytest
from sklearn.pipeline import Pipeline

from src.modeling.model_registry import (
    available_model_names,
    get_model_spec,
    list_model_specs,
    logistic_regression_factory,
)


def test_available_model_names_contains_baselines():
    names = available_model_names()
    assert "market_logistic" in names
    assert "form_logistic" in names
    assert "all_logistic" in names


def test_get_model_spec_returns_feature_set():
    spec = get_model_spec("market_logistic")
    assert spec.name == "market_logistic"
    assert spec.feature_set == "market_only"
    assert callable(spec.factory)


def test_get_model_spec_rejects_unknown_model():
    with pytest.raises(ValueError):
        get_model_spec("not_a_model")


def test_list_model_specs_sorted_by_name():
    specs = list_model_specs()
    names = [spec.name for spec in specs]
    assert names == sorted(names)


def test_logistic_factory_returns_pipeline():
    model = logistic_regression_factory()
    assert isinstance(model, Pipeline)
    assert "imputer" in model.named_steps
    assert "scaler" in model.named_steps
    assert "model" in model.named_steps
