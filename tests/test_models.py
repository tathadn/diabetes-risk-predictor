"""
Unit tests for src/models.py.

All tests use small synthetic datasets generated with sklearn's
make_classification — no real data needed.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (
    ModelTrainer,
    cross_validate_model,
    get_lightgbm,
    get_logistic_regression,
    get_random_forest,
    get_xgboost,
    load_model,
    save_model,
    train_model,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def binary_data():
    """Small binary classification dataset."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42,
    )
    return X, y


@pytest.fixture(scope="module")
def multiclass_data():
    """Small 3-class classification dataset matching diabetes target schema."""
    X, y = make_classification(
        n_samples=600,
        n_features=10,
        n_informative=6,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    return X, y


# ---------------------------------------------------------------------------
# Model factory tests
# ---------------------------------------------------------------------------


class TestGetLogisticRegression:
    def test_returns_correct_type(self):
        model = get_logistic_regression()
        assert isinstance(model, LogisticRegression)

    def test_default_random_state(self):
        model = get_logistic_regression()
        assert model.random_state == 42

    def test_custom_params_applied(self):
        model = get_logistic_regression({"C": 0.01, "max_iter": 200})
        assert model.C == pytest.approx(0.01)
        assert model.max_iter == 200

    def test_unfitted_by_default(self):
        model = get_logistic_regression()
        assert not hasattr(model, "coef_")


class TestGetRandomForest:
    def test_returns_correct_type(self):
        model = get_random_forest()
        assert isinstance(model, RandomForestClassifier)

    def test_default_random_state(self):
        model = get_random_forest()
        assert model.random_state == 42

    def test_custom_n_estimators(self):
        model = get_random_forest({"n_estimators": 50})
        assert model.n_estimators == 50


class TestGetXGBoost:
    def test_returns_correct_type(self):
        model = get_xgboost()
        assert isinstance(model, XGBClassifier)

    def test_multiclass_objective(self):
        model = get_xgboost()
        assert model.objective == "multi:softprob"

    def test_num_class_is_3(self):
        model = get_xgboost()
        assert model.num_class == 3


class TestGetLightGBM:
    def test_returns_correct_type(self):
        model = get_lightgbm()
        assert isinstance(model, LGBMClassifier)

    def test_multiclass_objective(self):
        model = get_lightgbm()
        assert model.objective == "multiclass"

    def test_num_class_is_3(self):
        model = get_lightgbm()
        assert model.num_class == 3


# ---------------------------------------------------------------------------
# train_model tests
# ---------------------------------------------------------------------------


class TestTrainModel:
    def test_returns_fitted_estimator(self, binary_data):
        X, y = binary_data
        model = get_logistic_regression()
        fitted = train_model(model, X, y, model_name="lr_test")
        # After fitting, coef_ should exist
        assert hasattr(fitted, "coef_")

    def test_returns_same_object(self, binary_data):
        X, y = binary_data
        model = get_logistic_regression()
        fitted = train_model(model, X, y)
        assert fitted is model

    def test_can_predict_after_training(self, binary_data):
        X, y = binary_data
        model = get_random_forest({"n_estimators": 10})
        fitted = train_model(model, X, y)
        preds = fitted.predict(X)
        assert preds.shape == (len(y),)

    def test_works_with_multiclass(self, multiclass_data):
        X, y = multiclass_data
        # Use a simple LR to avoid XGB num_class mismatch for this fixture
        model = get_logistic_regression()
        fitted = train_model(model, X, y, model_name="lr_multiclass")
        assert hasattr(fitted, "coef_")


# ---------------------------------------------------------------------------
# cross_validate_model tests
# ---------------------------------------------------------------------------


class TestCrossValidateModel:
    def test_returns_expected_keys(self, binary_data):
        X, y = binary_data
        model = get_logistic_regression()
        result = cross_validate_model(model, X, y, cv=3, scoring="accuracy")
        for key in ("mean_score", "std_score", "fold_scores", "mean_fit_time", "scoring"):
            assert key in result, f"Missing key: {key}"

    def test_fold_scores_length_matches_cv(self, binary_data):
        X, y = binary_data
        model = get_logistic_regression()
        result = cross_validate_model(model, X, y, cv=3, scoring="accuracy")
        assert len(result["fold_scores"]) == 3

    def test_mean_score_in_valid_range(self, binary_data):
        X, y = binary_data
        model = get_logistic_regression()
        result = cross_validate_model(model, X, y, cv=3, scoring="accuracy")
        assert 0.0 <= result["mean_score"] <= 1.0

    def test_scoring_field_matches_input(self, binary_data):
        X, y = binary_data
        model = get_logistic_regression()
        result = cross_validate_model(model, X, y, cv=3, scoring="accuracy")
        assert result["scoring"] == "accuracy"


# ---------------------------------------------------------------------------
# save_model / load_model tests
# ---------------------------------------------------------------------------


class TestSaveLoadModel:
    def test_roundtrip_preserves_predictions(self, binary_data, tmp_path):
        X, y = binary_data
        model = get_logistic_regression()
        model.fit(X, y)
        original_preds = model.predict(X)

        save_path = str(tmp_path / "lr.pkl")
        save_model(model, save_path)
        loaded = load_model(save_path)

        loaded_preds = loaded.predict(X)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_save_creates_file(self, binary_data, tmp_path):
        X, y = binary_data
        model = get_logistic_regression()
        model.fit(X, y)
        save_path = str(tmp_path / "subdir" / "model.pkl")
        save_model(model, save_path)
        assert Path(save_path).exists()

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model("/tmp/this_file_does_not_exist_xyz.pkl")

    def test_random_forest_roundtrip(self, binary_data, tmp_path):
        X, y = binary_data
        model = get_random_forest({"n_estimators": 10})
        model.fit(X, y)
        save_path = str(tmp_path / "rf.pkl")
        save_model(model, save_path)
        loaded = load_model(save_path)
        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


# ---------------------------------------------------------------------------
# ModelTrainer tests
# ---------------------------------------------------------------------------


class TestModelTrainer:
    @pytest.fixture
    def trainer(self):
        return ModelTrainer()

    def test_add_model_registers_name(self, trainer):
        trainer.add_model("lr", get_logistic_regression())
        assert "lr" in trainer.list_models()

    def test_get_model_raises_for_unknown(self, trainer):
        with pytest.raises(KeyError):
            trainer.get_model("nonexistent")

    def test_train_fits_model(self, trainer, binary_data):
        X, y = binary_data
        trainer.add_model("lr", get_logistic_regression())
        fitted = trainer.train("lr", X, y)
        assert hasattr(fitted, "coef_")

    def test_train_all_trains_all_models(self, trainer, binary_data):
        X, y = binary_data
        trainer.add_model("lr", get_logistic_regression())
        trainer.add_model("rf", get_random_forest({"n_estimators": 10}))
        trainer.train_all(X, y)
        for name in ["lr", "rf"]:
            assert trainer._trained[name] is True

    def test_record_and_get_comparison(self, trainer, binary_data):
        X, y = binary_data
        trainer.add_model("lr", get_logistic_regression())
        trainer.train("lr", X, y)
        trainer.record_result("lr", {"accuracy": 0.85, "roc_auc": 0.90, "f1_weighted": 0.84})
        df = trainer.get_comparison()
        assert "lr" in df.index
        assert df.loc["lr", "roc_auc"] == pytest.approx(0.90)

    def test_get_comparison_empty_before_results(self, trainer):
        import pandas as pd
        df = trainer.get_comparison()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_save_all_creates_files(self, trainer, binary_data, tmp_path):
        X, y = binary_data
        trainer.add_model("lr", get_logistic_regression())
        trainer.train("lr", X, y)
        trainer.save_all(str(tmp_path))
        assert (tmp_path / "lr.pkl").exists()

    def test_list_models(self, trainer):
        trainer.add_model("a", get_logistic_regression())
        trainer.add_model("b", get_random_forest({"n_estimators": 5}))
        assert set(trainer.list_models()) == {"a", "b"}

    def test_repr(self, trainer):
        trainer.add_model("lr", get_logistic_regression())
        r = repr(trainer)
        assert "ModelTrainer" in r
        assert "lr" in r
