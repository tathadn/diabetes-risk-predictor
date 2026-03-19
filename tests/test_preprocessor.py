"""
Unit tests for src/preprocessor.py.

All tests use synthetic DataFrames — no real dataset is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on the path so `src` can be imported.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessor import (
    PreprocessingPipeline,
    cap_outliers_iqr,
    detect_outliers_iqr,
    handle_missing_values,
    remove_outliers_iqr,
    scale_features,
    train_test_split_stratified,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Small numeric DataFrame with no missing values or outliers."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "A": rng.normal(50, 10, 100).astype(float),
            "B": rng.normal(20, 5, 100).astype(float),
            "C": rng.integers(0, 5, 100).astype(float),
        }
    )


@pytest.fixture
def df_with_nans() -> pd.DataFrame:
    """DataFrame where every column contains some NaN values."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "num1": rng.normal(10, 2, 50).astype(float),
            "num2": rng.normal(5, 1, 50).astype(float),
            "cat1": ["a", "b", "c"] * 16 + ["a", "b"],
        }
    )
    # Inject NaNs
    df.loc[[0, 5, 10], "num1"] = np.nan
    df.loc[[2, 7], "num2"] = np.nan
    df.loc[[3, 9], "cat1"] = np.nan
    return df


@pytest.fixture
def df_with_outliers() -> pd.DataFrame:
    """DataFrame with a few obvious outliers injected."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "X": rng.normal(0, 1, 200).astype(float),
            "Y": rng.normal(100, 5, 200).astype(float),
        }
    )
    # Inject clear outliers
    df.loc[0, "X"] = 1000.0
    df.loc[1, "X"] = -1000.0
    df.loc[2, "Y"] = 9999.0
    return df


@pytest.fixture
def classification_data():
    """Return (X, y) for a balanced 3-class synthetic problem."""
    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.tile([0, 1, 2], n // 3), name="target")
    return X, y


# ---------------------------------------------------------------------------
# Tests: handle_missing_values
# ---------------------------------------------------------------------------


class TestHandleMissingValues:
    def test_mean_strategy_removes_nans(self, df_with_nans):
        df_out, report = handle_missing_values(df_with_nans, strategy="mean")
        assert df_out["num1"].isnull().sum() == 0
        assert df_out["num2"].isnull().sum() == 0

    def test_median_strategy(self, df_with_nans):
        df_out, report = handle_missing_values(df_with_nans, strategy="median")
        assert df_out["num1"].isnull().sum() == 0

    def test_mode_strategy(self, df_with_nans):
        df_out, report = handle_missing_values(df_with_nans, strategy="mode")
        assert not df_out.isnull().any().any()

    def test_categorical_always_mode(self, df_with_nans):
        df_out, report = handle_missing_values(df_with_nans, strategy="mean")
        assert df_out["cat1"].isnull().sum() == 0

    def test_report_contains_imputed_columns(self, df_with_nans):
        _, report = handle_missing_values(df_with_nans, strategy="mean")
        assert "num1" in report
        assert "num2" in report

    def test_invalid_strategy_raises(self, simple_df):
        with pytest.raises(ValueError, match="Unknown strategy"):
            handle_missing_values(simple_df, strategy="invalid")

    def test_no_nans_returns_empty_report(self, simple_df):
        df_out, report = handle_missing_values(simple_df, strategy="mean")
        assert len(report) == 0

    def test_original_df_not_mutated(self, df_with_nans):
        original_nans = df_with_nans["num1"].isnull().sum()
        handle_missing_values(df_with_nans, strategy="mean")
        assert df_with_nans["num1"].isnull().sum() == original_nans


# ---------------------------------------------------------------------------
# Tests: detect_outliers_iqr
# ---------------------------------------------------------------------------


class TestDetectOutliersIQR:
    def test_returns_boolean_dataframe(self, df_with_outliers):
        mask = detect_outliers_iqr(df_with_outliers, ["X", "Y"])
        assert isinstance(mask, pd.DataFrame)
        assert mask.dtypes.eq(bool).all()

    def test_detects_known_outliers(self, df_with_outliers):
        mask = detect_outliers_iqr(df_with_outliers, ["X", "Y"])
        assert mask.loc[0, "X"]   # 1000.0 is an outlier
        assert mask.loc[1, "X"]   # -1000.0 is an outlier
        assert mask.loc[2, "Y"]   # 9999.0 is an outlier

    def test_no_false_positives_for_normal_data(self, simple_df):
        # With threshold=3, very few (if any) normal samples should be flagged
        mask = detect_outliers_iqr(simple_df, ["A", "B"], threshold=3.0)
        assert mask.any(axis=1).sum() < 10  # at most a handful

    def test_mask_shape_matches_columns(self, df_with_outliers):
        mask = detect_outliers_iqr(df_with_outliers, ["X"])
        assert mask.shape == (len(df_with_outliers), 1)
        assert list(mask.columns) == ["X"]


# ---------------------------------------------------------------------------
# Tests: remove_outliers_iqr
# ---------------------------------------------------------------------------


class TestRemoveOutliersIQR:
    def test_outlier_rows_removed(self, df_with_outliers):
        df_clean = remove_outliers_iqr(df_with_outliers, ["X", "Y"])
        assert len(df_clean) < len(df_with_outliers)

    def test_known_outlier_indices_gone(self, df_with_outliers):
        df_clean = remove_outliers_iqr(df_with_outliers, ["X"])
        # Row 0 had X=1000, row 1 had X=-1000 — both should be absent
        assert 1000.0 not in df_clean["X"].values
        assert -1000.0 not in df_clean["X"].values

    def test_index_reset(self, df_with_outliers):
        df_clean = remove_outliers_iqr(df_with_outliers, ["X"])
        assert df_clean.index[0] == 0
        assert list(df_clean.index) == list(range(len(df_clean)))

    def test_no_data_loss_when_no_outliers(self, simple_df):
        # Threshold very high — nothing should be removed
        df_clean = remove_outliers_iqr(simple_df, ["A"], threshold=10.0)
        assert len(df_clean) == len(simple_df)


# ---------------------------------------------------------------------------
# Tests: scale_features
# ---------------------------------------------------------------------------


class TestScaleFeatures:
    @pytest.fixture
    def train_test(self, simple_df):
        split_idx = 80
        X_train = simple_df.iloc[:split_idx]
        X_test = simple_df.iloc[split_idx:]
        return X_train, X_test

    def test_output_shape_preserved(self, train_test):
        X_train, X_test = train_test
        X_tr_scaled, X_te_scaled, _ = scale_features(X_train, X_test, method="standard")
        assert X_tr_scaled.shape == X_train.shape
        assert X_te_scaled.shape == X_test.shape

    def test_standard_scaler_mean_near_zero(self, train_test):
        X_train, X_test = train_test
        X_tr_scaled, _, _ = scale_features(X_train, X_test, method="standard")
        assert np.abs(X_tr_scaled.mean(axis=0)).max() < 1e-10

    def test_standard_scaler_std_near_one(self, train_test):
        X_train, X_test = train_test
        X_tr_scaled, _, _ = scale_features(X_train, X_test, method="standard")
        assert np.abs(X_tr_scaled.std(axis=0) - 1).max() < 1e-6

    def test_minmax_range(self, train_test):
        X_train, X_test = train_test
        X_tr_scaled, _, _ = scale_features(X_train, X_test, method="minmax")
        assert X_tr_scaled.min() >= 0.0 - 1e-10
        assert X_tr_scaled.max() <= 1.0 + 1e-10

    def test_scaler_object_returned(self, train_test):
        X_train, X_test = train_test
        _, _, scaler = scale_features(X_train, X_test, method="standard")
        assert hasattr(scaler, "transform")

    def test_invalid_method_raises(self, train_test):
        X_train, X_test = train_test
        with pytest.raises(ValueError, match="Unknown scaling method"):
            scale_features(X_train, X_test, method="z-score")


# ---------------------------------------------------------------------------
# Tests: train_test_split_stratified
# ---------------------------------------------------------------------------


class TestTrainTestSplitStratified:
    def test_sizes(self, classification_data):
        X, y = classification_data
        X_tr, X_te, y_tr, y_te = train_test_split_stratified(X, y, test_size=0.2)
        total = len(X)
        assert len(X_te) == pytest.approx(total * 0.2, abs=2)
        assert len(X_tr) + len(X_te) == total

    def test_stratification(self, classification_data):
        X, y = classification_data
        _, _, y_tr, y_te = train_test_split_stratified(X, y, test_size=0.2)
        # Each class should appear in both splits
        for cls in y.unique():
            assert cls in y_tr.values
            assert cls in y_te.values

    def test_proportions_roughly_equal(self, classification_data):
        X, y = classification_data
        _, _, y_tr, y_te = train_test_split_stratified(X, y, test_size=0.2)
        train_dist = y_tr.value_counts(normalize=True).sort_index()
        test_dist = y_te.value_counts(normalize=True).sort_index()
        np.testing.assert_allclose(train_dist.values, test_dist.values, atol=0.05)

    def test_reproducibility(self, classification_data):
        X, y = classification_data
        X_tr1, X_te1, _, _ = train_test_split_stratified(X, y, random_state=0)
        X_tr2, X_te2, _, _ = train_test_split_stratified(X, y, random_state=0)
        pd.testing.assert_frame_equal(X_tr1, X_tr2)


# ---------------------------------------------------------------------------
# Tests: PreprocessingPipeline
# ---------------------------------------------------------------------------


class TestPreprocessingPipeline:
    @pytest.fixture
    def pipeline(self):
        return PreprocessingPipeline(
            missing_strategy="mean",
            scale_method="standard",
            outlier_threshold=1.5,
        )

    @pytest.fixture
    def feature_df(self, df_with_nans):
        # Drop categorical column so we get a fully numeric DataFrame
        return df_with_nans[["num1", "num2"]].copy()

    def test_fit_transform_returns_array(self, pipeline, feature_df):
        result = pipeline.fit_transform(feature_df)
        assert isinstance(result, np.ndarray)

    def test_fit_transform_shape(self, pipeline, feature_df):
        result = pipeline.fit_transform(feature_df)
        assert result.shape[0] == len(feature_df)
        assert result.shape[1] == 2

    def test_is_fitted_after_fit_transform(self, pipeline, feature_df):
        pipeline.fit_transform(feature_df)
        assert pipeline._fitted is True

    def test_transform_without_fit_raises(self, feature_df):
        new_pipeline = PreprocessingPipeline()
        with pytest.raises(RuntimeError, match="fit_transform"):
            new_pipeline.transform(feature_df)

    def test_transform_consistent_with_fit_transform(self, pipeline, feature_df):
        # Same data should give same result
        X_fit = pipeline.fit_transform(feature_df)
        X_transform = pipeline.transform(feature_df)
        np.testing.assert_allclose(X_fit, X_transform, atol=1e-6)

    def test_save_and_load_roundtrip(self, pipeline, feature_df, tmp_path):
        pipeline.fit_transform(feature_df)
        save_path = str(tmp_path / "pipeline.pkl")
        pipeline.save(save_path)
        loaded = PreprocessingPipeline.load(save_path)
        assert loaded._fitted is True
        X_original = pipeline.transform(feature_df)
        X_loaded = loaded.transform(feature_df)
        np.testing.assert_allclose(X_original, X_loaded, atol=1e-8)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            PreprocessingPipeline.load("/tmp/nonexistent_pipeline.pkl")

    def test_repr_shows_fitted_status(self, pipeline, feature_df):
        assert "fitted=False" in repr(pipeline)
        pipeline.fit_transform(feature_df)
        assert "fitted=True" in repr(pipeline)
