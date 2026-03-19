"""
Preprocessing pipeline for the Diabetes Prediction project.

Provides standalone helper functions and a ``PreprocessingPipeline`` class
that chains all preprocessing steps into a single fit/transform interface.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from src.config import RANDOM_STATE, TEST_SIZE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------------------------

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "mean",
) -> Tuple[pd.DataFrame, dict]:
    """Fill missing values in numeric and categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (may contain NaN values).
    strategy : str, optional
        Imputation strategy for numeric columns.
        One of ``'mean'``, ``'median'``, or ``'mode'``.
        Categorical columns are always filled with the mode.
        Defaults to ``'mean'``.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        - Cleaned DataFrame with no missing values in treated columns.
        - Report dict: keys are column names, values are fill values used.

    Raises
    ------
    ValueError
        If *strategy* is not one of the supported options.
    """
    valid_strategies = {"mean", "median", "mode"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from {valid_strategies}."
        )

    df_out = df.copy()
    fill_report: dict = {}

    numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_out.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in numeric_cols:
        if df_out[col].isnull().any():
            if strategy == "mean":
                fill_val = df_out[col].mean()
            elif strategy == "median":
                fill_val = df_out[col].median()
            else:  # mode
                fill_val = df_out[col].mode()[0]
            df_out[col] = df_out[col].fillna(fill_val)
            fill_report[col] = fill_val

    for col in categorical_cols:
        if df_out[col].isnull().any():
            fill_val = df_out[col].mode()[0]
            df_out[col] = df_out[col].fillna(fill_val)
            fill_report[col] = fill_val

    logger.info(
        "Missing values handled (%s strategy) — %d columns imputed.",
        strategy,
        len(fill_report),
    )
    return df_out, fill_report


# ---------------------------------------------------------------------------
# Outlier detection and treatment
# ---------------------------------------------------------------------------

def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: List[str],
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Return a boolean mask DataFrame where True indicates an outlier.

    Uses the IQR method: a value is an outlier if it lies more than
    ``threshold * IQR`` below Q1 or above Q3.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : List[str]
        Numeric columns to check.
    threshold : float, optional
        IQR multiplier. Defaults to ``1.5``.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame with the same shape as ``df[columns]``.
        ``True`` means the value is an outlier.
    """
    mask = pd.DataFrame(False, index=df.index, columns=columns)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask[col] = (df[col] < lower) | (df[col] > upper)
    return mask


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: List[str],
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Remove rows that contain IQR outliers in any of *columns*.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : List[str]
        Columns to check for outliers.
    threshold : float, optional
        IQR multiplier. Defaults to ``1.5``.

    Returns
    -------
    pd.DataFrame
        DataFrame with outlier rows removed and index reset.
    """
    mask = detect_outliers_iqr(df, columns, threshold)
    outlier_rows = mask.any(axis=1)
    n_removed = int(outlier_rows.sum())
    df_clean = df[~outlier_rows].reset_index(drop=True)
    logger.info("Removed %d outlier rows (%.2f%%).", n_removed, n_removed / len(df) * 100)
    return df_clean


def cap_outliers_iqr(
    df: pd.DataFrame,
    columns: List[str],
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Cap (winsorise) outlier values to the IQR fence values.

    Values below Q1 - threshold*IQR are replaced with the lower fence;
    values above Q3 + threshold*IQR are replaced with the upper fence.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : List[str]
        Columns to cap.
    threshold : float, optional
        IQR multiplier. Defaults to ``1.5``.

    Returns
    -------
    pd.DataFrame
        DataFrame with capped values (copy).
    """
    df_out = df.copy()
    for col in columns:
        q1 = df_out[col].quantile(0.25)
        q3 = df_out[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        df_out[col] = df_out[col].clip(lower=lower, upper=upper)
    logger.info("Capped outliers in %d columns.", len(columns))
    return df_out


# ---------------------------------------------------------------------------
# Feature scaling
# ---------------------------------------------------------------------------

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = "standard",
) -> Tuple[np.ndarray, np.ndarray, object]:
    """Fit a scaler on *X_train* and apply it to both splits.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    method : str, optional
        Scaling method — ``'standard'`` (zero mean, unit variance) or
        ``'minmax'`` (scale to [0, 1]). Defaults to ``'standard'``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, object]
        ``(X_train_scaled, X_test_scaled, fitted_scaler)``

    Raises
    ------
    ValueError
        If *method* is not ``'standard'`` or ``'minmax'``.
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method '{method}'. Use 'standard' or 'minmax'.")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Features scaled using %s scaler.", method)
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

def encode_categoricals(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode all object/category columns in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, LabelEncoder]]
        - DataFrame with categorical columns replaced by integer codes.
        - Mapping from column name to fitted ``LabelEncoder``.
    """
    df_out = df.copy()
    encoders: Dict[str, LabelEncoder] = {}
    cat_cols = df_out.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        df_out[col] = le.fit_transform(df_out[col].astype(str))
        encoders[col] = le
        logger.debug("Encoded column '%s' (%d classes).", col, len(le.classes_))

    logger.info("Encoded %d categorical column(s).", len(cat_cols))
    return df_out, encoders


# ---------------------------------------------------------------------------
# SMOTE oversampling
# ---------------------------------------------------------------------------

def apply_smote(
    X_train: np.ndarray,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to balance class distribution in training data.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature array.
    y_train : pd.Series
        Training target series.
    random_state : int, optional
        Random seed for reproducibility. Defaults to ``RANDOM_STATE``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(X_resampled, y_resampled)`` with balanced classes.
    """
    logger.info(
        "Applying SMOTE — class distribution before: %s",
        dict(pd.Series(y_train).value_counts().sort_index()),
    )
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(
        "SMOTE done — class distribution after: %s",
        dict(pd.Series(y_res).value_counts().sort_index()),
    )
    return X_res, y_res


# ---------------------------------------------------------------------------
# Train-test split
# ---------------------------------------------------------------------------

def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train-test split preserving class proportions.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float, optional
        Proportion of data for the test set. Defaults to ``TEST_SIZE``.
    random_state : int, optional
        Random seed. Defaults to ``RANDOM_STATE``.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        ``(X_train, X_test, y_train, y_test)``
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "Train-test split — train: %d, test: %d (test_size=%.2f).",
        len(X_train),
        len(X_test),
        test_size,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# PreprocessingPipeline class
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """End-to-end preprocessing pipeline for the diabetes dataset.

    Steps applied in order during ``fit_transform``:
    1. Handle missing values (mean imputation).
    2. Encode categorical columns.
    3. Cap outliers using IQR (for continuous features).
    4. Scale features (StandardScaler).

    The fitted state (encoders, scaler, column order) is preserved so
    that ``transform`` can be applied to new data consistently.

    Parameters
    ----------
    missing_strategy : str, optional
        Strategy for numeric imputation (``'mean'``, ``'median'``,
        ``'mode'``). Defaults to ``'mean'``.
    scale_method : str, optional
        Scaling method (``'standard'`` or ``'minmax'``).
        Defaults to ``'standard'``.
    outlier_threshold : float, optional
        IQR multiplier for outlier capping. Defaults to ``1.5``.
    outlier_columns : List[str], optional
        Columns to cap. If ``None``, all numeric columns are used.
    """

    def __init__(
        self,
        missing_strategy: str = "mean",
        scale_method: str = "standard",
        outlier_threshold: float = 1.5,
        outlier_columns: Optional[List[str]] = None,
    ) -> None:
        self.missing_strategy = missing_strategy
        self.scale_method = scale_method
        self.outlier_threshold = outlier_threshold
        self.outlier_columns = outlier_columns

        # State set during fit_transform
        self._fill_report: dict = {}
        self._encoders: Dict[str, LabelEncoder] = {}
        self._scaler: Optional[object] = None
        self._feature_columns: List[str] = []
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit the pipeline on *df* and return the transformed array.

        Parameters
        ----------
        df : pd.DataFrame
            Input feature DataFrame (must NOT contain the target column).

        Returns
        -------
        np.ndarray
            Scaled feature array ready for model training.
        """
        logger.info("PreprocessingPipeline: fit_transform started.")

        # Step 1 – missing values
        df_out, self._fill_report = handle_missing_values(df, self.missing_strategy)

        # Step 2 – encode categoricals
        df_out, self._encoders = encode_categoricals(df_out)

        # Step 3 – outlier capping
        num_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
        cap_cols = self.outlier_columns if self.outlier_columns is not None else num_cols
        cap_cols = [c for c in cap_cols if c in num_cols]
        df_out = cap_outliers_iqr(df_out, cap_cols, self.outlier_threshold)

        # Step 4 – scale (fit on full data passed; caller should pass X_train only)
        self._feature_columns = df_out.columns.tolist()
        if self.scale_method == "standard":
            self._scaler = StandardScaler()
        else:
            self._scaler = MinMaxScaler()
        X_scaled = self._scaler.fit_transform(df_out)

        self._fitted = True
        logger.info("PreprocessingPipeline: fit_transform complete — shape %s.", X_scaled.shape)
        return X_scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the fitted pipeline to new data.

        Parameters
        ----------
        df : pd.DataFrame
            New feature DataFrame with the same schema as training data.

        Returns
        -------
        np.ndarray
            Scaled feature array.

        Raises
        ------
        RuntimeError
            If the pipeline has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform before transform.")

        # Missing values
        df_out = df.copy()
        for col, fill_val in self._fill_report.items():
            if col in df_out.columns and df_out[col].isnull().any():
                df_out[col] = df_out[col].fillna(fill_val)

        # Encode categoricals
        for col, le in self._encoders.items():
            if col in df_out.columns:
                # Handle unseen labels gracefully
                known = set(le.classes_)
                df_out[col] = df_out[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df_out[col] = le.transform(df_out[col])

        # Outlier capping
        num_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
        cap_cols = self.outlier_columns if self.outlier_columns is not None else num_cols
        cap_cols = [c for c in cap_cols if c in num_cols]
        df_out = cap_outliers_iqr(df_out, cap_cols, self.outlier_threshold)

        # Align columns
        df_out = df_out.reindex(columns=self._feature_columns, fill_value=0)

        X_scaled = self._scaler.transform(df_out)
        return X_scaled

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the fitted pipeline to *path* using pickle.

        Parameters
        ----------
        path : str
            File path to save the pipeline (e.g., ``'pipeline.pkl'``).
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Pipeline saved to %s.", save_path)

    @classmethod
    def load(cls, path: str) -> "PreprocessingPipeline":
        """Load a previously saved pipeline from *path*.

        Parameters
        ----------
        path : str
            File path to a pickled ``PreprocessingPipeline``.

        Returns
        -------
        PreprocessingPipeline
            The loaded pipeline.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {load_path}")
        with open(load_path, "rb") as f:
            pipeline = pickle.load(f)
        logger.info("Pipeline loaded from %s.", load_path)
        return pipeline

    def __repr__(self) -> str:
        return (
            f"PreprocessingPipeline("
            f"missing_strategy={self.missing_strategy!r}, "
            f"scale_method={self.scale_method!r}, "
            f"fitted={self._fitted})"
        )
