"""
Data loading and validation utilities for the Diabetes Prediction project.

This module provides functions to load the Diabetes Health Indicators Dataset,
inspect it, validate its schema, and split it into features and target.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """Load the diabetes dataset from a CSV file.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a DataFrame.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at *path*.
    ValueError
        If the file is empty or cannot be parsed as CSV.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'.\n"
            "Please download the Diabetes Health Indicators Dataset from Kaggle:\n"
            "  https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset\n"
            f"and place the CSV file at: {csv_path}"
        )

    logger.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"The file at '{csv_path}' is empty.")

    logger.info("Dataset loaded: %d rows x %d columns", *df.shape)
    return df


def basic_info(df: pd.DataFrame) -> dict:
    """Return a dictionary of basic dataset information.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to inspect.

    Returns
    -------
    dict
        Keys: 'shape', 'columns', 'dtypes', 'missing_values',
        'missing_pct', 'duplicates', 'memory_usage_mb'.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    info: dict = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": missing[missing > 0].to_dict(),
        "missing_pct": missing_pct[missing_pct > 0].to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 3),
    }
    return info


def get_feature_target_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features (X) and target (y).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including the target column.
    target_col : str, optional
        Name of the target column. Defaults to ``TARGET_COLUMN`` from config.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) where X contains all columns except *target_col*.

    Raises
    ------
    KeyError
        If *target_col* is not found in the DataFrame.
    """
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]
    logger.info(
        "Feature-target split — X: %s, y: %s (classes: %s)",
        X.shape,
        y.shape,
        sorted(y.unique().tolist()),
    )
    return X, y


def validate_dataset(df: pd.DataFrame) -> dict:
    """Validate the dataset against expected schema.

    Checks that:
    - All expected feature columns are present.
    - The target column is present.
    - Target values are within {0, 1, 2}.
    - No column has 100% missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.

    Returns
    -------
    dict
        Keys: 'valid' (bool), 'missing_columns' (list),
        'unexpected_columns' (list), 'target_values' (list),
        'invalid_target_values' (list), 'fully_missing_columns' (list),
        'warnings' (list).
    """
    expected_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    actual_cols = set(df.columns)

    missing_cols = sorted(expected_cols - actual_cols)
    unexpected_cols = sorted(actual_cols - expected_cols)

    target_vals: List = []
    invalid_target_vals: List = []
    if TARGET_COLUMN in df.columns:
        target_vals = sorted(df[TARGET_COLUMN].dropna().unique().tolist())
        invalid_target_vals = [v for v in target_vals if v not in {0, 1, 2}]

    fully_missing = df.columns[df.isnull().all()].tolist()

    warnings: List[str] = []
    if missing_cols:
        warnings.append(f"Missing expected columns: {missing_cols}")
    if unexpected_cols:
        warnings.append(f"Unexpected columns found: {unexpected_cols}")
    if invalid_target_vals:
        warnings.append(f"Invalid target values: {invalid_target_vals} (expected 0, 1, 2)")
    if fully_missing:
        warnings.append(f"Columns with 100% missing: {fully_missing}")

    is_valid = not missing_cols and not invalid_target_vals and not fully_missing

    result = {
        "valid": is_valid,
        "missing_columns": missing_cols,
        "unexpected_columns": unexpected_cols,
        "target_values": target_vals,
        "invalid_target_values": invalid_target_vals,
        "fully_missing_columns": fully_missing,
        "warnings": warnings,
    }

    if is_valid:
        logger.info("Dataset validation passed.")
    else:
        logger.warning("Dataset validation failed: %s", warnings)

    return result


def describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return an extended statistical summary of the DataFrame.

    Extends ``pd.DataFrame.describe()`` with skewness and kurtosis rows
    for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to summarise.

    Returns
    -------
    pd.DataFrame
        Transposed statistical summary including count, mean, std, min,
        25%, 50%, 75%, max, skewness, and kurtosis.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    base = numeric_df.describe().T

    base["skewness"] = numeric_df.skew()
    base["kurtosis"] = numeric_df.kurtosis()

    return base.round(4)


# ---------------------------------------------------------------------------
# Helper: pretty-print basic_info
# ---------------------------------------------------------------------------

def print_basic_info(df: pd.DataFrame) -> None:
    """Print a formatted summary of basic dataset information.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to summarise.
    """
    info = basic_info(df)
    print(f"Shape           : {info['shape']}")
    print(f"Duplicates      : {info['duplicates']}")
    print(f"Memory (MB)     : {info['memory_usage_mb']}")
    print(f"Missing columns : {info['missing_values'] or 'None'}")
    print(f"\nData types:\n{pd.Series(info['dtypes']).value_counts().to_string()}")
