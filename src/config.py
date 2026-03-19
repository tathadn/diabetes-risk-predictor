"""
Configuration settings for the Diabetes Prediction project.

All paths are defined relative to the project root using pathlib.Path.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (two levels up from this file: src/ -> v1-basic-ml/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

# ---------------------------------------------------------------------------
# Results directories
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"
REPORTS_DIR = RESULTS_DIR / "reports"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Split / cross-validation settings
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
TARGET_COLUMN: str = "Diabetes_binary"

FEATURE_COLUMNS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]

ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------
MODEL_PARAMS = {
    "LogisticRegression": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "max_iter": [500, 1000],
        "class_weight": [None, "balanced"],
    },
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
    },
    "LightGBM": {
        "n_estimators": [100, 200, 300],
        "max_depth": [-1, 5, 10, 20],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [31, 63, 127],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
    },
}

# ---------------------------------------------------------------------------
# Minimum performance targets (used in evaluation checks)
# ---------------------------------------------------------------------------
PERFORMANCE_TARGETS = {
    "accuracy": 0.70,
    "roc_auc": 0.75,
    "f1_weighted": 0.65,
}

# ---------------------------------------------------------------------------
# Raw dataset filename (expected in data/raw/)
# ---------------------------------------------------------------------------
RAW_DATASET_FILENAME = "diabetes_binary.csv"
RAW_DATASET_PATH = RAW_DATA_DIR / RAW_DATASET_FILENAME

# ---------------------------------------------------------------------------
# Processed dataset filenames
# ---------------------------------------------------------------------------
X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.csv"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.csv"
X_TRAIN_SCALED_PATH = PROCESSED_DATA_DIR / "X_train_scaled.npy"
X_TEST_SCALED_PATH = PROCESSED_DATA_DIR / "X_test_scaled.npy"
