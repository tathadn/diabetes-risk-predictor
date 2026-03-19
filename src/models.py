"""
Model training utilities for the Diabetes Prediction project.

Provides factory functions for each supported model type, a unified
``train_model`` interface, cross-validation helpers, hyperparameter
tuning, persistence helpers, and a ``ModelTrainer`` management class.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from xgboost import XGBClassifier

from src.config import CV_FOLDS, RANDOM_STATE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def get_logistic_regression(params: Optional[Dict] = None) -> LogisticRegression:
    """Create a ``LogisticRegression`` classifier.

    Parameters
    ----------
    params : dict, optional
        Override default hyperparameters. Any keyword accepted by
        ``sklearn.linear_model.LogisticRegression`` is valid.

    Returns
    -------
    LogisticRegression
        Unfitted model instance.
    """
    defaults = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    if params:
        defaults.update(params)
    logger.debug("Creating LogisticRegression with params: %s", defaults)
    return LogisticRegression(**defaults)


def get_random_forest(params: Optional[Dict] = None) -> RandomForestClassifier:
    """Create a ``RandomForestClassifier``.

    Parameters
    ----------
    params : dict, optional
        Override default hyperparameters.

    Returns
    -------
    RandomForestClassifier
        Unfitted model instance.
    """
    defaults = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "class_weight": "balanced",
    }
    if params:
        defaults.update(params)
    logger.debug("Creating RandomForestClassifier with params: %s", defaults)
    return RandomForestClassifier(**defaults)


def get_xgboost(params: Optional[Dict] = None) -> XGBClassifier:
    """Create an ``XGBClassifier`` configured for binary classification.

    Parameters
    ----------
    params : dict, optional
        Override default hyperparameters.

    Returns
    -------
    XGBClassifier
        Unfitted model instance.
    """
    defaults = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    if params:
        defaults.update(params)
    logger.debug("Creating XGBClassifier with params: %s", defaults)
    return XGBClassifier(**defaults)


def get_lightgbm(params: Optional[Dict] = None) -> LGBMClassifier:
    """Create an ``LGBMClassifier`` configured for binary classification.

    Parameters
    ----------
    params : dict, optional
        Override default hyperparameters.

    Returns
    -------
    LGBMClassifier
        Unfitted model instance.
    """
    defaults = {
        "n_estimators": 200,
        "max_depth": -1,
        "learning_rate": 0.1,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        defaults.update(params)
    logger.debug("Creating LGBMClassifier with params: %s", defaults)
    return LGBMClassifier(**defaults)


def get_neural_network(input_dim: int, num_classes: int = 2):
    """Build a Keras dense neural network for binary classification.

    Architecture:
    - Dense(256, relu) + BatchNorm + Dropout(0.3)
    - Dense(128, relu) + BatchNorm + Dropout(0.3)
    - Dense(64, relu)  + BatchNorm + Dropout(0.2)
    - Dense(1, sigmoid)

    Parameters
    ----------
    input_dim : int
        Number of input features.
    num_classes : int, optional
        Unused for binary; kept for API compatibility. Defaults to 2.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for the neural network model. "
            "Install it with: pip install tensorflow"
        ) from exc

    tf.random.set_seed(RANDOM_STATE)

    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="DiabetesNN",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    logger.info("Neural network created — input_dim=%d (binary output).", input_dim)
    return model


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "",
) -> Any:
    """Fit *model* on the training data and return it.

    Parameters
    ----------
    model : estimator
        Scikit-learn compatible estimator with a ``fit`` method.
    X_train : array-like of shape (n_samples, n_features)
        Training features.
    y_train : array-like of shape (n_samples,)
        Training labels.
    model_name : str, optional
        Display name used in log messages.

    Returns
    -------
    fitted estimator
        The same *model* object after fitting.
    """
    label = model_name or type(model).__name__
    logger.info("Training %s on %d samples …", label, len(y_train))
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("%s trained in %.2f s.", label, elapsed)
    return model


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = CV_FOLDS,
    scoring: str = "roc_auc",
) -> dict:
    """Run stratified k-fold cross-validation and return aggregated scores.

    Parameters
    ----------
    model : estimator
        Unfitted (or cloned) scikit-learn estimator.
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    cv : int, optional
        Number of folds. Defaults to ``CV_FOLDS``.
    scoring : str, optional
        Scoring metric string accepted by ``sklearn``.
        Defaults to ``'roc_auc'``.

    Returns
    -------
    dict
        Keys: ``'mean_score'``, ``'std_score'``, ``'fold_scores'``,
        ``'mean_fit_time'``, ``'scoring'``.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=skf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )
    result = {
        "mean_score": float(cv_results["test_score"].mean()),
        "std_score": float(cv_results["test_score"].std()),
        "fold_scores": cv_results["test_score"].tolist(),
        "mean_train_score": float(cv_results["train_score"].mean()),
        "mean_fit_time": float(cv_results["fit_time"].mean()),
        "scoring": scoring,
    }
    logger.info(
        "CV (%s, %d folds) — %.4f ± %.4f",
        scoring,
        cv,
        result["mean_score"],
        result["std_score"],
    )
    return result


def hyperparameter_tuning(
    model: Any,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = CV_FOLDS,
    method: str = "random",
    n_iter: int = 20,
    scoring: str = "roc_auc",
) -> Tuple[Any, dict]:
    """Search for the best hyperparameters using grid or randomised search.

    Parameters
    ----------
    model : estimator
        Base estimator to tune.
    param_grid : dict
        Parameter grid (or distributions for random search).
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    cv : int, optional
        Cross-validation folds. Defaults to ``CV_FOLDS``.
    method : str, optional
        ``'random'`` for ``RandomizedSearchCV``, ``'grid'`` for
        ``GridSearchCV``. Defaults to ``'random'``.
    n_iter : int, optional
        Number of random parameter settings (used when method='random').
        Defaults to 20.
    scoring : str, optional
        Scoring metric. Defaults to ``'roc_auc'``.

    Returns
    -------
    Tuple[estimator, dict]
        ``(best_estimator, best_params_dict)``

    Raises
    ------
    ValueError
        If *method* is not ``'random'`` or ``'grid'``.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    if method == "random":
        searcher = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=skf,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
        )
    elif method == "grid":
        searcher = GridSearchCV(
            model,
            param_grid=param_grid,
            scoring=scoring,
            cv=skf,
            n_jobs=-1,
            verbose=1,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'random' or 'grid'.")

    logger.info("Starting hyperparameter search (%s) …", method)
    searcher.fit(X_train, y_train)
    best_params = searcher.best_params_
    logger.info("Best params: %s  |  best score: %.4f", best_params, searcher.best_score_)
    return searcher.best_estimator_, best_params


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(model: Any, path: str) -> None:
    """Serialise *model* to *path* using pickle.

    Parameters
    ----------
    model : object
        Any picklable Python object (typically a fitted estimator).
    path : str
        Destination file path.
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s.", save_path)


def load_model(path: str) -> Any:
    """Load a pickled model from *path*.

    Parameters
    ----------
    path : str
        Path to the pickled model file.

    Returns
    -------
    object
        The deserialised model.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Model file not found: {load_path}")
    with open(load_path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s.", load_path)
    return model


# ---------------------------------------------------------------------------
# ModelTrainer class
# ---------------------------------------------------------------------------


class ModelTrainer:
    """Manage training, evaluation, and comparison of multiple models.

    Example
    -------
    >>> trainer = ModelTrainer()
    >>> trainer.add_model('lr', get_logistic_regression())
    >>> trainer.train_all(X_train, y_train)
    >>> trainer.get_comparison()
    """

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self._results: Dict[str, dict] = {}
        self._trained: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------

    def add_model(self, name: str, model: Any) -> None:
        """Register a model under *name*.

        Parameters
        ----------
        name : str
            Unique identifier for the model.
        model : estimator
            Unfitted scikit-learn-compatible estimator.
        """
        self._models[name] = model
        self._trained[name] = False
        logger.info("Model '%s' registered.", name)

    def get_model(self, name: str) -> Any:
        """Return the (possibly fitted) model registered as *name*.

        Parameters
        ----------
        name : str
            Model identifier.

        Returns
        -------
        estimator

        Raises
        ------
        KeyError
            If no model with *name* has been registered.
        """
        if name not in self._models:
            raise KeyError(f"No model named '{name}'. Registered: {list(self._models)}")
        return self._models[name]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, name: str, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train the model registered under *name*.

        Parameters
        ----------
        name : str
            Model identifier.
        X_train : array-like
            Training features.
        y_train : array-like
            Training labels.

        Returns
        -------
        fitted estimator
        """
        model = self.get_model(name)
        fitted = train_model(model, X_train, y_train, model_name=name)
        self._models[name] = fitted
        self._trained[name] = True
        return fitted

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train all registered models.

        Parameters
        ----------
        X_train : array-like
            Training features.
        y_train : array-like
            Training labels.
        """
        for name in self._models:
            self.train(name, X_train, y_train)

    # ------------------------------------------------------------------
    # Results tracking
    # ------------------------------------------------------------------

    def record_result(self, name: str, metrics: dict) -> None:
        """Store evaluation metrics for model *name*.

        Parameters
        ----------
        name : str
            Model identifier.
        metrics : dict
            Dictionary of metric name → value.
        """
        self._results[name] = metrics

    def get_comparison(self) -> pd.DataFrame:
        """Return a DataFrame comparing all models across recorded metrics.

        Returns
        -------
        pd.DataFrame
            Rows are models, columns are metric names.
        """
        if not self._results:
            logger.warning("No results recorded yet. Call record_result first.")
            return pd.DataFrame()
        df = pd.DataFrame(self._results).T
        return df.sort_values("roc_auc", ascending=False) if "roc_auc" in df.columns else df

    def save_all(self, directory: str) -> None:
        """Save all trained models to *directory*.

        Parameters
        ----------
        directory : str
            Directory path. Created if it does not exist.
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        for name, model in self._models.items():
            if self._trained.get(name):
                save_path = dir_path / f"{name}.pkl"
                save_model(model, str(save_path))

    def list_models(self) -> List[str]:
        """Return names of all registered models.

        Returns
        -------
        List[str]
        """
        return list(self._models.keys())

    def __repr__(self) -> str:
        return (
            f"ModelTrainer("
            f"models={self.list_models()}, "
            f"trained={[k for k, v in self._trained.items() if v]})"
        )
