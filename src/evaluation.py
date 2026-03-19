"""
Model evaluation utilities for the Diabetes Prediction project.

Provides metric calculation, confusion-matrix analysis, cross-model
comparison, overfitting detection, and a ``ModelEvaluator`` class that
accumulates results for multiple models and generates reports.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> dict:
    """Calculate a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.
    y_prob : array-like of shape (n_samples, n_classes), optional
        Predicted class probabilities. Required for ROC-AUC.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``precision_macro``, ``precision_weighted``,
        ``recall_macro``, ``recall_weighted``, ``f1_macro``,
        ``f1_weighted``, ``roc_auc`` (if y_prob provided),
        ``specificity`` (binary or macro-averaged for multi-class).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    # ROC-AUC
    if y_prob is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
        except Exception as exc:
            logger.warning("ROC-AUC calculation failed: %s", exc)
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    # Specificity (macro-averaged across all classes treated as one-vs-rest)
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        denom = tn + fp
        specificities.append(tn / denom if denom > 0 else 0.0)
    metrics["specificity"] = float(np.mean(specificities))

    return metrics


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Return per-class precision, recall, F1, and support as a DataFrame.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    pd.DataFrame
        Index = class labels; columns = precision, recall, f1, support.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

    rows = []
    for cls in classes:
        binary_true = (y_true == cls).astype(int)
        binary_pred = (y_pred == cls).astype(int)
        rows.append(
            {
                "class": cls,
                "precision": precision_score(binary_true, binary_pred, zero_division=0),
                "recall": recall_score(binary_true, binary_pred, zero_division=0),
                "f1": f1_score(binary_true, binary_pred, zero_division=0),
                "support": int((y_true == cls).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("class")


def confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return TP, FP, TN, FN counts (macro-averaged over all classes OVR).

    For multi-class problems each class is treated as the positive class
    and the remaining are negative; the counts are averaged.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        Keys: ``TP``, ``FP``, ``TN``, ``FN``,
        ``per_class`` (list of per-class dicts).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(np.unique(y_true))
    per_class = []
    total_tp = total_fp = total_tn = total_fn = 0

    for i, cls in enumerate(classes):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        per_class.append({"class": cls, "TP": tp, "FP": fp, "TN": tn, "FN": fn})
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

    n = len(classes)
    return {
        "TP": total_tp // n,
        "FP": total_fp // n,
        "TN": total_tn // n,
        "FN": total_fn // n,
        "per_class": per_class,
    }


def classification_report_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Wrap sklearn's classification_report as a tidy DataFrame.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    pd.DataFrame
        Classification report with rows for each class + averages.
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).T
    df.index.name = "class"
    return df.round(4)


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------


def compare_models(results: Dict[str, dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from a mapping of model metrics.

    Parameters
    ----------
    results : dict
        ``{model_name: metrics_dict}`` where each metrics dict is returned
        by :func:`calculate_metrics`.

    Returns
    -------
    pd.DataFrame
        Rows are models sorted by ROC-AUC descending; columns are metrics.
    """
    df = pd.DataFrame(results).T
    df.index.name = "model"
    sort_col = "roc_auc" if "roc_auc" in df.columns else df.columns[0]
    return df.sort_values(sort_col, ascending=False).round(4)


# ---------------------------------------------------------------------------
# Overfitting check
# ---------------------------------------------------------------------------


def check_overfitting(
    train_metrics: dict,
    test_metrics: dict,
    threshold: float = 0.05,
) -> dict:
    """Check whether a model is overfitting by comparing train vs test scores.

    A model is considered to be overfitting on a metric if
    ``train_score - test_score > threshold``.

    Parameters
    ----------
    train_metrics : dict
        Metrics calculated on training data.
    test_metrics : dict
        Metrics calculated on test data.
    threshold : float, optional
        Allowable gap before flagging as overfitting. Defaults to 0.05.

    Returns
    -------
    dict
        Per-metric dict with keys ``train``, ``test``, ``gap``,
        ``overfit`` (bool), plus a top-level ``overall_overfit`` bool.
    """
    common_metrics = set(train_metrics) & set(test_metrics)
    result = {}
    any_overfit = False

    for metric in sorted(common_metrics):
        train_val = train_metrics[metric]
        test_val = test_metrics[metric]
        if not isinstance(train_val, (int, float)) or np.isnan(train_val):
            continue
        gap = float(train_val) - float(test_val)
        overfit = gap > threshold
        any_overfit = any_overfit or overfit
        result[metric] = {
            "train": round(float(train_val), 4),
            "test": round(float(test_val), 4),
            "gap": round(gap, 4),
            "overfit": overfit,
        }

    result["overall_overfit"] = any_overfit
    return result


# ---------------------------------------------------------------------------
# ModelEvaluator class
# ---------------------------------------------------------------------------


class ModelEvaluator:
    """Accumulate and report evaluation results for multiple models.

    Example
    -------
    >>> evaluator = ModelEvaluator()
    >>> evaluator.evaluate('lr', y_test, y_pred_lr, y_prob_lr)
    >>> evaluator.get_report()
    """

    def __init__(self) -> None:
        self._results: Dict[str, dict] = {}
        self._train_results: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> dict:
        """Evaluate a single model and store its metrics.

        Parameters
        ----------
        model_name : str
            Display name used as the row label in reports.
        y_true : array-like
            Ground-truth labels.
        y_pred : array-like
            Predicted labels.
        y_prob : array-like, optional
            Predicted probabilities.

        Returns
        -------
        dict
            Computed metrics dictionary.
        """
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        self._results[model_name] = metrics
        logger.info(
            "[%s] accuracy=%.4f  roc_auc=%.4f  f1_w=%.4f",
            model_name,
            metrics["accuracy"],
            metrics.get("roc_auc", float("nan")),
            metrics["f1_weighted"],
        )
        return metrics

    def evaluate_train(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> dict:
        """Evaluate on training data (used for overfitting detection).

        Parameters
        ----------
        model_name : str
            Model identifier (must match the test-set evaluation name).
        y_true, y_pred, y_prob
            Same semantics as :meth:`evaluate`.

        Returns
        -------
        dict
        """
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        self._train_results[model_name] = metrics
        return metrics

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def get_report(self) -> pd.DataFrame:
        """Return a DataFrame comparing all evaluated models.

        Returns
        -------
        pd.DataFrame
        """
        return compare_models(self._results)

    def get_overfitting_report(self, threshold: float = 0.05) -> pd.DataFrame:
        """Return an overfitting report for models with train metrics stored.

        Parameters
        ----------
        threshold : float, optional
            Gap threshold passed to :func:`check_overfitting`.

        Returns
        -------
        pd.DataFrame
            Multi-index DataFrame: (model, metric) with columns
            train, test, gap, overfit.
        """
        rows = []
        for name in self._train_results:
            if name not in self._results:
                continue
            overfit_info = check_overfitting(
                self._train_results[name], self._results[name], threshold
            )
            overall = overfit_info.pop("overall_overfit", None)
            for metric, vals in overfit_info.items():
                rows.append(
                    {
                        "model": name,
                        "metric": metric,
                        **vals,
                        "overall_overfit": overall,
                    }
                )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index(["model", "metric"])

    def best_model(self, metric: str = "roc_auc") -> str:
        """Return the name of the best model by *metric*.

        Parameters
        ----------
        metric : str, optional
            Metric to rank by. Defaults to ``'roc_auc'``.

        Returns
        -------
        str
            Name of the best-performing model.
        """
        if not self._results:
            raise ValueError("No evaluation results recorded yet.")
        return max(self._results, key=lambda m: self._results[m].get(metric, 0.0))

    def __repr__(self) -> str:
        return f"ModelEvaluator(models={list(self._results.keys())})"
