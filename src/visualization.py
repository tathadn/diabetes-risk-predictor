"""
Visualization utilities for the Diabetes Prediction project.

All functions use Matplotlib + Seaborn and optionally save figures to disk
when *save_path* is provided.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import learning_curve

logger = logging.getLogger(__name__)

# Global style settings
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 100, "font.size": 11})

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save the figure to *save_path* if provided, otherwise display it."""
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        logger.info("Figure saved to %s.", out)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------


def plot_distribution(
    df: pd.DataFrame,
    columns: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Plot histograms with KDE for each column in *columns*.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the columns to plot.
    columns : List[str]
        Column names to visualise.
    save_path : str, optional
        File path to save the figure.
    """
    n_cols = 3
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(columns):
        if col in df.columns:
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="steelblue")
            axes[i].set_title(f"Distribution: {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Count")
        else:
            axes[i].set_visible(False)

    # Hide unused axes
    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Plot a heatmap of the Pearson correlation matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric DataFrame.
    save_path : str, optional
        File path to save the figure.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Pearson Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Box plots
# ---------------------------------------------------------------------------


def plot_box_plots(
    df: pd.DataFrame,
    columns: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Plot box-plots for each column to visualise outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    columns : List[str]
        Columns to plot.
    save_path : str, optional
        File path to save the figure.
    """
    valid_cols = [c for c in columns if c in df.columns]
    n_cols = 3
    n_rows = int(np.ceil(len(valid_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(valid_cols):
        sns.boxplot(y=df[col].dropna(), ax=axes[i], color="lightcoral")
        axes[i].set_title(f"Boxplot: {col}")
        axes[i].set_ylabel(col)

    for j in range(len(valid_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Box Plots for Outlier Detection", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Target distribution
# ---------------------------------------------------------------------------


def plot_target_distribution(
    y: pd.Series,
    save_path: Optional[str] = None,
) -> None:
    """Bar chart of class distribution for the target variable.

    Parameters
    ----------
    y : pd.Series
        Target variable.
    save_path : str, optional
        File path to save the figure.
    """
    counts = y.value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(1)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index.astype(str), counts.values, color=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_title("Target Variable Distribution (Diabetes_binary)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

    for bar, p in zip(bars, pct.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{p}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Plot a normalised confusion matrix.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    model_name : str, optional
        Title prefix.
    save_path : str, optional
        File path to save the figure.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    title = f"Confusion Matrix — {model_name}" if model_name else "Confusion Matrix"
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------


def plot_roc_curves(
    models_results: Dict[str, dict],
    save_path: Optional[str] = None,
) -> None:
    """Overlay ROC curves for multiple models.

    Parameters
    ----------
    models_results : dict
        ``{model_name: {'y_true': ..., 'y_prob': ...}}``
        ``y_prob`` should be of shape (n_samples, n_classes) for multi-class.
    save_path : str, optional
        File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, (name, data) in enumerate(models_results.items()):
        y_true = np.asarray(data["y_true"])
        y_prob = np.asarray(data["y_prob"])
        classes = np.unique(y_true)
        color = colors[i % len(colors)]

        if len(classes) == 2:
            prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            fpr, tpr, _ = roc_curve(y_true, prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, label=f"{name} (AUC={roc_auc:.3f})", lw=2)
        else:
            # Macro-average OVR for multi-class
            all_fpr = np.linspace(0, 1, 200)
            mean_tpr = np.zeros_like(all_fpr)
            for cls_idx, cls in enumerate(classes):
                binary_true = (y_true == cls).astype(int)
                cls_prob = y_prob[:, cls_idx] if y_prob.ndim == 2 else y_prob
                try:
                    fpr, tpr, _ = roc_curve(binary_true, cls_prob)
                    mean_tpr += np.interp(all_fpr, fpr, tpr)
                except Exception:
                    pass
            mean_tpr /= len(classes)
            roc_auc = auc(all_fpr, mean_tpr)
            ax.plot(
                all_fpr, mean_tpr, color=color,
                label=f"{name} macro-avg (AUC={roc_auc:.3f})", lw=2,
            )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Precision-Recall curves
# ---------------------------------------------------------------------------


def plot_precision_recall_curves(
    models_results: Dict[str, dict],
    save_path: Optional[str] = None,
) -> None:
    """Overlay Precision-Recall curves for multiple models.

    Parameters
    ----------
    models_results : dict
        ``{model_name: {'y_true': ..., 'y_prob': ...}}``
    save_path : str, optional
        File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, (name, data) in enumerate(models_results.items()):
        y_true = np.asarray(data["y_true"])
        y_prob = np.asarray(data["y_prob"])
        classes = np.unique(y_true)
        color = colors[i % len(colors)]

        if len(classes) == 2:
            prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            precision, recall, _ = precision_recall_curve(y_true, prob)
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, color=color, label=f"{name} (PR-AUC={pr_auc:.3f})", lw=2)
        else:
            for cls_idx, cls in enumerate(classes):
                binary_true = (y_true == cls).astype(int)
                cls_prob = y_prob[:, cls_idx] if y_prob.ndim == 2 else y_prob
                try:
                    precision, recall, _ = precision_recall_curve(binary_true, cls_prob)
                    pr_auc = auc(recall, precision)
                    ax.plot(
                        recall, precision, color=color, alpha=0.6, lw=1.5,
                        label=f"{name} cls={cls} (AUC={pr_auc:.3f})",
                    )
                except Exception:
                    pass

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    save_path: Optional[str] = None,
) -> None:
    """Horizontal bar chart of feature importances.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with columns ``['feature', 'importance']``.
    top_n : int, optional
        Number of top features to display. Defaults to 15.
    save_path : str, optional
        File path to save the figure.
    """
    df_plot = (
        importance_df.nlargest(top_n, "importance")
        if "importance" in importance_df.columns
        else importance_df.head(top_n)
    )
    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.4)))
    ax.barh(df_plot["feature"], df_plot["importance"], color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Model comparison bar plots
# ---------------------------------------------------------------------------


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Grouped bar charts comparing models across multiple metrics.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of :func:`src.evaluation.compare_models`. Index = model names.
    metrics : List[str]
        Metric column names to plot (must exist in *comparison_df*).
    save_path : str, optional
        File path to save the figure.
    """
    valid_metrics = [m for m in metrics if m in comparison_df.columns]
    if not valid_metrics:
        logger.warning("None of the requested metrics found in comparison_df.")
        return

    n_metrics = len(valid_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]

    for ax, metric in zip(axes, valid_metrics):
        vals = comparison_df[metric].sort_values(ascending=False)
        ax.bar(vals.index, vals.values, color=colors[: len(vals)])
        ax.set_title(metric.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=30)
        for i, (idx, val) in enumerate(vals.items()):
            ax.text(i, val + 0.01, f"{val:.3f}", ha="center", fontsize=9)

    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------


def plot_learning_curve(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    save_path: Optional[str] = None,
) -> None:
    """Plot the learning curve (train and CV score vs training size).

    Parameters
    ----------
    model : estimator
        Unfitted scikit-learn estimator.
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    cv : int, optional
        Number of cross-validation folds. Defaults to 5.
    save_path : str, optional
        File path to save the figure.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_sizes, train_scores, test_scores = learning_curve(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
        )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", color="#4c72b0", label="Training score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#4c72b0")
    ax.plot(train_sizes, test_mean, "o-", color="#dd8452", label="CV score")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color="#dd8452")

    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Feature-target relationship
# ---------------------------------------------------------------------------


def plot_feature_target_relationship(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = "Diabetes_binary",
    save_path: Optional[str] = None,
) -> None:
    """Grouped bar charts showing mean feature values per target class.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset including *target_col*.
    features : List[str]
        Features to visualise.
    target_col : str, optional
        Name of the target column.
    save_path : str, optional
        File path to save the figure.
    """
    valid_features = [f for f in features if f in df.columns]
    n_cols = 3
    n_rows = int(np.ceil(len(valid_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(valid_features):
        grouped = df.groupby(target_col)[feat].mean()
        axes[i].bar(grouped.index.astype(str), grouped.values, color=["#4c72b0", "#dd8452", "#55a868"])
        axes[i].set_title(f"{feat} by {target_col}")
        axes[i].set_xlabel(target_col)
        axes[i].set_ylabel(f"Mean {feat}")

    for j in range(len(valid_features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature–Target Relationships", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, save_path)
