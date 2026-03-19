"""
diabetes_prediction — v1-basic-ml package.

Exports key classes and functions from all sub-modules for convenient
top-level access:

    from src import (
        load_dataset, basic_info, validate_dataset, describe_dataset,
        get_feature_target_split, PreprocessingPipeline, scale_features,
        apply_smote, train_test_split_stratified,
        get_logistic_regression, get_random_forest, get_xgboost,
        get_lightgbm, get_neural_network,
        train_model, cross_validate_model, hyperparameter_tuning,
        save_model, load_model, ModelTrainer,
        calculate_metrics, compare_models, check_overfitting,
        per_class_metrics, confusion_matrix_metrics,
        classification_report_df, ModelEvaluator,
        plot_distribution, plot_correlation_heatmap, plot_box_plots,
        plot_target_distribution, plot_confusion_matrix,
        plot_roc_curves, plot_precision_recall_curves,
        plot_feature_importance, plot_model_comparison,
        plot_learning_curve,
    )
"""

# ---- data_loader -----------------------------------------------------------
from src.data_loader import (
    basic_info,
    describe_dataset,
    get_feature_target_split,
    load_dataset,
    print_basic_info,
    validate_dataset,
)

# ---- preprocessor ----------------------------------------------------------
from src.preprocessor import (
    PreprocessingPipeline,
    apply_smote,
    cap_outliers_iqr,
    detect_outliers_iqr,
    encode_categoricals,
    handle_missing_values,
    remove_outliers_iqr,
    scale_features,
    train_test_split_stratified,
)

# ---- models ----------------------------------------------------------------
from src.models import (
    ModelTrainer,
    cross_validate_model,
    get_lightgbm,
    get_logistic_regression,
    get_neural_network,
    get_random_forest,
    get_xgboost,
    hyperparameter_tuning,
    load_model,
    save_model,
    train_model,
)

# ---- evaluation ------------------------------------------------------------
from src.evaluation import (
    ModelEvaluator,
    calculate_metrics,
    check_overfitting,
    classification_report_df,
    compare_models,
    confusion_matrix_metrics,
    per_class_metrics,
)

# ---- visualization ---------------------------------------------------------
from src.visualization import (
    plot_box_plots,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_distribution,
    plot_feature_importance,
    plot_feature_target_relationship,
    plot_learning_curve,
    plot_model_comparison,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_target_distribution,
)

__all__ = [
    # data_loader
    "load_dataset",
    "basic_info",
    "print_basic_info",
    "validate_dataset",
    "describe_dataset",
    "get_feature_target_split",
    # preprocessor
    "handle_missing_values",
    "detect_outliers_iqr",
    "remove_outliers_iqr",
    "cap_outliers_iqr",
    "scale_features",
    "encode_categoricals",
    "apply_smote",
    "train_test_split_stratified",
    "PreprocessingPipeline",
    # models
    "get_logistic_regression",
    "get_random_forest",
    "get_xgboost",
    "get_lightgbm",
    "get_neural_network",
    "train_model",
    "cross_validate_model",
    "hyperparameter_tuning",
    "save_model",
    "load_model",
    "ModelTrainer",
    # evaluation
    "calculate_metrics",
    "per_class_metrics",
    "confusion_matrix_metrics",
    "classification_report_df",
    "compare_models",
    "check_overfitting",
    "ModelEvaluator",
    # visualization
    "plot_distribution",
    "plot_correlation_heatmap",
    "plot_box_plots",
    "plot_target_distribution",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "plot_feature_importance",
    "plot_model_comparison",
    "plot_learning_curve",
    "plot_feature_target_relationship",
]
