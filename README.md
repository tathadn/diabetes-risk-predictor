# Diabetes Prediction — v1-basic-ml

A complete, production-quality machine learning project for predicting diabetes risk
using the **Diabetes Health Indicators Dataset** from the CDC Behavioral Risk Factor
Surveillance System (BRFSS), sourced from Kaggle.

---

## Table of Contents

1. [Project Description](#project-description)
2. [Dataset Information](#dataset-information)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage Guide](#usage-guide)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Preprocessing](#preprocessing)
8. [Model Training](#model-training)
9. [Results](#results)
10. [Final Predictions](#final-predictions)
11. [Dependencies](#dependencies)
12. [Contributing](#contributing)
13. [Tools Used](#tools-used)
14. [License](#license)

---

## Project Description

This project tackles a **binary classification** problem to identify whether an individual:
- `0` — Does **not** have diabetes or pre-diabetes
- `1` — Has **diabetes** or pre-diabetes

based on 21 health survey features (BMI, blood pressure, cholesterol levels, lifestyle factors, etc.).

### Key Features

- End-to-end ML pipeline from raw CSV to final predictions.
- Modular, reusable `src/` Python package with full type hints and docstrings.
- Five Jupyter notebooks covering EDA, preprocessing, training, evaluation, and prediction export.
- Four trained model types: Logistic Regression, Random Forest, XGBoost, LightGBM.
- Optional Keras neural network (3 hidden layers, batch normalisation, dropout).
- SMOTE oversampling to address severe class imbalance (~84% class 0).
- Comprehensive evaluation: ROC-AUC, F1-weighted, confusion matrices, learning curves.
- Pytest unit tests for preprocessing and model utilities.

---

## Dataset Information

**Source**: [Kaggle — Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

**Origin**: CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey.

| Property | Value |
|----------|-------|
| Records | ~253,680 |
| Features | 21 |
| Target | `Diabetes_binary` (0 = no diabetes, 1 = diabetes/pre-diabetes) |
| Missing values | None |
| File size | ~25 MB |

### Feature List

| Feature | Type | Description |
|---------|------|-------------|
| `HighBP` | binary | High blood pressure (1=yes) |
| `HighChol` | binary | High cholesterol (1=yes) |
| `CholCheck` | binary | Cholesterol check in past 5 years |
| `BMI` | continuous | Body Mass Index |
| `Smoker` | binary | Smoked ≥ 100 cigarettes lifetime |
| `Stroke` | binary | Ever told had a stroke |
| `HeartDiseaseorAttack` | binary | Coronary heart disease or MI |
| `PhysActivity` | binary | Physical activity in past 30 days |
| `Fruits` | binary | Eats fruit 1+ times per day |
| `Veggies` | binary | Eats vegetables 1+ times per day |
| `HvyAlcoholConsump` | binary | Heavy alcohol consumption |
| `AnyHealthcare` | binary | Any kind of health care coverage |
| `NoDocbcCost` | binary | Could not see doctor due to cost |
| `GenHlth` | ordinal | General health (1=excellent, 5=poor) |
| `MentHlth` | continuous | Days of poor mental health (0-30) |
| `PhysHlth` | continuous | Days of poor physical health (0-30) |
| `DiffWalk` | binary | Serious difficulty walking/climbing |
| `Sex` | binary | 0=female, 1=male |
| `Age` | ordinal | Age category (1-13) |
| `Education` | ordinal | Education level (1-6) |
| `Income` | ordinal | Income level (1-8) |

### Download Instructions

1. Create a free [Kaggle account](https://www.kaggle.com) if you do not have one.
2. Visit the [dataset page](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).
3. Download `diabetes_binary_health_indicators_BRFSS2015.csv`.
4. Rename the file to `diabetes_binary.csv`.
5. Place it in `data/raw/diabetes_binary.csv`.

> **Note**: All notebooks gracefully detect a missing dataset and fall back to a small
> synthetic demo dataset so you can explore the code without the real data.

---

## Project Structure

```
v1-basic-ml/
├── notebooks/
│   ├── 01_eda_analysis.ipynb        # Exploratory data analysis
│   ├── 02_preprocessing.ipynb       # Data cleaning and feature engineering
│   ├── 03_model_training.ipynb      # Train LR, RF, XGBoost, LightGBM, NN
│   ├── 04_evaluation.ipynb          # Metrics, curves, overfitting analysis
│   └── 05_final_predictions.ipynb   # Export predictions + model card
├── src/
│   ├── __init__.py                  # Package exports
│   ├── config.py                    # Paths, constants, hyperparameter grids
│   ├── data_loader.py               # Load, validate, describe dataset
│   ├── preprocessor.py              # Missing values, outliers, scaling, SMOTE
│   ├── models.py                    # Model factories, trainer, persistence
│   ├── evaluation.py                # Metrics, comparison, overfitting checks
│   └── visualization.py            # Matplotlib/Seaborn plotting functions
├── data/
│   ├── raw/                         # Place diabetes_binary.csv here
│   ├── processed/                   # Auto-generated by notebook 02
│   └── predictions/                 # Auto-generated by notebook 05
├── results/
│   ├── plots/                       # All saved figures
│   ├── models/                      # Saved .pkl model files
│   └── reports/                     # Evaluation CSVs, model card
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py         # Pytest tests for preprocessor
│   └── test_models.py               # Pytest tests for models
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone / download the project

```bash
git clone <repo-url>
cd v1-basic-ml
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# or
venv\Scripts\activate.bat       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the package in development mode

```bash
pip install -e .
```

### 5. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name=diabetes_ml --display-name "Diabetes ML"
```

---

## Usage Guide

### Phase 1 — Exploratory Data Analysis

Open and run `notebooks/01_eda_analysis.ipynb`:

```bash
jupyter notebook notebooks/01_eda_analysis.ipynb
```

### Phase 2 — Preprocessing

Run `notebooks/02_preprocessing.ipynb`.

### Phase 3 — Model Training

Run `notebooks/03_model_training.ipynb`.

### Phase 4 — Evaluation

Run `notebooks/04_evaluation.ipynb`.

### Phase 5 — Final Predictions

Run `notebooks/05_final_predictions.ipynb`.

### Running Tests

```bash
pytest tests/ -v --tb=short
```

To generate a coverage report:

```bash
pytest tests/ --cov=src --cov-report=html
```

---

## Exploratory Data Analysis

Notebook `01_eda_analysis.ipynb` explores the raw dataset through distributions, correlations, and feature-target relationships.

### Target Distribution & Class Balance

The dataset has a severe class imbalance: ~84% of records are class 0 (no diabetes) and ~16% are class 1 (diabetes/pre-diabetes). This imbalance motivates the use of SMOTE during preprocessing.

| | |
|---|---|
| ![Target Distribution](results/plots/target_distribution.png) | ![Class Balance](results/plots/class_balance.png) |

### Feature Distributions

Distribution of all 21 features across the dataset. Binary features show strong skews; continuous features (BMI, MentHlth, PhysHlth) are right-skewed.

![Feature Distributions](results/plots/feature_distributions.png)

### Correlation Heatmap

Pairwise Pearson correlations across all features. Notable clusters of positively correlated health indicators (HighBP, HighChol, HeartDiseaseorAttack) suggest compounding risk factors.

![Correlation Heatmap](results/plots/correlation_heatmap.png)

### Feature–Target Relationships

Mean feature values stratified by diabetes status. Features with the largest gap between classes (GenHlth, HighBP, BMI, Age) are the most predictive.

![Feature–Target Relationships](results/plots/feature_target_relationships.png)

### Outlier Analysis (Box Plots)

Box plots of continuous and ordinal features identifying extreme values. BMI, MentHlth, and PhysHlth contain significant upper-tail outliers.

![Box Plots](results/plots/box_plots.png)

---

## Preprocessing

Notebook `02_preprocessing.ipynb` prepares the data for modelling through the following steps:

1. **Outlier capping** — IQR-based capping on continuous columns (BMI, MentHlth, PhysHlth).
2. **Train-test split** — Stratified 80/20 split (~202,944 train / ~50,736 test).
3. **Scaling** — `StandardScaler` fit exclusively on training data, applied to both splits.
4. **SMOTE** — Synthetic Minority Oversampling applied to the training set only to balance classes.

### Outlier Treatment

| Before | After |
|--------|-------|
| ![Outliers Before](results/plots/outliers_before.png) | ![Outliers After](results/plots/outliers_after.png) |

IQR-based capping effectively removes extreme values while preserving the underlying distributions.

---

## Model Training

Notebook `03_model_training.ipynb` trains four classifiers and runs cross-validation.

### Models Trained

| Model | Key Hyperparameters |
|-------|---------------------|
| Logistic Regression | `C=1.0`, `max_iter=1000`, `solver=lbfgs` |
| Random Forest | `n_estimators=100`, `max_depth=None` |
| XGBoost | Tuned via `RandomizedSearchCV` (50 iterations, 3-fold CV) |
| LightGBM | Default with early stopping |

### Cross-Validation Comparison

3-fold stratified cross-validation ROC-AUC scores across all models. LightGBM and XGBoost consistently outperform the others.

![CV Comparison](results/plots/cv_comparison.png)

---

## Results

All models were evaluated on the held-out 20% test split (~50,736 samples).

### Overall Performance Comparison

![Model Comparison](results/plots/model_comparison.png)

### Full Metrics Table

| Model | Accuracy | ROC-AUC | F1-weighted | F1-macro | Precision (weighted) | Recall (weighted) | Specificity |
|-------|----------|---------|-------------|----------|----------------------|-------------------|-------------|
| **LightGBM** ⭐ | **86.4%** | **82.6%** | **83.6%** | 60.8% | 83.3% | 86.4% | 58.6% |
| XGBoost | 86.3% | 82.4% | 84.0% | 62.1% | 83.5% | 86.3% | 59.7% |
| Logistic Regression | 73.5% | 82.1% | 77.3% | 63.5% | 86.1% | 73.5% | 74.6% |
| Random Forest | 78.6% | 81.8% | 81.0% | 66.1% | 85.2% | 78.6% | 72.4% |

**Best model: LightGBM** (selected by ROC-AUC). All models exceeded every performance target.

### Performance Targets

| Metric | Target | Best Achieved |
|--------|--------|---------------|
| Accuracy | ≥ 0.70 | **0.864** (LightGBM) |
| ROC-AUC | ≥ 0.75 | **0.826** (LightGBM) |
| F1-weighted | ≥ 0.65 | **0.840** (XGBoost) |

> **Note on class imbalance**: The dataset is heavily skewed (~84% class 0). SMOTE oversampling is applied to the training set to mitigate this. Macro-averaged metrics reflect per-class performance more honestly than accuracy alone.

### ROC Curves

Receiver Operating Characteristic curves for all four models. LightGBM achieves the highest AUC (0.826), closely followed by XGBoost (0.824).

![ROC Curves](results/plots/roc_curves.png)

### Precision-Recall Curves

Precision-Recall curves are more informative under class imbalance. A higher area under the PR curve indicates better performance on the minority class (diabetes).

![Precision-Recall Curves](results/plots/precision_recall_curves.png)

### Confusion Matrices

Per-model confusion matrices on the test set. Each cell shows absolute counts (top) and row-normalised rates (bottom).

| LightGBM | XGBoost |
|----------|---------|
| ![Confusion Matrix — LightGBM](results/plots/confusion_matrix_LightGBM.png) | ![Confusion Matrix — XGBoost](results/plots/confusion_matrix_XGBoost.png) |

| Random Forest | Logistic Regression |
|---------------|---------------------|
| ![Confusion Matrix — Random Forest](results/plots/confusion_matrix_RandomForest.png) | ![Confusion Matrix — Logistic Regression](results/plots/confusion_matrix_LogisticRegression.png) |

### Feature Importance

Feature importance scores extracted from tree-based models. **GenHlth**, **BMI**, **Age**, **HighBP**, and **HighChol** consistently rank as the top predictors across all three tree models.

#### LightGBM Feature Importance
![Feature Importance — LightGBM](results/plots/feature_importance_LightGBM.png)

#### XGBoost Feature Importance
![Feature Importance — XGBoost](results/plots/feature_importance_XGBoost.png)

#### Random Forest Feature Importance
![Feature Importance — Random Forest](results/plots/feature_importance_RandomForest.png)

### Learning Curve — LightGBM

The learning curve for the best model shows training and cross-validation scores as a function of training set size. The narrow gap between train and CV scores indicates low overfitting.

![Learning Curve — LightGBM](results/plots/learning_curve_LightGBM.png)

---

## Final Predictions

Notebook `05_final_predictions.ipynb` loads the best model (LightGBM) and exports predictions on the test set.

### Output

- **`data/predictions/predictions.csv`** — Contains original features, true label, predicted class, per-class probabilities, and confidence score for each test sample.
- **`results/reports/model_card.md`** — Full model card with performance summary.

### Prediction Confidence Distribution

Distribution of the model's confidence (max predicted probability) across all test samples. High confidence on both classes indicates a well-calibrated model.

![Confidence Distribution](results/plots/confidence_distribution.png)

---

## Dependencies

Core libraries:

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scikit-learn` | ML algorithms, preprocessing, evaluation |
| `xgboost` | Gradient boosting |
| `lightgbm` | Fast gradient boosting |
| `imbalanced-learn` | SMOTE oversampling |
| `tensorflow` | Neural network (optional) |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualisation |
| `joblib` | Model serialisation |
| `pytest` | Unit testing |

See `requirements.txt` for pinned versions.

---

## Contributing

Contributions, bug reports, and feature requests are welcome.

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Write tests for new functionality in `tests/`.
4. Ensure all tests pass: `pytest tests/ -v`.
5. Format your code: `black src/ tests/`.
6. Open a pull request with a clear description of your changes.

---

## Tools Used

| Tool | Purpose |
|------|---------|
| [Claude](https://claude.ai) (Anthropic) | AI assistant used during development |

---

## License

This project is released under the MIT License.

The dataset is sourced from the CDC BRFSS and made available on Kaggle under its
respective terms of use.
