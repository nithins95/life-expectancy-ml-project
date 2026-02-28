"""
Modeling Module

Implements a proper ML lifecycle:

1. Data split (before preprocessing)
2. Outlier removal (training only)
3. Missing value imputation (training-fitted)
4. Feature scaling (training-fitted)
5. Model training (Linear Regression)
6. Evaluation (RMSE, R², Adjusted R²)
7. Cross-validation (5-fold)
"""

from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# 1️. Split Data (BEFORE preprocessing to prevent data leakage)
# ---------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    target: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into train and test sets."""
    numeric_df = df.select_dtypes(include=np.number)

    features = numeric_df.drop(columns=[target])
    target_series = numeric_df[target]

    return train_test_split(
        features, target_series,
        test_size=0.2,
        random_state=42
    )


# ---------------------------------------------------------------------
# 2️. Remove Outliers (Training Only)
# ---------------------------------------------------------------------

def remove_outliers_iqr(features, target_series):
    """Remove outliers using IQR method (training only)."""

    q1 = target_series.quantile(0.25)
    q3 = target_series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (target_series >= lower) & (target_series <= upper)

    return features[mask], target_series[mask]


# ---------------------------------------------------------------------
# 3️. Preprocess Training Data
# ---------------------------------------------------------------------

def preprocess_training_data(train_features):
    """Impute and scale training features."""

    # Impute missing values (fit on training only)
    imputer = SimpleImputer(strategy="mean")
    train_features_imputed = imputer.fit_transform(train_features)

    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features_imputed)

    return train_features_scaled, imputer, scaler

# ---------------------------------------------------------------------
# 4️. Preprocess Test Data
# ---------------------------------------------------------------------

def preprocess_test_data(test_features, imputer, scaler):
    """Transform test features using fitted imputer and scaler."""

    test_features_imputed = imputer.transform(test_features)
    test_features_scaled = scaler.transform(test_features_imputed)

    return test_features_scaled


# ---------------------------------------------------------------------
# 5️. Train Model
# ---------------------------------------------------------------------

def train_model(train_features, target_series):
    """Train Linear Regression model."""

    model = LinearRegression()
    model.fit(train_features, target_series)

    return model


# ---------------------------------------------------------------------
# 6️. Evaluation
# ---------------------------------------------------------------------

def evaluate_model(model, features, target_series):
    """Evaluate model performance using RMSE and R²."""

    predictions = model.predict(features)

    rmse = np.sqrt(mean_squared_error(target_series, predictions))
    r2 = r2_score(target_series, predictions)

    return rmse, r2


def adjusted_r2(r2: float, n: int, p: int) -> float:
    """Calculate Adjusted R²."""
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))


def print_model_report(results: dict) -> None:
    """Print formatted model evaluation report."""

    print("\nMODEL REPORT")
    print("-" * 60)

    print(f"Observations (train): {results['n_train']}")
    print(f"Features used: {results['n_features']}")

    print("\nTRAIN PERFORMANCE")
    print(f"R²: {results['train_r2']:.4f}")
    print(f"RMSE: {results['train_rmse']:.4f}")

    print("\nTEST PERFORMANCE")
    print(f"R²: {results['test_r2']:.4f}")
    print(f"Adjusted R²: {results['adjusted_r2']:.4f}")
    print(f"RMSE: {results['test_rmse']:.4f}")

    print("\nCROSS-VALIDATION (5-fold)")
    print(f"Mean R²: {results['cv_mean']:.4f}")
    print(f"Std Dev: {results['cv_std']:.4f}")


# ---------------------------------------------------------------------
# Full ML Pipeline
# ---------------------------------------------------------------------

def run_modelling(df: pd.DataFrame) -> None:
    """Execute full machine learning lifecycle pipeline."""

    print("\n" + "=" * 100)
    print("STEP 5: Modeling - Machine Learning Lifecycle")
    print("=" * 100)

    # 1️. Split
    train_features, test_features, train_target, test_target = split_data(
        df,
        target="life_expectancy"
    )

    # 2️. Remove outliers (training only)
    train_features, train_target = remove_outliers_iqr(train_features, train_target)

    # 3️. Preprocess
    train_features_processed, imputer, scaler = preprocess_training_data(
        train_features
    )

    test_features_processed = preprocess_test_data(
        test_features,
        imputer,
        scaler
    )

    # 4️. Train
    model = train_model(train_features_processed, train_target)

    # 5️. Evaluate
    train_rmse, train_r2 = evaluate_model(
        model, train_features_processed, train_target
    )

    test_rmse, test_r2 = evaluate_model(
        model, test_features_processed, test_target
    )

    adjusted = adjusted_r2(
        test_r2,
        len(test_target),
        test_features_processed.shape[1]
    )

    # 6️. Cross-validation
    cv_scores = cross_val_score(
        model,
        train_features_processed,
        train_target,
        cv=5,
        scoring="r2"
    )

# -----------------------------------------------------------------
# REPORT
# -----------------------------------------------------------------

    results = {
        "n_train": len(train_target),
        "n_features": train_features.shape[1],
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "adjusted_r2": adjusted,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }

    print_model_report(results)

    print("\n✓ Modeling completed successfully!\n")
