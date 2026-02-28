import numpy as np
import pandas as pd

from src.modelling import (
    split_data,
    remove_outliers_iqr,
    preprocess_training_data,
    preprocess_test_data,
    train_model,
    evaluate_model,
    adjusted_r2
)


# ---------------------------------------------------------------------
# Create synthetic dataset for testing
# ---------------------------------------------------------------------

def create_sample_df():
    return pd.DataFrame({
        "life_expectancy": [70, 72, 75, 78, 80, 82, 85, 88],
        "gdp_per_capita": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
        "fertility_rate": [3.5, 3.2, 2.8, 2.5, 2.1, 1.9, 1.8, 1.6]
    })


# ---------------------------------------------------------------------
# Test 1: Data Splitting
# ---------------------------------------------------------------------

def test_split_data():
    df = create_sample_df()

    X_train, X_test, y_train, y_test = split_data(
        df, target="life_expectancy"
    )

    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)


# ---------------------------------------------------------------------
# Test 2: Outlier Removal
# ---------------------------------------------------------------------

def test_remove_outliers():
    X = pd.DataFrame({"feature": [1, 2, 3, 4, 100]})
    y = pd.Series([70, 71, 72, 73, 200])  # 200 is outlier

    X_filtered, y_filtered = remove_outliers_iqr(X, y)

    assert 200 not in y_filtered.values
    assert len(y_filtered) < len(y)


# ---------------------------------------------------------------------
# Test 3: Preprocessing
# ---------------------------------------------------------------------

def test_preprocessing_pipeline():
    df = create_sample_df()

    X_train, X_test, y_train, y_test = split_data(
        df, target="life_expectancy"
    )

    X_train_processed, imputer, scaler = preprocess_training_data(X_train)
    X_test_processed = preprocess_test_data(X_test, imputer, scaler)

    assert isinstance(X_train_processed, np.ndarray)
    assert isinstance(X_test_processed, np.ndarray)
    assert X_train_processed.shape[1] == X_test_processed.shape[1]


# ---------------------------------------------------------------------
# Test 4: Model Training & Evaluation
# ---------------------------------------------------------------------

def test_model_training_and_evaluation():
    df = create_sample_df()

    X_train, X_test, y_train, y_test = split_data(
        df, target="life_expectancy"
    )

    X_train_processed, imputer, scaler = preprocess_training_data(X_train)
    X_test_processed = preprocess_test_data(X_test, imputer, scaler)

    model = train_model(X_train_processed, y_train)

    rmse, r2 = evaluate_model(model, X_test_processed, y_test)

    assert isinstance(rmse, float)
    assert isinstance(r2, float)


# ---------------------------------------------------------------------
# Test 5: Adjusted R²
# ---------------------------------------------------------------------

def test_adjusted_r2():
    r2 = 0.8
    n = 100
    p = 3

    adj = adjusted_r2(r2, n, p)

    assert adj < r2
    assert isinstance(adj, float)

# ---------------------------------------------------------------------
# Test 6: Full ML Pipeline Integration
# ---------------------------------------------------------------------

def test_full_ml_pipeline_runs():
    df = create_sample_df()

    # Split
    X_train, X_test, y_train, y_test = split_data(
        df, target="life_expectancy"
    )

    # Outlier removal
    X_train, y_train = remove_outliers_iqr(X_train, y_train)

    # Preprocessing
    X_train_processed, imputer, scaler = preprocess_training_data(X_train)
    X_test_processed = preprocess_test_data(X_test, imputer, scaler)

    # Train + evaluate
    model = train_model(X_train_processed, y_train)
    rmse, r2 = evaluate_model(model, X_test_processed, y_test)

    assert rmse >= 0
    assert -1 <= r2 <= 1


# ---------------------------------------------------------------------
# Test 7: Cross-Validation Integration
# ---------------------------------------------------------------------

def test_cross_validation_scores():
    from sklearn.model_selection import cross_val_score

    df = create_sample_df()

    X_train, X_test, y_train, y_test = split_data(
        df, target="life_expectancy"
    )

    X_train_processed, imputer, scaler = preprocess_training_data(X_train)

    model = train_model(X_train_processed, y_train)

    scores = cross_val_score(
        model,
        X_train_processed,
        y_train,
        cv=3,
        scoring="r2"
    )

    assert len(scores) == 3
    assert isinstance(scores.mean(), float)


# ---------------------------------------------------------------------
# Test 8: Missing Value Imputation Works
# ---------------------------------------------------------------------

def test_imputer_removes_missing_values():
    df = create_sample_df()
    df.loc[0, "gdp_per_capita"] = np.nan  # Inject missing value

    X_train, X_test, y_train, y_test = split_data(
        df, target="life_expectancy"
    )

    X_train_processed, imputer, scaler = preprocess_training_data(X_train)

    assert not np.isnan(X_train_processed).any()


# ---------------------------------------------------------------------
# Test 9: Feature Scaling Standardises Data
# ---------------------------------------------------------------------

def test_scaling_centers_features():
    df = create_sample_df()

    X_train, X_test, y_train, y_test = split_data(
        df, target="life_expectancy"
    )

    X_train_processed, imputer, scaler = preprocess_training_data(X_train)

    means = X_train_processed.mean(axis=0)

    # Scaled features should be ~0 mean
    assert np.all(np.abs(means) < 1e-6)


# ---------------------------------------------------------------------
# Test 10: Model Coefficient Count Matches Features
# ---------------------------------------------------------------------

def test_model_coefficients_match_feature_count():
    df = create_sample_df()

    X_train, X_test, y_train, y_test = split_data(
        df, target="life_expectancy"
    )

    X_train_processed, imputer, scaler = preprocess_training_data(X_train)

    model = train_model(X_train_processed, y_train)

    assert len(model.coef_) == X_train_processed.shape[1]