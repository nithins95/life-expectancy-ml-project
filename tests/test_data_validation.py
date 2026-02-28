import pandas as pd
from src.data_loader import integrate_datasets


def test_master_dataset_validation():
    """
    Validate that the integrated dataset satisfies
    assessment requirements.
    """

    df = integrate_datasets()

    # 1️. Target column exists
    assert "life_expectancy" in df.columns

    # 2️. Target column is numeric
    assert pd.api.types.is_numeric_dtype(df["life_expectancy"])

    # 3️. No missing values in target
    assert df["life_expectancy"].isna().sum() == 0

    # 4️. At least 5 explanatory variables
    explanatory_vars = [
        col for col in df.columns
        if col not in ["life_expectancy", "country", "year"]
    ]

    assert len(explanatory_vars) >= 5

    # 5️. Dataset is not empty
    assert len(df) > 0