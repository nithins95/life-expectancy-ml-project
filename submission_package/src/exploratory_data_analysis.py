"""
Exploratory Data Analysis (EDA) - Summary Module
------------------------------------------------

This module is responsible for:

• Loading the integrated master dataset
• Performing descriptive statistics
• Conducting data quality checks
• Printing structured dataset summaries

NOTE:
This module does NOT generate plots.
All visualisation logic is handled in visualisations.py.
"""

# =============================
# Imports
# =============================

from pathlib import Path
import pandas as pd


# =============================
# Data Loading
# =============================

def load_data() -> pd.DataFrame:
    """
    Load the integrated master dataset from the processed directory.

    Returns:
        pd.DataFrame: Clean, integrated master dataset.
    """

    data_path = (
        Path(__file__).parent.parent
        / "data"
        / "processed"
        / "master_dataset.csv"
    )

    return pd.read_csv(data_path)


# =============================
# Summary & Data Quality Checks
# =============================

def basic_summary(df: pd.DataFrame) -> None:
    """
    Print structured dataset summary including:

    - Dataset shape
    - Sample rows
    - Descriptive statistics
    - Data types
    - Missing value analysis
    - Duplicate checks
    """

    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    # Dataset size
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Preview
    print("\nFirst 5 rows:")
    print(df.head())

    # Numerical summary
    print("\nDescriptive statistics:")
    print(df.describe().round(2))

    # Data types
    print("\nData types:")
    print(df.dtypes)

    # Missing values
    print("\nMissing values per column:")
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)

    print(pd.DataFrame({
        "Missing Count": missing_counts,
        "Missing %": missing_pct
    }))

    # Duplicate check
    print("\nDuplicate rows based on (iso3, year):")
    duplicates = df.duplicated(subset=["iso3", "year"], keep=False).sum()
    print(f"Duplicate rows: {duplicates}")

    print("=" * 80)


# =============================
# Orchestration Function
# =============================

def run_eda_summary() -> None:
    """
    Execute the EDA summary step.

    This function:
    1. Loads the master dataset
    2. Prints structured summary information
    """

    print("\n" + "=" * 100)
    print("STEP 3: Exploratory Data Summary")
    print("=" * 100)

    df = load_data()
    print(f"✓ Data loaded successfully: {df.shape}")

    basic_summary(df)
