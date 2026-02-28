"""
Main Execution Script
---------------------

This script orchestrates the full Life Expectancy ML pipeline.

Pipeline Steps:
1. Data Acquisition (Eurostat + World Bank APIs)
2. Data Processing & Integration
3. Exploratory Data Summary
4. Visualisation Generation
5. Modelling (Linear Regression)

This file should be executed from the project root directory.
"""

# =============================
# Import Pipeline Components
# =============================

from src.data_fetcher import run_data_acquisition
from src.data_loader import integrate_datasets
from src.exploratory_data_analysis import run_eda_summary
from src.visualizations import run_visualisations
from src.modelling import run_modelling


def main() -> None:
    """
    Execute the full end-to-end data pipeline.
    """

    # STEP 1: Data Acquisition
    print("=" * 100)
    print("STEP 1: Data Acquisition")
    print("=" * 100)

    try:
        run_data_acquisition()
    except Exception as e:
        print("⚠ Error during data acquisition.")
        print(e)
        return

    # STEP 2: Data Integration
    print("\n" + "=" * 100)
    print("STEP 2: Data Processing & Integration")
    print("=" * 100)

    try:
        df_master = integrate_datasets()
    except FileNotFoundError as e:
        print("⚠ Some datasets missing. Skipping integration.")
        print(e)
        return

    print("\n✓ Master dataset created successfully.")
    print("Final dataset shape:", df_master.shape)

    # STEP 3: Exploratory Data Summary
    print("\n" + "=" * 100)
    print("STEP 3: Exploratory Data Summary")
    print("=" * 100)

    try:
        run_eda_summary()
    except Exception as e:
        print("⚠ Error during EDA summary.")
        print(e)
        return

    # STEP 4: Generating Visualisations
    print("\n" + "=" * 100)
    print("STEP 4: Generating Visualisations")
    print("=" * 100)

    try:
        run_visualisations(df_master)
    except Exception as e:
        print("⚠ Error during visualisation execution.")
        print(e)
        return

    print("\n✓ Full pipeline completed successfully!")

  # STEP 5: Run Modelling
    print("\n" + "=" * 100)
    print("STEP 5: Running Modelling")
    print("=" * 100)
    
    try:
        run_modelling(df_master)
    except Exception as e:
        print("⚠ Error during modeling execution.")
        print(e)
        return
    


# Entry point of the script
if __name__ == "__main__":
    main()