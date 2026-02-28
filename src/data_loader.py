"""
Data processing and integration module.

This module:
1. Loads raw CSV files
2. Standardizes country codes
   - Eurostat: ISO2 → ISO3 conversion
   - World Bank: rename ISO3 column properly
3. Renames indicator columns
4. Merges all datasets into one master dataset
5. Saves processed dataset for modeling
"""

from pathlib import Path
import pandas as pd


# ISO2 → ISO3 mapping (EU countries only)
# Used to convert Eurostat country codes
COUNTRY_ISO2_TO_ISO3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "HR": "HRV",
    "CY": "CYP", "CZ": "CZE", "DK": "DNK", "EE": "EST",
    "FI": "FIN", "FR": "FRA", "DE": "DEU", "EL": "GRC",
    "GR": "GRC", "HU": "HUN", "IE": "IRL", "IT": "ITA",
    "LV": "LVA", "LT": "LTU", "LU": "LUX", "MT": "MLT",
    "NL": "NLD", "PL": "POL", "PT": "PRT", "RO": "ROU",
    "SK": "SVK", "SI": "SVN", "ES": "ESP", "SE": "SWE"
}


# Folder paths
RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")


# Load dataset from raw folder
def load_dataset(filename: str) -> pd.DataFrame:
    """
    Loads a CSV file from data/raw folder.
    """
    filepath = RAW_DATA_PATH / f"{filename}.csv"
    df = pd.read_csv(filepath)
    return df


# Standardize Eurostat datasets
def standardize_eurostat(df: pd.DataFrame, indicator_name: str) -> pd.DataFrame:
    """
    Standardizes Eurostat dataset:
    - Extract ISO2 country code
    - Convert ISO2 → ISO3
    - Keep iso3, year, value
    - Rename value column to indicator name
    """

    # Eurostat uses ISO2 country codes (e.g., AT, DE)
    if "country" in df.columns:
        df["iso2"] = df["country"]
    elif "geo" in df.columns:  # fallback if geo column exists
        df["iso2"] = df["geo"].str[:2]

    # Convert ISO2 → ISO3 using mapping dictionary
    df["iso3"] = df["iso2"].map(COUNTRY_ISO2_TO_ISO3)

    # Keep only required columns
    df = df[["iso3", "year", "value"]].copy()

    # Ensure year is integer
    df["year"] = df["year"].astype(int)

    # Rename value column to indicator name
    df = df.rename(columns={"value": indicator_name})

    return df


# Standardize World Bank datasets
def standardize_worldbank(df: pd.DataFrame, indicator_name: str) -> pd.DataFrame:
    """
    Standardizes World Bank dataset:
    - Rename ISO3 column to iso3
    - Keep iso3, year, value
    - Rename value column to indicator name
    """

    # World Bank CSV structure:
    # countryName | country (ISO3) | year | value

    # Rename ISO3 column
    if "country" in df.columns:
        df = df.rename(columns={"country": "iso3"})

    # Keep only required columns
    df = df[["iso3", "year", "value"]].copy()

    # Ensure year is integer
    df["year"] = df["year"].astype(int)

    # Filter to EU countries only to reduce memory usage
    eu_iso3 = set(COUNTRY_ISO2_TO_ISO3.values())
    df = df[df["iso3"].isin(eu_iso3)]

    # Rename value column to indicator
    df = df.rename(columns={"value": indicator_name})

    return df


# Integrate all datasets
def integrate_datasets() -> pd.DataFrame:
    """
    Loads, standardizes, and merges all datasets.
    Returns a final master dataset.
    """

    # -------- Eurostat datasets --------
    life = standardize_eurostat(
        load_dataset("life_expectancy"),
        "life_expectancy"
    )

    doctors = standardize_eurostat(
        load_dataset("doctors_per_100k"),
        "doctors_per_100k"
    )

    household = standardize_eurostat(
        load_dataset("household_expenditure"),
        "household_expenditure"
    )

    hospital_capacity = standardize_eurostat(
        load_dataset("hospital_capacity"),
        "hospital_capacity"
    )

    gov_health = standardize_eurostat(
        load_dataset("government_health_expenditure"),
        "gov_health_expenditure"
    )

    # -------- World Bank datasets --------
    gdp = standardize_worldbank(
        load_dataset("gdp_per_capita"),
        "gdp_per_capita"
    )

    fertility = standardize_worldbank(
        load_dataset("fertility_rate"),
        "fertility_rate"
    )

    urban = standardize_worldbank(
        load_dataset("urban_population_pct"),
        "urban_population_pct"
    )

    population_density = standardize_worldbank(
        load_dataset("population_density"),
        "population_density"
    )

    # Remove duplicates in each dataset
    life = life.drop_duplicates(subset=["iso3", "year"])
    doctors = doctors.drop_duplicates(subset=["iso3", "year"])
    household = household.drop_duplicates(subset=["iso3", "year"])
    hospital_capacity = hospital_capacity.drop_duplicates(subset=["iso3", "year"])
    gov_health = gov_health.drop_duplicates(subset=["iso3", "year"])
    gdp = gdp.drop_duplicates(subset=["iso3", "year"])
    fertility = fertility.drop_duplicates(subset=["iso3", "year"])
    urban = urban.drop_duplicates(subset=["iso3", "year"])
    population_density = population_density.drop_duplicates(subset=["iso3", "year"])

    print("Life:", life.shape)
    print("Doctors:", doctors.shape)
    print("Household:", household.shape)
    print("Hospital:", hospital_capacity.shape)
    print("Gov health:", gov_health.shape)
    print("GDP:", gdp.shape)
    print("Fertility:", fertility.shape)
    print("Urban:", urban.shape)
    print("Pop density:", population_density.shape)

    # -------- Merge all datasets --------
    dfs = [
        life, doctors, household, hospital_capacity,
        gov_health, gdp, fertility, urban, population_density
    ]

    for df in dfs:
        df.drop_duplicates(subset=["iso3", "year"], inplace=True)

    df_master = dfs[0]

    for df in dfs[1:]:
        df_master = df_master.merge(df, on=["iso3", "year"], how="inner")

    # Sort by country and year for readability
    df_master = df_master.sort_values(["iso3", "year"]).reset_index(drop=True)

    # Ensure processed folder exists
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    # Save final dataset
    df_master.to_csv(PROCESSED_DATA_PATH / "master_dataset.csv", index=False)

    return df_master
