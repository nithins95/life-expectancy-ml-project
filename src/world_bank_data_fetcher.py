"""
Module for fetching indicator data from the World Bank API.
"""

# ---------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------
from pathlib import Path

# ---------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# List of EU ISO3 country codes
# These codes will be joined using ";" in the API request
EU_ISO3 = [
    "AUT","BEL","BGR","HRV","CYP","CZE","DNK","EST","FIN","FRA",
    "DEU","GRC","HUN","IRL","ITA","LVA","LTU","LUX","MLT","NLD",
    "POL","PRT","ROU","SVK","SVN","ESP","SWE"
]

def create_retry_session() -> requests.Session:
    """Create requests session with retry strategy."""
    session = requests.Session()

    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    return session

def fetch_worldbank_indicator(indicator_code: str, filename: str) -> pd.DataFrame:

    """
    Fetch data for a given World Bank indicator and save it to CSV.

    Parameters
    ----------
    indicator_code : str
        World Bank indicator code (e.g., 'SP.DYN.TFRT.IN')
    filename : str
        Name of output CSV file (without extension)

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns:
        countryName, country (ISO3), year, value
    """

    # Join EU country codes into semicolon-separated string
    # Required format for World Bank API
    countries = ";".join(EU_ISO3)

    # Construct API URL
    url = f"https://api.worldbank.org/v2/country/{countries}/indicator/{indicator_code}"

    # Request parameters:
    # - format=json → return JSON format
    # - per_page=20000 → retrieve all results in one request
    params = {
        "format": "json",
        "per_page": 20000
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    # Create session with retry logic (CRITICAL for API stability)

    session = create_retry_session()

    # Make API request
    # Timeout set to 120 seconds because API response is slow
    try:
        response = session.get(
            url,
            params=params,
            headers=headers,
            timeout=(10, 120)  # (connect timeout, read timeout)
        )
        response.raise_for_status()

    # Raise exception if HTTP request failed (e.g., 500, 502 errors)
    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            f"World Bank API request failed: {error}"
        ) from error

    # Parse JSON response

    # World Bank returns:
    # data[0] → metadata
    # data[1] → actual records
    data = response.json()

    # Ensure response format is valid
    if len(data) < 2:
        raise ValueError("Unexpected API response format")

    records = data[1]

    # Convert JSON records to pandas DataFrame
    df = pd.DataFrame(records)

    # Clean and transform dataset

    # Extract country name from nested dictionary
    df["countryName"] = df["country"].apply(lambda x: x["value"])

    # Create ISO3 column
    df["country"] = df["countryiso3code"]

    # Rename date column to year
    df["year"] = df["date"]

    # Keep only relevant columns
    df = df[["countryName", "country", "year", "value"]]

    # Remove rows with missing indicator values
    df = df.dropna(subset=["value"])

    # Convert year column to integer
    df["year"] = df["year"].astype(int)

    # Save cleaned dataset to data/raw directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / f"{filename}.csv", index=False)

    #print(f"{filename}.csv saved successfully.")

    return df

    # Individual Indicator Wrappers

def fetch_gdp_per_capita() -> pd.DataFrame:
    """Fetch GDP per capita (current US$)."""
    return fetch_worldbank_indicator("NY.GDP.PCAP.CD", "gdp_per_capita")


def fetch_urban_population_percentage() -> pd.DataFrame:
    """Fetch urban population (% of total population)."""
    return fetch_worldbank_indicator("SP.URB.TOTL.IN.ZS", "urban_population_pct")


def fetch_fertility_rate() -> pd.DataFrame:
    """Fetch fertility rate (births per woman)."""
    return fetch_worldbank_indicator("SP.DYN.TFRT.IN", "fertility_rate")


def fetch_population_density() -> pd.DataFrame:
    """Fetch population density (people per sq. km)."""
    return fetch_worldbank_indicator("EN.POP.DNST", "population_density")
