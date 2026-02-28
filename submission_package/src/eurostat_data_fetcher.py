"""
Module for fetching data from the Eurostat API.
"""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
from urllib3 import Retry

from requests.adapters import HTTPAdapter

def fetch_eurostat_dataset(
    dataset_code: str, filters: Dict[str, Any], filename: str
) -> pd.DataFrame:
    """
    Fetch dataset from the Eurostat API and save to CSV.

    Retrieves data from a Eurostat dataset in SDMX-JSON format, processes it
    into a tidy DataFrame, and saves it to a CSV file in the data/raw directory.

    Args:
        dataset_code: The Eurostat dataset code (e.g., 'hlth_silc_01').
        filters: Dictionary of filters to apply to the dataset.
                 Keys are dimension codes, values are lists of codes or a single code.
                 Example: {'geo': ['AT', 'BE'], 'time': ['2020', '2021']}
        filename: The name of the output CSV file (without .csv extension).

    Returns:
        A pandas DataFrame containing the processed dataset with columns:
        - country: Country/geographic code
        - year: Year (integer)
        - value: Indicator value

    Raises:
        requests.exceptions.RequestException: If the API request fails.
        ValueError: If the HTTP status code indicates an error.
        KeyError: If the response format is unexpected.

    Example:
        >>> filters = {'geo': ['AT', 'BE'], 'time': ['2020', '2021']}
        >>> df = fetch_eurostat_dataset('hlth_silc_01', filters, 'health_data')
        >>> df.head()
    """
    # Construct the API URL
    api_url = (
        f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
        f"{dataset_code}"
    )

    params = {"format": "JSON"}

    # Add filters to the URL
    for dim, codes in filters.items():
        if isinstance(codes, list):
            params[dim] = "+".join(codes)
        else:
            params[dim] = codes

    # Fetch data from the API

    # Create session with retry logic

    session = requests.Session()

    retry_strategy = Retry(
        total=5,                     # retry up to 5 times
        backoff_factor=1,            # exponential backoff
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    try:
        response = session.get(
            api_url,
            params=params,
            timeout=(5, 60)  # (connect timeout, read timeout)
        )

        response.raise_for_status()

    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            f"Eurostat API request failed: {error}"
        ) from error

    # Parse JSON response
    try:
        response_data: Dict[str, Any] = response.json()
    except requests.exceptions.JSONDecodeError as error:
        raise ValueError(
            f"Failed to parse JSON response: {error}"
        ) from error

    # Convert SDMX-style JSON to DataFrame
    df = sdmx_to_dataframe(response_data)

    # Ensure output directory exists
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    output_path = output_dir / f"{filename}.csv"
    df.to_csv(output_path, index=False)

    return df


def sdmx_to_dataframe(sdmx_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert SDMX-JSON format data to a tidy pandas DataFrame.

    Args:
        sdmx_data: The SDMX-JSON data dictionary from Eurostat API.

    Returns:
        A tidy DataFrame with columns: country, year, value.

    Raises:
        KeyError: If the expected SDMX structure is not found.
    """
    try:
        # Extract dimension information
        dimension = sdmx_data.get("dimension", {})
        values = sdmx_data.get("value", {})

        # Get dimension metadata
        geo_idx = None
        time_idx = None

        dimensions_list = []
        dimension_sizes = []

        for i, (dim_name, dim_data) in enumerate(dimension.items()):
            dimensions_list.append(dim_name)
            dimension_sizes.append(len(dim_data.get("category", {}).get("index", {})))

            if dim_name == "geo":
                geo_idx = i
            elif dim_name == "time":
                time_idx = i

        if geo_idx is None or time_idx is None:
            raise KeyError("Expected 'geo' and 'time' dimensions not found in data")

        # Build mapping from index to values
        geo_map = {
            idx: code
            for code, idx in dimension["geo"]["category"]["index"].items()
        }
        time_map = {
            idx: code
            for code, idx in dimension["time"]["category"]["index"].items()
        }

        # Convert SDMX value indices to DataFrame rows
        rows = []
        for value_idx_str, value in values.items():
            if value is None:
                continue

            # Decode the multi-dimensional index
            value_idx = int(value_idx_str)
            indices = decode_multidimensional_index(value_idx, dimension_sizes)

            country = geo_map.get(indices[geo_idx])
            year = time_map.get(indices[time_idx])

            if country and year:
                try:
                    year_int = int(year)
                except ValueError:
                    continue

                rows.append(
                    {"country": country, "year": year_int, "value": float(value)}
                )

        if not rows:
            raise ValueError("No valid data rows extracted from SDMX response")

        df = pd.DataFrame(rows)
        return df[["country", "year", "value"]].sort_values(
            by=["country", "year"]
        ).reset_index(drop=True)

    except KeyError as error:
        raise KeyError(
            f"Unexpected SDMX data structure: {error}"
        ) from error


def decode_multidimensional_index(
    value_idx: int, dimension_sizes: List[int]
) -> List[int]:
    """
    Decode a flattened multi-dimensional index into individual dimension indices.

    In SDMX-JSON, multi-dimensional data is stored with a single flattened index.
    This function reverses the flattening process using the size of each dimension.

    Args:
        value_idx: The flattened index value.
        dimension_sizes: List of sizes for each dimension in order.

    Returns:
        List of indices for each dimension.
    """
    indices = []
    remaining = value_idx

    for size in reversed(dimension_sizes):
        indices.insert(0, remaining % size)
        remaining //= size

    return indices


def fetch_life_expectancy() -> pd.DataFrame:
    """
    Fetch life expectancy data from Eurostat for 65 Years Old.

    Returns:
        A pandas DataFrame containing life expectancy data with columns:
        country, year, and value.
    """
    filters = {"sex": "T", "age": ["Y65"]}
    return fetch_eurostat_dataset("demo_r_mlifexp", filters, "life_expectancy")


def fetch_doctors_per_100k() -> pd.DataFrame:
    """
    Fetch practicing doctors per 100,000 population data from Eurostat.

    Returns:
        A pandas DataFrame containing doctors per 100k data with columns:
        country, year, and value.
    """
    filters = {"unit": "P_HTHAB"}
    return fetch_eurostat_dataset("hlth_rs_physreg", filters, "doctors_per_100k")


def fetch_household_expenditure() -> pd.DataFrame:
    """
    Fetch household expenditure data from Eurostat.

    Returns:
        A pandas DataFrame containing household expenditure data with columns:
        country, year, and value.
    """
    filters = {"unit": "CP_EUR_HAB","na_item": "P41"}
    return fetch_eurostat_dataset("nama_10_pc", filters, "household_expenditure")


def fetch_hospital_capacity() -> pd.DataFrame:
    """
    Fetch hospital capacity data from Eurostat.

    Returns:
        A pandas DataFrame containing hospital capacity data with columns:
        country, year, and value.
    """
    filters = {"facility": "HBEDT", "unit": "NR"}
    return fetch_eurostat_dataset("hlth_rs_bdsrg", filters, "hospital_capacity")


def fetch_gov_health_expenditure() -> pd.DataFrame:
    """
    Fetch government health expenditure data from Eurostat.

    Returns:
        A pandas DataFrame containing government health expenditure data with columns:
        country, year, and value.
    """
    filters = {"unit": "MIO_EUR", "sector": "S13", "cofog99": "GF07", "na_item": "TE"}
    return fetch_eurostat_dataset("gov_10a_exp", filters, "government_health_expenditure")
