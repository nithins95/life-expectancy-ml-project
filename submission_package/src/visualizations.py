"""
Visualisations Module

Generates and saves all visualisations for the Life Expectancy ML Project.

Design Principles:
- Uses Agg backend (stable for scripts)
- Avoids seaborn (prevents rendering freezes)
- Controls DPI for stability
- Samples large datasets safely
- Closes figures after saving (prevents memory leaks)
"""

# ---------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------
from pathlib import Path

# ---------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------
# Backend Configuration (CRITICAL for stability)
# ---------------------------------------------------------------------
matplotlib.use("Agg")  # Non-interactive backend

# ---------------------------------------------------------------------
# Global Plot Configuration
# ---------------------------------------------------------------------
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 10


# ---------------------------------------------------------------------
# Safe Save Function
# ---------------------------------------------------------------------
def save_figure(fig, output_path: Path) -> None:
    """
    Save figure safely with controlled DPI.
    """

    output_path = output_path.with_suffix(".png")

    fig.savefig(
        output_path,
        dpi=100,               # Keep low for stability
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)
    print(f"✓ Saved: {output_path}")


# ---------------------------------------------------------------------
# 1️. Correlation Heatmap (Pure Matplotlib)
# ---------------------------------------------------------------------
def plot_correlation(df: pd.DataFrame, figures_dir: Path) -> None:
    """Generate and save correlation heatmap of numeric features."""
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["year"], errors="ignore")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr, cmap="coolwarm", aspect="auto")

    fig.colorbar(cax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    ax.set_title("Correlation Heatmap - Life Expectancy Dataset")

    save_figure(fig, figures_dir / "correlation_heatmap")


# ---------------------------------------------------------------------
# 2️. Life Expectancy Trend Over Time
# ---------------------------------------------------------------------
def plot_trends(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot life expectancy trends over time for all countries."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for iso in df["iso3"].unique():
        country_data = df[df["iso3"] == iso].sort_values("year")
        ax.plot(
            country_data["year"],
            country_data["life_expectancy"],
            linewidth=1,
            alpha=0.7
        )

    ax.set_title("Life Expectancy Trends Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Life Expectancy")

    save_figure(fig, figures_dir / "life_expectancy_trends")


# ---------------------------------------------------------------------
# 3️. GDP vs Life Expectancy
# ---------------------------------------------------------------------
def plot_gdp_relationship(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot GDP per capita vs life expectancy with log-scale regression."""
    data = df[["gdp_per_capita", "life_expectancy"]].dropna()

    # Sample large datasets (prevents freeze)
    if len(data) > 3000:
        data = data.sample(3000, random_state=42)

    fig, ax = plt.subplots(figsize=(8, 6))

    x = data["gdp_per_capita"]
    y = data["life_expectancy"]

    ax.scatter(x, y, s=20, alpha=0.6)

    # Log scale (economically correct)
    ax.set_xscale("log")

    # Regression (log-x)
    z = np.polyfit(np.log(x), y, 1)
    p = np.poly1d(z)

    x_sorted = np.sort(x)
    ax.plot(x_sorted, p(np.log(x_sorted)), linewidth=2)

    ax.set_xlabel("GDP per Capita (log scale)")
    ax.set_ylabel("Life Expectancy")
    ax.set_title("GDP vs Life Expectancy")

    save_figure(fig, figures_dir / "gdp_vs_life_expectancy")


# ---------------------------------------------------------------------
# 4️. Fertility vs Life Expectancy
# ---------------------------------------------------------------------
def plot_fertility_relationship(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot fertility rate vs life expectancy with regression line."""
    data = df[["fertility_rate", "life_expectancy"]].dropna()

    if len(data) > 3000:
        data = data.sample(3000, random_state=42)

    fig, ax = plt.subplots(figsize=(8, 6))

    x = data["fertility_rate"]
    y = data["life_expectancy"]

    ax.scatter(x, y, s=20, alpha=0.6)

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    x_sorted = np.sort(x)
    ax.plot(x_sorted, p(x_sorted), linewidth=2)

    ax.set_xlabel("Fertility Rate")
    ax.set_ylabel("Life Expectancy")
    ax.set_title("Fertility Rate vs Life Expectancy")

    save_figure(fig, figures_dir / "fertility_vs_life_expectancy")


# ---------------------------------------------------------------------
# 5️. Distribution of Life Expectancy
# ---------------------------------------------------------------------
def plot_distribution(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot histogram and KDE distribution of life expectancy."""
    life_exp = df["life_expectancy"].dropna()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(life_exp, bins=30, density=True, alpha=0.7)

    # KDE (safe version)
    kde = gaussian_kde(life_exp)
    x_vals = np.linspace(life_exp.min(), life_exp.max(), 200)
    ax.plot(x_vals, kde(x_vals), linewidth=2)

    ax.set_xlabel("Life Expectancy")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Life Expectancy")

    save_figure(fig, figures_dir / "life_expectancy_distribution")


# ---------------------------------------------------------------------
# Main Visualisation Runner
# ---------------------------------------------------------------------
def run_visualisations(df: pd.DataFrame) -> None:
    """Generate and save all project visualisations."""
    print("\n" + "=" * 100)
    print("STEP 4: Generating Visualisations")
    print("=" * 100)

    figures_dir = Path(__file__).parent.parent / "data" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_correlation(df, figures_dir)
    plot_trends(df, figures_dir)
    plot_gdp_relationship(df, figures_dir)
    plot_fertility_relationship(df, figures_dir)
    plot_distribution(df, figures_dir)

    plt.close("all")

    print("✓ All visualisations generated successfully!\n")
