"""
Main entry point for the forecasting UK ageing trends project.
"""

import os
from pathlib import Path

# Project Configuration

CONFIG = {
    # # File paths
    
    "raw_data": {
        "population": "data/raw/mid_year_population_estimates_uk.xlsx",
        "projected": "data/raw/SNPP18dt2.xlsx",
        "scotland": "data/raw/scppvsumpop20.xls",
        "wales": "data/raw/wappvsumpop20.xls",
        "england": "data/raw/enppvsumpop20.xls",
        "uk": "data/raw/ukppvsumpop20.xls"
    },
    "processed_data": {
        "cleaned_pop": "data/processed/cleaned_population_long.csv",
        "proj_pop": "data/processed/projected_population_long.csv",
        "scotland": "data/processed/scotland_clean.csv",
        "wales": "data/processed/wales_clean.csv",
        "england": "data/processed/england_clean.csv",
        "uk": "data/processed/uk_clean.csv",
        "merged_all": "data/processed/uk_population_projection_all.csv",
        "ageing_ratio": "data/processed/ageing_ratio_per_region.csv",
        "cluster_input": "data/processed/ageing_cluster_input.csv"
    },
    "output": {
        "trend": "output/ageing_trend_65plus_fixed.png",
        "prophet_england": "output/prophet_65plus_england.png",
        "prophet_all": "output/prophet_65plus_all_regions.png",
        "england_ts": "output/england_timeseries.csv",
        "england_forecast": "output/england_forecast.csv",
        "england_plot": "output/england_forecast.png",
        "compare_england": "output/compare_arima_prophet_england.png",
        "multi_compare_dir": "output/multi_compare",
        "clusters": "output/ageing_clusters.png"
    },
    # Model parameters
    
    "regions": ["England", "Wales", "Scotland"],
    "end_year": 2070,
    "test_year_start": 2030,
    "horizon": 30,
    "n_clusters": 3,
    "random_state": 42
}

# Data Preprocessing

from src.preprocess import clean_population_data
from src.preprocess_projections import clean_projected_population_data
from src.preprocess_scotland import clean_scotland_projection_xls
from src.preprocess_wales import clean_wales_projection_xls
from src.preprocess_england import clean_england_projection_xls
from src.preprocess_uk import clean_uk_projection_xls
from src.merge_projection_data import merge_population_data

# Visualization & Modeling

from src.plot_ageing import plot_65plus_trend
from src.model_prophet import forecast_65plus_prophet, forecast_all_regions
from src.generate_england_timeseries import generate_england_timeseries
from src.forecast_export import run_forecast_pipeline
from src.plot_forecast_england import plot_england_forecast
from src.model_arima import forecast_65plus_arima
from src.cluster_analysis import cluster_ageing_trends
from src.generate_cluster_input import generate_cluster_input
from src.multi_region_compare import run_multi_region_prophet_arima_compare


# Utility Functions

def file_exists(path):
    return Path(path).is_file()

def log(msg):
    print(f"[INFO] {msg}")

# Pipeline Steps
def step_1_preprocess():
    """Step 1: Clean and preprocess raw population data."""
    log("Cleaning and preprocessing raw population data...")
    if not file_exists(CONFIG["processed_data"]["cleaned_pop"]):
        clean_population_data(CONFIG["raw_data"]["population"], CONFIG["processed_data"]["cleaned_pop"])
    if not file_exists(CONFIG["processed_data"]["proj_pop"]):
        clean_projected_population_data(CONFIG["raw_data"]["projected"], CONFIG["processed_data"]["proj_pop"])
    if not file_exists(CONFIG["processed_data"]["scotland"]):
        clean_scotland_projection_xls(CONFIG["raw_data"]["scotland"], CONFIG["processed_data"]["scotland"])
    if not file_exists(CONFIG["processed_data"]["wales"]):
        clean_wales_projection_xls(CONFIG["raw_data"]["wales"], CONFIG["processed_data"]["wales"])
    if not file_exists(CONFIG["processed_data"]["england"]):
        clean_england_projection_xls(CONFIG["raw_data"]["england"], CONFIG["processed_data"]["england"])
    if not file_exists(CONFIG["processed_data"]["uk"]):
        clean_uk_projection_xls(CONFIG["raw_data"]["uk"], CONFIG["processed_data"]["uk"])


def step_2_merge():
    """Step 2: Merge population projections for all regions."""
    log("Step 2: Merging population projection data...")
    merge_population_data(
        CONFIG["processed_data"]["england"],
        CONFIG["processed_data"]["wales"],
        CONFIG["processed_data"]["scotland"],
        CONFIG["processed_data"]["merged_all"]
    )


def step_3_trend_and_cluster_input():
    """Step 3: Plot ageing trend and prepare clustering input data."""
    log("Step 3: Plotting ageing trend and preparing cluster input data...")
    plot_65plus_trend(
        CONFIG["processed_data"]["merged_all"],
        CONFIG["output"]["trend"],
        CONFIG["processed_data"]["ageing_ratio"]
    )
    generate_cluster_input(CONFIG["processed_data"]["ageing_ratio"], CONFIG["processed_data"]["cluster_input"])


def step_4_prophet():
    """Step 4: Forecast using Prophet (single region & multi-region)."""
    log("Step 4: Prophet forecasting for single and multiple regions...")
    forecast_65plus_prophet(
        CONFIG["processed_data"]["merged_all"],
        CONFIG["output"]["prophet_england"],
        region="England",
        end_year=CONFIG["end_year"]
    )
    forecast_all_regions(
        CONFIG["processed_data"]["merged_all"],
        CONFIG["output"]["prophet_all"],
        CONFIG["regions"],
        end_year=CONFIG["end_year"]
    )


def step_5_england_timeseries():
    """Step 5: Generate custom time series for England and forecast."""
    log("Step 5: Generating England time series and forecasting...")
    generate_england_timeseries(CONFIG["processed_data"]["england"], CONFIG["output"]["england_ts"])
    run_forecast_pipeline("England", CONFIG["output"]["england_ts"], CONFIG["output"]["england_forecast"])
    plot_england_forecast(CONFIG["output"]["england_forecast"], CONFIG["output"]["england_plot"])


def step_6_arima_compare_england():
    """Step 6: Compare ARIMA and Prophet performance for England."""
    log("Step 6: Comparing ARIMA vs Prophet for England...")
    forecast_65plus_arima(
        CONFIG["output"]["england_ts"],
        CONFIG["output"]["england_forecast"],
        CONFIG["output"]["compare_england"],
        hist_end_year=CONFIG["end_year"] - 1,
        forecast_end_year=CONFIG["end_year"] + CONFIG["horizon"]
    )

def step_7_multi_region_compare():
    """Step 7: Compare ARIMA and Prophet performance for multiple regions."""
    log("Step 7: Comparing ARIMA vs Prophet for multiple regions...")
    run_multi_region_prophet_arima_compare(
        input_csv=CONFIG["processed_data"]["merged_all"],
        outdir=CONFIG["output"]["multi_compare_dir"],
        regions=CONFIG["regions"],
        region_col="Country",
        date_col="Year",
        value_col=None,  # None → Automatically aggregate Percent65plus
        test_year_start=CONFIG["test_year_start"],
        horizon=CONFIG["horizon"],
        is_percent=True,
        save_forecast_csv=True
    )

def step_8_cluster_analysis():
    """Step 8: Perform clustering analysis on ageing trends."""
    log("Step 8: Performing clustering analysis...")
    cluster_ageing_trends(CONFIG["processed_data"]["ageing_ratio"], CONFIG["output"]["clusters"], CONFIG["n_clusters"])

# Main Execution

def main():
    log("Starting UK ageing trends forecasting pipeline...")

    step_1_preprocess()
    step_2_merge()
    step_3_trend_and_cluster_input()
    step_4_prophet()
    step_5_england_timeseries()
    step_6_arima_compare_england()
    step_7_multi_region_compare()
    step_8_cluster_analysis()

    log("All tasks completed!")


if __name__ == "__main__":
    main()