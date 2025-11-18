"""
Align, compare, evaluate, and plot ARIMA vs Prophet forecasts 
for the proportion of population aged 65+.

Dependencies:
- From src/arima_model.py:
    - load_arima_data(input_path) -> pd.DataFrame
    - train_arima_model(df, use_auto=True) -> (fitted_model, order_tuple)
    - forecast_arima(df, model_fit, n_years=...) -> pd.DataFrame

- Flexible column name handling:
  Historical data may have columns like y / y_ratio, 
  ARIMA output may have yhat or yhat_arima,
  Prophet files usually have yhat.
  Here, all are normalized before alignment.
"""

from __future__ import annotations

import os
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.arima_model import (
    load_arima_data,
    train_arima_model,
    forecast_arima,
)


# Utility functions: Column/data normalization

def _col_exists(df: pd.DataFrame, names: Iterable[str]) -> bool:
    """Check if any column in 'names' exists in df (case-insensitive)."""
    lower = {c.lower() for c in df.columns}
    return any((name.lower() in lower) for name in names)


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    """Return the first matching column name from 'names' (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in names:
        key = name.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _safe_to_datetime(ds: pd.Series) -> pd.Series:
    """Safely convert strings/integers to pandas datetime (default Jan 1 of the year)."""
    def _to_ts(x):
        try:
            # Integer year
            if isinstance(x, (int, np.integer)) and 1800 <= x <= 2200:
                return pd.Timestamp(year=int(x), month=1, day=1)
            # YYYYMMDD integer format
            if isinstance(x, (int, np.integer)) and 18000101 <= x <= 22001231:
                year = x // 10000
                return pd.Timestamp(year=year, month=1, day=1)
            # String year
            if isinstance(x, str) and x.isdigit() and len(x) == 4:
                return pd.Timestamp(year=int(x), month=1, day=1)
            # Fallback: normal parse
            return pd.to_datetime(x)
        except Exception:
            return pd.NaT

    out = ds.apply(_to_ts)
    return out.dt.to_period("Y").dt.to_timestamp("Y")


def _ensure_hist_columns(df_hist_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize historical data to ['ds', 'y'], with ds set to Jan 1 of each year.
    Compatible with possible column names:
      - Date: ['ds', 'date', 'year']
      - Target: ['y', 'y_ratio', 'ratio', 'value']
    """
    df = df_hist_raw.copy()

    # Date column
    ds_col = _first_existing(df, ["ds", "date", "year"])
    if ds_col is None:
        raise ValueError(f"Historical data missing date column (needs ds/date/year). Columns found: {list(df.columns)}")
    df.rename(columns={ds_col: "ds"}, inplace=True)
    df["ds"] = _safe_to_datetime(df["ds"])
    df = df.dropna(subset=["ds"])

    # Target column
    y_col = _first_existing(df, ["y", "y_ratio", "ratio", "value"])
    if y_col is None:
        raise ValueError(
            f"Historical data missing target column (needs y/y_ratio/ratio/value). Columns found: {list(df.columns)}"
        )
    df.rename(columns={y_col: "y"}, inplace=True)

    return df[["ds", "y"]].sort_values("ds").reset_index(drop=True)


def _normalize_prophet_df(prophet_df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize Prophet forecast to ['ds', 'yhat_prophet'] with ds set to Jan 1."""
    df = prophet_df_raw.copy()
    if not _col_exists(df, ["ds"]):
        raise ValueError(f"Prophet output missing ds column. Columns found: {list(df.columns)}")
    if not _col_exists(df, ["yhat"]):
        raise ValueError(f"Prophet output missing yhat column. Columns found: {list(df.columns)}")

    real_ds = _first_existing(df, ["ds"])
    real_yhat = _first_existing(df, ["yhat"])

    df.rename(columns={real_ds: "ds", real_yhat: "yhat_prophet"}, inplace=True)
    df["ds"] = _safe_to_datetime(df["ds"])
    df = df.dropna(subset=["ds"])

    return df[["ds", "yhat_prophet"]].sort_values("ds").reset_index(drop=True)


def _normalize_arima_df(arima_df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ARIMA forecast to ['ds', 'yhat_arima'] with ds set to Jan 1.
    Compatible with both yhat and yhat_arima column names.
    """
    df = arima_df_raw.copy()
    if not _col_exists(df, ["ds"]):
        raise ValueError(f"ARIMA output missing ds column. Columns found: {list(df.columns)}")

    yhat_col = _first_existing(df, ["yhat_arima", "yhat"])
    if yhat_col is None:
        raise ValueError(f"ARIMA output missing yhat or yhat_arima column. Columns found: {list(df.columns)}")

    df.rename(columns={yhat_col: "yhat_arima"}, inplace=True)
    df["ds"] = _safe_to_datetime(df["ds"])
    df = df.dropna(subset=["ds"])

    return df[["ds", "yhat_arima"]].sort_values("ds").reset_index(drop=True)


def _year_range(df: pd.DataFrame, ds_col: str = "ds") -> Tuple[int, int]:
    """Return (min_year, max_year) of the DataFrame."""
    if df.empty:
        return (None, None)  # type: ignore
    years = df[ds_col].dt.year
    return int(years.min()), int(years.max())


def _inner_merge_on_year(p_df: pd.DataFrame, a_df: pd.DataFrame) -> pd.DataFrame:
    """
    First try inner merge on exact ds match.
    If mismatch (due to timezone/format), fallback to merge on year.
    """
    merged = p_df.merge(a_df, on="ds", how="inner")
    if not merged.empty:
        return merged

    # Fallback: merge on year
    p_tmp = p_df.copy()
    a_tmp = a_df.copy()
    p_tmp["Year"] = p_tmp["ds"].dt.year
    a_tmp["Year"] = a_tmp["ds"].dt.year
    merged = p_tmp.merge(a_tmp, on="Year", how="inner").drop(columns=["Year"])
    if "ds_x" in merged.columns and "ds_y" in merged.columns:
        merged = merged.rename(columns={"ds_x": "ds"}).drop(columns=["ds_y"])
    return merged



# Main function: Train ARIMA, read Prophet, align, evaluate, plot

def forecast_65plus_arima(
    input_path,
    prophet_path,
    compare_img_path,
    n_years=47,
    hist_end_year=None,
    forecast_end_year=None
):
    """
    Align ARIMA and Prophet forecasts for 65+ population ratio,
    ensure same future period, evaluate differences, and plot results.
    """

    # Load historical data (ds, y)
    df = load_arima_data(input_path).copy()
    if "ds" not in df or "y" not in df:
        raise ValueError(f"Input {input_path} must contain ['ds','y'], found: {list(df.columns)}")
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])

    # Limit historical range
    hist_year_max = df["ds"].dt.year.max()
    _hist_end_year = hist_end_year if hist_end_year is not None else hist_year_max
    df_hist = df[df["ds"].dt.year <= _hist_end_year].copy()
    if df_hist.empty:
        raise ValueError(f"No historical data after limiting to hist_end_year={_hist_end_year}.")

    # Train ARIMA
    model_fit, order = train_arima_model(df_hist[["ds", "y"]], use_auto=True)

    # Load Prophet output
    if not os.path.exists(prophet_path):
        raise FileNotFoundError(f"Prophet forecast file not found: {prophet_path}")
    prophet_df = pd.read_csv(prophet_path, parse_dates=["ds"])
    if "yhat" not in prophet_df:
        raise ValueError(f"{prophet_path} must contain ['ds','yhat'], found: {list(prophet_df.columns)}")

    prophet_df = prophet_df[["ds", "yhat"]].copy()
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    # Define consistent future range
    prophet_future_year_max = prophet_df["ds"].dt.year.max()
    _forecast_end_year = forecast_end_year if forecast_end_year is not None else prophet_future_year_max

    start_future_year = _hist_end_year + 1
    if start_future_year > _forecast_end_year:
        raise ValueError(f"Empty future range: start={start_future_year} > end={_forecast_end_year}")

    future_years = list(range(start_future_year, _forecast_end_year + 1))
    future_ds = pd.to_datetime([f"{y}-01-01" for y in future_years])

    # Filter Prophet future period
    prophet_future = prophet_df[prophet_df["ds"].isin(future_ds)].copy()
    if len(prophet_future) != len(future_ds):
        missing = set(future_ds) - set(prophet_future["ds"])
        print(f"Prophet missing these future dates (will be dropped): {sorted(list(missing))}")

    # Forecast ARIMA with same horizon
    arima_h = len(future_years)
    arima_df = forecast_arima(df_hist[["ds", "y"]], model_fit, n_years=arima_h).copy()
    if "yhat" not in arima_df.columns:
        cand = [c for c in arima_df.columns if c.lower() in ("yhat", "y_pred", "forecast", "pred")]
        if len(cand) == 1:
            arima_df = arima_df.rename(columns={cand[0]: "yhat"})
        else:
            raise ValueError(f"ARIMA output missing 'yhat'. Columns found: {list(arima_df.columns)}")

    arima_df = arima_df.head(len(future_ds)).copy()
    arima_df["ds"] = future_ds
    arima_df = arima_df[["ds", "yhat"]].copy()

    # Merge aligned forecasts
    merged = pd.merge(
        prophet_future.rename(columns={"yhat": "yhat_prophet"}),
        arima_df.rename(columns={"yhat": "yhat_arima"}),
        on="ds",
        how="inner"
    ).sort_values("ds")

    if merged.empty:
        raise ValueError("Merged result is empty. Ensure ds values are yearly (Jan 1).")

    # Log ranges
    hist_min = df_hist["ds"].dt.year.min()
    hist_max = df_hist["ds"].dt.year.max()
    pr_min = merged["ds"].dt.year.min()
    pr_max = merged["ds"].dt.year.max()
    print(f"HIST range = {hist_min} → {_hist_end_year}")
    print(f"Aligned future range = {pr_min} → {pr_max}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(merged["ds"], merged["yhat_prophet"], label="Prophet", linewidth=2)
    plt.plot(merged["ds"], merged["yhat_arima"], label="ARIMA", linestyle="--", linewidth=2)
    plt.xlabel("Year")
    plt.ylabel("65+ Population Ratio")
    plt.title("England 65+ Forecast: Prophet vs ARIMA (Aligned)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(compare_img_path), exist_ok=True)
    plt.savefig(compare_img_path)
    plt.close()
    print(f"ARIMA vs Prophet comparison plot saved: {compare_img_path}")

    # Evaluation metrics
    mae = mean_absolute_error(merged["yhat_prophet"], merged["yhat_arima"])
    rmse = sqrt(mean_squared_error(merged["yhat_prophet"], merged["yhat_arima"]))
    mape = np.mean(
        np.abs((merged["yhat_prophet"] - merged["yhat_arima"]) / merged["yhat_prophet"])
    ) * 100

    print("Prophet vs ARIMA error evaluation:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return mae, rmse, mape