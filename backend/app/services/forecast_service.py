import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
AGEING_RATIO_PATH = os.path.join(BASE_DIR, "data/processed/ageing_ratio_per_region.csv")
MULTI_COMPARE_DIR = os.path.join(BASE_DIR, "output/multi_compare")
METRICS_PATH = os.path.join(MULTI_COMPARE_DIR, "prophet_arima_metrics.csv")


def _historical_series(region: str) -> pd.DataFrame:
    df = pd.read_csv(AGEING_RATIO_PATH)
    df = df[df["Country"] == region][["Year", "Percent65plus"]].copy()
    df.columns = ["year", "value"]
    df["type"] = "historical"
    df["value"] = df["value"].round(4)
    return df


def _read_forecast_csv(region: str, model: str) -> pd.DataFrame | None:
    path = os.path.join(MULTI_COMPARE_DIR, f"{region}_{model}_forecast.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["year"] = pd.to_datetime(df["ds"]).dt.year
    df = df[["year", "forecast"]].copy()
    df.columns = ["year", "value"]
    df["type"] = "forecast"
    df["value"] = df["value"].round(4)
    return df


def get_prophet_forecast(region: str):
    hist = _historical_series(region)
    fc = _read_forecast_csv(region, "prophet")
    combined = pd.concat([hist, fc], ignore_index=True) if fc is not None else hist
    return combined.to_dict(orient="records")


def get_arima_forecast(region: str):
    hist = _historical_series(region)
    fc = _read_forecast_csv(region, "arima")
    combined = pd.concat([hist, fc], ignore_index=True) if fc is not None else hist
    return combined.to_dict(orient="records")


def get_metrics():
    if not os.path.exists(METRICS_PATH):
        return []
    df = pd.read_csv(METRICS_PATH)
    # Normalise column names to snake_case
    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        "region": "region",
        "prophet_MAE": "prophet_mae",
        "prophet_RMSE": "prophet_rmse",
        "prophet_MAPE": "prophet_mape",
        "arima_MAE": "arima_mae",
        "arima_RMSE": "arima_rmse",
        "arima_MAPE": "arima_mape",
        "test_start_year": "test_start_year",
        "last_observed_year": "last_observed_year",
        "horizon_years": "horizon_years",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    for col in ["prophet_mae", "prophet_rmse", "prophet_mape", "arima_mae", "arima_rmse", "arima_mape"]:
        if col in df.columns:
            df[col] = df[col].round(4)
    return df.to_dict(orient="records")
