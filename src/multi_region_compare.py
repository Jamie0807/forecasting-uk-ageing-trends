"""
Multi-region comparison module for 65+ share (ageing ratio) forecasting:
Prophet vs ARIMA (revised version)

Key improvements:
- More robust parsing of Age in long tables ("65-69" / "70–74" / "70 to 74" / "90+" / "65")
- More tolerant Sex filtering (Total / Persons / All persons / All / All sexes)
- All other interfaces and behaviors remain consistent
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Console logging with INFO by default
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)



# Optional dependency: Prophet (either 'prophet' or 'fbprophet')

_PROPHET_AVAILABLE = False
try:
    from prophet import Prophet  # type: ignore
    _PROPHET_AVAILABLE = True
except Exception:
    try:
        from fbprophet import Prophet  # type: ignore
        _PROPHET_AVAILABLE = True
    except Exception:
        pass


# Optional dependency: pmdarima (auto ARIMA order selection)

_ARIMA_AVAILABLE = False
try:
    import pmdarima as pm  # type: ignore
    _ARIMA_AVAILABLE = True
except Exception:
    pass



# Internal utility helpers


def _to_year_start(v) -> pd.Timestamp:
    """Convert a year-like value to the timestamp of Jan 1 of that year."""
    try:
        y = int(float(v))
        return pd.Timestamp(f"{y}-01-01")
    except Exception:
        return pd.to_datetime(v)


def _mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (as %)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Unified metrics: MAE / RMSE / MAPE."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"MAE": mae, "RMSE": rmse, "MAPE": _mape(y_true, y_pred)}


def _parse_age_lower_bound_series(age_series: pd.Series) -> pd.Series:
    """
    Parse the lower bound from an age string into a float.

    Supports:
      - Ranges: "65-69", "70–74", "70 — 74", "70 to 74"
      - Open-ended: "90+", "85+"
      - Single number: "65"
    Returns NaN if parsing fails.
    """
    s = age_series.astype(str).str.strip().str.lower()

    # Normalize separators
    s_norm = (
        s.str.replace("—", "-", regex=False)
         .str.replace("–", "-", regex=False)
         .str.replace("to", "-", regex=False)
    )

    # Handle 90+, 85+, etc.
    plus_mask = s_norm.str.endswith("+")
    lb = pd.Series(np.nan, index=s.index, dtype="float64")
    lb[plus_mask] = pd.to_numeric(s_norm[plus_mask].str[:-1], errors="coerce")

    # Handle ranges like 65-69
    rng_mask = s_norm.str.contains("-")
    left = s_norm[rng_mask].str.split("-", n=1, expand=True)[0].str.strip()
    lb[rng_mask] = pd.to_numeric(left, errors="coerce")

    # Handle plain numbers "65"
    rem_mask = ~(plus_mask | rng_mask)
    lb[rem_mask] = pd.to_numeric(s_norm[rem_mask], errors="coerce")

    return lb


def _plot_one_region(
    *,
    region: str,
    df_hist: pd.DataFrame,              # columns: ds, value (full history)
    df_test: pd.DataFrame,              # columns: ds, value (test ground truth; may be empty)
    df_prophet_test: Optional[pd.DataFrame],   # columns: ds, yhat (test-period forecast)
    df_prophet_future: Optional[pd.DataFrame], # columns: ds, yhat (future forecast)
    df_arima_test: Optional[pd.DataFrame],     # columns: ds, yhat (test-period forecast)
    df_arima_future: Optional[pd.DataFrame],   # columns: ds, yhat (future forecast)
    m_prophet: Optional[Dict[str, float]],
    m_arima: Optional[Dict[str, float]],
    out_png: str,
    y_unit: str,
) -> None:
    """Plot and save the Prophet vs ARIMA comparison chart for one region."""
    plt.figure(figsize=(10, 6))

    # Historical & test ground truth
    plt.plot(df_hist["ds"], df_hist["value"], label="Historical")
    if not df_test.empty:
        plt.plot(df_test["ds"], df_test["value"], label="Test (Ground Truth)")

    # Prophet: test (dashed) and future (dotted)
    if df_prophet_test is not None and not df_prophet_test.empty:
        plt.plot(df_prophet_test["ds"], df_prophet_test["yhat"], label="Prophet (test)", linestyle="--")
    if df_prophet_future is not None and not df_prophet_future.empty:
        plt.plot(df_prophet_future["ds"], df_prophet_future["yhat"], label="Prophet (future)", linestyle=":")

    # ARIMA: test (dashed) and future (dotted)
    if df_arima_test is not None and not df_arima_test.empty:
        plt.plot(df_arima_test["ds"], df_arima_test["yhat"], label="ARIMA (test)", linestyle="--")
    if df_arima_future is not None and not df_arima_future.empty:
        plt.plot(df_arima_future["ds"], df_arima_future["yhat"], label="ARIMA (future)", linestyle=":")

    plt.title(f"{region} — 65+ share ({y_unit}) — Prophet vs ARIMA")
    plt.xlabel("Year")
    plt.ylabel(f"65+ share ({y_unit})")
    plt.legend()

    # Metrics box at top-right
    ymax = plt.ylim()[1]
    xmin = df_hist["ds"].min()

    def _fmt(m: Optional[Dict[str, float]]) -> str:
        return (f"MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}, MAPE={m['MAPE']:.2f}%"
                if m is not None else "N/A")

    plt.text(xmin, ymax, f"Prophet: {_fmt(m_prophet)}\nARIMA: {_fmt(m_arima)}", va="top")

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()



# Public API


def run_multi_region_prophet_arima_compare(
    *,
    input_csv: str,
    outdir: str,
    regions: List[str],
    region_col: str = "region",
    date_col: str = "year",
    value_col: Optional[str] = None,
    test_year_start: Optional[int] = 2015,
    horizon: int = 30,
    is_percent: bool = True,
    save_forecast_csv: bool = True,
) -> str:
    """
    Run Prophet vs ARIMA comparison across multiple regions and write out plots/metrics.

    Parameters
    ----------
    input_csv : str
        Input CSV path. Either provide a tidy table with columns [region_col, date_col, value_col],
        or a long table with [Age, Sex, Population, region_col, date_col] for auto-aggregation.
    outdir : str
        Output directory for plots and metrics CSV.
    regions : list[str]
        List of region names to process.
    region_col : str, default "region"
        Column name for region.
    date_col : str, default "year"
        Column name for year/date.
    value_col : str or None
        Column containing the 65+ share / ageing ratio. If None, the function will try to
        auto-aggregate from long tables or infer from common column names.
    test_year_start : int or None, default 2015
        If provided, years >= this value form the test set; else, use last 20% as test set.
    horizon : int, default 30
        Number of future years to forecast.
    is_percent : bool, default True
        Whether the value is in 0-100 (%) scale; if True, values are normalized to 0-1 for modeling.
    save_forecast_csv : bool, default True
        If True, save per-region future forecast CSVs.

    Returns
    -------
    metrics_csv : str
        Path to the saved metrics CSV file.
    """
    #  Read & normalize column names 
    if not os.path.exists(input_csv):
        raise FileNotFoundError(input_csv)
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(input_csv)
    logger.info("Prophet available=%s; pmdarima available=%s", _PROPHET_AVAILABLE, _ARIMA_AVAILABLE)
    df.columns = [str(c).strip() for c in df.columns]

    # Auto-aggregate from long table if needed 
    cols_lc = {c.lower(): c for c in df.columns}
    required_for_asp = {"age", "sex", "population", region_col.lower(), date_col.lower()}

    if (value_col is None or value_col not in df.columns) and required_for_asp.issubset(set(cols_lc.keys())):
        c_age = cols_lc["age"]
        c_sex = cols_lc["sex"]
        c_pop = cols_lc["population"]
        c_reg = cols_lc[region_col.lower()]
        c_year = cols_lc[date_col.lower()]

        tmp = df.copy()

        # Relaxed filtering for Sex = Total
        SEX_TOTAL_SET = {"total", "persons", "all persons", "all", "all sexes"}
        sex_series = tmp[c_sex].astype(str).str.lower().str.strip()
        tmp = tmp[sex_series.isin(SEX_TOTAL_SET)]

        # Parse lower bound of age
        tmp["_age_lb"] = _parse_age_lower_bound_series(tmp[c_age])

        # Keep valid ages
        tmp = tmp[~tmp["_age_lb"].isna()]

        grp = [c_reg, c_year]
        total = (
            tmp.groupby(grp, as_index=False)[c_pop]
              .sum()
              .rename(columns={c_pop: "PopTotal"})
        )
        over65 = (
            tmp[tmp["_age_lb"] >= 65]
              .groupby(grp, as_index=False)[c_pop]
              .sum()
              .rename(columns={c_pop: "Pop65plus"})
        )

        agg = pd.merge(total, over65, on=grp, how="left").fillna({"Pop65plus": 0})
        agg["Percent65plus"] = agg["Pop65plus"] / agg["PopTotal"] * 100.0

        # Replace df with the aggregated one and set unified column names
        df = agg.rename(columns={c_reg: region_col, c_year: date_col})
        value_col = "Percent65plus"

        logger.info(
            "Aggregated Percent65plus from long table. Rows: total=%d, regions=%d, years=%d",
            len(df), df[region_col].nunique(), df[date_col].nunique()
        )

    #  Auto-detect value_col if still None (case-insensitive) 
    if value_col is None:
        cols_lc = {c.lower(): c for c in df.columns}
        candidates = [
            "share_65plus",
            "ageing_ratio_65plus",
            "ageing_ratio",
            "ratio_65plus",
            "percent65plus",
            "percent_65plus",
        ]
        for lc in candidates:
            if lc in cols_lc:
                value_col = cols_lc[lc]
                break

    #  Basic required columns check 
    if value_col is None or value_col not in df.columns:
        logger.error("Available columns: %s", list(df.columns))
        raise ValueError("Please provide `value_col` or ensure one of the common ageing-ratio column names exists.")
    if region_col not in df.columns or date_col not in df.columns:
        raise ValueError(f"CSV missing required columns: {region_col}, {date_col}")

    # Standardize internal column names & scale values 
    df = df.rename(columns={region_col: "region", date_col: "date", value_col: "value"}).copy()
    df["ds"] = df["date"].apply(_to_year_start)

    # Prophet/ARIMA use y; if percentage (0-100), convert to 0-1
    df["y"] = (df["value"] / 100.0) if is_percent else df["value"].astype(float)

    # Simple sanity check
    if df["y"].notna().sum() == 0 or (df["y"].fillna(0).abs().sum() == 0):
        logger.warning("All values are zero or NaN after aggregation — please check the input schema.")
    else:
        logger.info(
            "Value range after prep: min=%.4f, max=%.4f (unit=%s)",
            float(df["y"].min()), float(df["y"].max()), "percent(0-1)" if is_percent else "ratio"
        )

    metrics_rows: List[Dict[str, object]] = []

    # ---------- Model per region ----------
    for region in regions:
        sub = df[df["region"] == region].sort_values("ds").copy()
        if sub.empty:
            logger.warning("Region '%s' not found; skip.", region)
            continue

        # Train/test split
        if test_year_start is not None:
            mask_test = sub["ds"].dt.year >= int(test_year_start)
            train = sub.loc[~mask_test]
            test = sub.loc[mask_test]
        else:
            n = len(sub)
            k = max(1, int(round(n * 0.2)))  # last 20% as test
            train = sub.iloc[:-k]
            test = sub.iloc[-k:]

        # Prophet (logistic growth + cap)

        prophet_metrics: Optional[Dict[str, float]] = None
        prophet_test = pd.DataFrame()
        prophet_future = pd.DataFrame()
        if _PROPHET_AVAILABLE and len(train) > 2:
            try:
                # Auto cap in 0-1 space
                ymax = float(sub["y"].max())
                cap = min(0.45, max(ymax * 1.2, ymax + 0.05))  # cap no higher than 0.45
                floor = 0.0

                # Train with cap/floor
                train_p = train[["ds", "y"]].copy()
                train_p["cap"] = cap
                train_p["floor"] = floor

                m = Prophet(
                    growth="logistic",
                    changepoint_prior_scale=0.2,   # a bit flexible; tune 0.05~0.5 if needed
                    changepoint_range=0.9
                )
                m.fit(train_p)

                # Test-period prediction + metrics
                if not test.empty:
                    test_p = test[["ds"]].copy()
                    test_p["cap"] = cap
                    test_p["floor"] = floor
                    prophet_test = m.predict(test_p)[["ds", "yhat"]]
                    prophet_metrics = _metrics(test["y"].values, prophet_test["yhat"].values)

                # Future predictions (yearly steps)
                last_ds = sub["ds"].max()
                future_idx = pd.date_range(
                    start=last_ds + pd.offsets.YearBegin(1),
                    periods=horizon,
                    freq="YS",
                )
                if len(future_idx) > 0:
                    fut = pd.DataFrame({"ds": future_idx})
                    fut["cap"] = cap
                    fut["floor"] = floor
                    prophet_future = m.predict(fut)[["ds", "yhat"]]

            except Exception as e:
                logger.warning("Prophet failed for %s: %s", region, e)
        else:
            if not _PROPHET_AVAILABLE:
                logger.info("Prophet not available; skipping Prophet for %s.", region)

        # ARIMA (with trend/drift). Fallback to statsmodels if pmdarima is unavailable.
        arima_metrics: Optional[Dict[str, float]] = None
        arima_test = pd.DataFrame()
        arima_future = pd.DataFrame()
        try:
            if _ARIMA_AVAILABLE and len(train) > 5:
                model = pm.auto_arima(
                    train["y"].values,
                    start_p=0, start_q=0, max_p=3, max_q=3,
                    start_d=0, max_d=2,
                    seasonal=False,                # yearly demographic series: no seasonal component
                    stationary=False,
                    with_intercept=True,           # allow intercept (drift when d > 0)
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    trace=False,
                    information_criterion="aicc",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

                if not test.empty:
                    yhat_test = model.predict(n_periods=len(test))
                    arima_test = pd.DataFrame({"ds": test["ds"].values, "yhat": yhat_test})
                    arima_metrics = _metrics(test["y"].values, yhat_test)

                if horizon > 0:
                    yhat_future = model.predict(n_periods=horizon)
                    last_ds = sub["ds"].max()
                    future_idx = pd.date_range(
                        start=last_ds + pd.offsets.YearBegin(1),
                        periods=horizon,
                        freq="YS",
                    )
                    arima_future = pd.DataFrame({"ds": future_idx, "yhat": yhat_future})

            else:
                # Fallback: statsmodels SARIMAX with a simple linear trend
                from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
                if len(train) > 5:
                    sm_model = SARIMAX(
                        train["y"].values,
                        order=(1, 1, 1),
                        trend="t",
                        seasonal_order=(0, 0, 0, 0)
                    ).fit(disp=False)

                    if not test.empty:
                        yhat_test = sm_model.forecast(steps=len(test))
                        arima_test = pd.DataFrame({"ds": test["ds"].values, "yhat": yhat_test})
                        arima_metrics = _metrics(test["y"].values, yhat_test)

                    if horizon > 0:
                        yhat_future = sm_model.forecast(steps=horizon)
                        last_ds = sub["ds"].max()
                        future_idx = pd.date_range(
                            start=last_ds + pd.offsets.YearBegin(1),
                            periods=horizon,
                            freq="YS",
                        )
                        arima_future = pd.DataFrame({"ds": future_idx, "yhat": yhat_future})
                else:
                    logger.info("Not enough data points for ARIMA in %s.", region)
        except Exception as e:
            logger.warning("ARIMA failed for %s: %s", region, e)

        # Metrics row & visualization
        row = {
            "region": region,
            "prophet_MAE": prophet_metrics["MAE"] if prophet_metrics else np.nan,
            "prophet_RMSE": prophet_metrics["RMSE"] if prophet_metrics else np.nan,
            "prophet_MAPE": prophet_metrics["MAPE"] if prophet_metrics else np.nan,
            "arima_MAE": arima_metrics["MAE"] if arima_metrics else np.nan,
            "arima_RMSE": arima_metrics["RMSE"] if arima_metrics else np.nan,
            "arima_MAPE": arima_metrics["MAPE"] if arima_metrics else np.nan,
            "test_start_year": int(test["ds"].dt.year.min()) if not test.empty else None,
            "last_observed_year": int(sub["ds"].dt.year.max()),
            "horizon_years": horizon,
            "value_unit": "percent" if is_percent else "ratio",
        }
        metrics_rows.append(row)

        # Convert y (0-1) back to display unit
        hist_plot = sub[["ds", "y"]].copy()
        hist_plot["value"] = hist_plot["y"] * (100.0 if is_percent else 1.0)
        test_plot = test[["ds", "y"]].copy()
        test_plot["value"] = test_plot["y"] * (100.0 if is_percent else 1.0)

        # Same for predicted curves
        def _to_unit(df_pred):
            if df_pred is None or df_pred.empty:
                return None
            out = df_pred.copy()
            out["yhat"] = out["yhat"] * (100.0 if is_percent else 1.0)
            return out

        p_test_plot   = _to_unit(prophet_test)
        p_future_plot = _to_unit(prophet_future)
        a_test_plot   = _to_unit(arima_test)
        a_future_plot = _to_unit(arima_future)

        out_png = os.path.join(outdir, f"prophet_vs_arima_{region.lower().replace(' ', '_')}.png")
        _plot_one_region(
            region=region,
            df_hist=hist_plot[["ds", "value"]],
            df_test=test_plot[["ds", "value"]],
            df_prophet_test=p_test_plot,
            df_prophet_future=p_future_plot,
            df_arima_test=a_test_plot,
            df_arima_future=a_future_plot,
            m_prophet=prophet_metrics,
            m_arima=arima_metrics,
            out_png=out_png,
            y_unit=("%" if is_percent else "ratio"),
        )

        # Optional: export per-region future forecast CSVs (columns: ds, forecast)
        if save_forecast_csv:
            if p_future_plot is not None and not np.isnan(row["prophet_MAE"]):
                p_out = os.path.join(outdir, f"{region}_prophet_forecast.csv")
                p_future_plot.rename(columns={"yhat": "forecast"}).to_csv(p_out, index=False)
            if a_future_plot is not None and not np.isnan(row["arima_MAE"]):
                a_out = os.path.join(outdir, f"{region}_arima_forecast.csv")
                a_future_plot.rename(columns={"yhat": "forecast"}).to_csv(a_out, index=False)

    #  Write metrics summary 
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(outdir, "prophet_arima_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    logger.info("Multi-region metrics saved: %s", metrics_csv)
    return metrics_csv