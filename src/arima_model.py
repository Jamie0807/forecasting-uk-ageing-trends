import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from math import sqrt

try:
    import pmdarima as pm
except Exception as e:
    raise ImportError(
        "pmdarima is not installed. Please install it first: pip install pmdarima"
    ) from e


# Utility Functions 
def _rmse(y_true, y_pred) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _yearly_future_index(last_year: int, n_years: int) -> pd.DatetimeIndex:
    """
    Generate a DatetimeIndex for the next `n_years` years starting from `last_year` + 1.
    """
    years = np.arange(last_year + 1, last_year + 1 + n_years, dtype=int)
    return pd.to_datetime(pd.Series(years).astype(str) + "-01-01")


# Data I/O
def load_arima_data(csv_path: str,
                    date_col: str = "ds",
                    y_col: str = "y") -> pd.DataFrame:
    """
    Load a time series CSV for ARIMA.

    Requirements:
    - Must contain [date_col, y_col] columns.
    - date_col must be parseable as a date.
    - y_col is the target variable (numeric).

    Returns:
    DataFrame with columns ['ds', 'y'] sorted by date.
    """
    df = pd.read_csv(csv_path)
    if date_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {date_col}, {y_col}. Current columns: {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df[[date_col, y_col]].rename(columns={date_col: "ds", y_col: "y"})


# Training & Forecasting
def train_arima_model(
    df: pd.DataFrame,
    use_auto: bool = True,
    order: Optional[Tuple[int, int, int]] = None,
    prefer_trend: bool = True,
    val_years: int = 3,
) -> Tuple[pm.ARIMA, Tuple[int, int, int]]:
    """
    Train an ARIMA model.

    Parameters
    ----------
    df : DataFrame
        Must contain columns ['ds', 'y'], sorted by year.
    use_auto : bool
        If True, use pmdarima.auto_arima for parameter search.
    order : tuple(int, int, int) or None
        If provided, overrides `use_auto` and directly fits with this order.
    prefer_trend : bool
        If True and auto_arima finds (0,0,0) or no order, try trend-based candidates
        and select the best based on validation RMSE.
    val_years : int
        Number of years at the end of the series to use for validation.

    Returns
    -------
    model : pm.ARIMA
        Trained ARIMA model.
    chosen_order : tuple
        Final (p, d, q) order used.
    """
    if not {"ds", "y"} <= set(df.columns):
        raise ValueError("df must contain columns ['ds', 'y']")

    y_full = df["y"].values.astype(float)

    # Split into training and validation sets
    if len(df) <= val_years + 3:
        # If too few data points, train on all available data
        y_train, y_valid = y_full, np.array([])
    else:
        y_train, y_valid = y_full[:-val_years], y_full[-val_years:]

    chosen_order: Optional[Tuple[int, int, int]] = None
    model: Optional[pm.ARIMA] = None

    # 1) Use explicitly provided order
    if order is not None:
        chosen_order = tuple(order)
        model = pm.ARIMA(order=chosen_order, seasonal=False)
        model.fit(y_train)
        return model, chosen_order

    # 2) Auto parameter search
    if use_auto:
        auto_model = pm.auto_arima(
            y_train,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            start_p=1, start_q=1, start_d=1,   # Prefer trend inclusion
            max_p=5, max_q=5, max_d=2,
            information_criterion="aic",
            error_action="ignore",
        )
        chosen_order = auto_model.order  # type: ignore[attr-defined]
        model = auto_model

    # 3) If auto_arima finds (0,0,0) or no order, try trend-based candidates
    if prefer_trend and (chosen_order == (0, 0, 0) or chosen_order is None):
        candidates: List[Tuple[int, int, int]] = [
            (1, 1, 0), (0, 1, 1), (1, 1, 1),
            (2, 1, 1), (1, 1, 2), (2, 1, 2),
        ]
        best_rmse = np.inf
        best_model = None
        best_order = None

        for od in candidates:
            try:
                m = pm.ARIMA(order=od, seasonal=False)
                m.fit(y_train)
                if y_valid.size > 0:
                    y_pred = m.predict(n_periods=len(y_valid))
                    score = _rmse(y_valid, y_pred)
                else:
                    # If no validation set, use AIC as a proxy
                    score = float(m.aic()) if hasattr(m, "aic") else 1e9
            except Exception:
                continue

            if score < best_rmse:
                best_rmse = score
                best_model = m
                best_order = od

        # If candidate beats auto_arima or auto_arima failed, use candidate
        if best_model is not None:
            model = best_model
            chosen_order = best_order

    # 4) Fallback to a safe default (1,1,1)
    if model is None or chosen_order is None:
        chosen_order = (1, 1, 1)
        model = pm.ARIMA(order=chosen_order, seasonal=False)
        model.fit(y_train)

    print(f"Chosen ARIMA order: {chosen_order}")
    return model, chosen_order


def forecast_arima(
    df: pd.DataFrame,
    model: pm.ARIMA,
    n_years: int = 47
) -> pd.DataFrame:
    """
    Forecast the next `n_years` using a trained ARIMA model.

    Parameters
    ----------
    df : DataFrame
        Must contain columns ['ds', 'y'].
    model : pm.ARIMA
        Trained ARIMA model.
    n_years : int
        Number of years to forecast into the future.

    Returns
    -------
    DataFrame
        Columns: ['ds', 'yhat'] where ds is the forecast year and yhat is the predicted value.
    """
    if not {"ds", "y"} <= set(df.columns):
        raise ValueError("df must contain columns ['ds', 'y']")

    last_year = df["ds"].dt.year.max()
    future_idx = _yearly_future_index(last_year, n_years)
    y_pred = model.predict(n_periods=n_years)

    out = pd.DataFrame({
        "ds": future_idx,
        "yhat": y_pred
    })
    return out