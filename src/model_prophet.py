import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.plot_comparison import plot_forecast_comparison

# Smoothed Prophet: Disable seasonality + tighten changepoints + logistic growth + rolling average smoothing
def build_prophet_forecast(df_model, end_year,
                           cps=0.04, n_changepoints=8, roll_k=5,
                           use_logistic=True, cap_value=None, floor_value=0.0):
    """
    Build and run a Prophet model with optional logistic growth and smoothing.

    Parameters:
    ----------
    df_model : pd.DataFrame
        Two columns: ['ds', 'y'], where 'y' is the percentage (0-100) and 'ds' is datetime (annual).
    end_year : int
        The target year for forecasting (e.g., 2070).
    cps : float
        Changepoint prior scale, controls trend flexibility.
    n_changepoints : int
        Maximum number of potential changepoints.
    roll_k : int
        Rolling window size for smoothing predictions.
    use_logistic : bool
        If True, use logistic growth with saturation.
    cap_value : float or None
        Upper bound (cap) for logistic growth. If None, defaults to 1.25 × historical max, capped at 35%.
    floor_value : float
        Lower bound (floor) for logistic growth.

    Returns:
    -------
    model : Prophet object
        Trained Prophet model.
    forecast : pd.DataFrame
        Prophet forecast output including yhat.
    forecast_smooth : pd.DataFrame
        Forecast with an additional 'yhat_smooth' column (rolling mean).
    """
    dfm = df_model.copy()

    # Set logistic cap/floor (y is in percentage)
    if use_logistic:
        if cap_value is None:
            cap_value = min(dfm["y"].max() * 1.25, 35.0)
        dfm["cap"] = cap_value
        dfm["floor"] = floor_value

    # Disable all seasonality and tighten changepoints to avoid noise
    model = Prophet(
        growth='logistic' if use_logistic else 'linear',
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=cps,
        n_changepoints=n_changepoints,
        interval_width=0.6
    )
    model.fit(dfm)

    last_year = dfm["ds"].dt.year.max()
    future = model.make_future_dataframe(periods=end_year - last_year, freq="YE")
    if use_logistic:
        future["cap"] = cap_value
        future["floor"] = floor_value

    forecast = model.predict(future)

    # Apply light smoothing (centered rolling mean)
    forecast_smooth = forecast.copy()
    forecast_smooth["yhat_smooth"] = (
        forecast_smooth["yhat"]
        .rolling(window=roll_k, center=True, min_periods=1)
        .mean()
    )
    return model, forecast, forecast_smooth


def evaluate_forecast(y_true, y_pred):
    """
    Evaluate forecast accuracy with common metrics.

    Metrics:
    - MAE (Mean Absolute Error): Lower is better.
    - RMSE (Root Mean Squared Error): Penalizes large errors.
    - MAPE (Mean Absolute Percentage Error): Measures error as a percentage of actual value.

    Parameters:
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Model predictions.

    Returns:
    -------
    mae : float
        Mean absolute error.
    rmse : float
        Root mean squared error.
    mape : float
        Mean absolute percentage error (%).
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = (abs((y_true - y_pred) / y_true)).mean() * 100
    return mae, rmse, mape


def forecast_65plus_prophet(input_path, output_path, region="England", end_year=2070):
    """
    Forecast the 65+ population ratio for a specific UK region using Prophet.

    Parameters:
    ----------
    input_path : str
        Path to population CSV (must contain Age, Year, Country, Population).
    output_path : str
        Path to save forecast chart (PNG).
    region : str
        Region name (e.g., "England").
    end_year : int
        Forecast horizon end year.

    Output:
    -------
    Saves a PNG chart showing Prophet fit and forecast with uncertainty intervals.
    """
    df = pd.read_csv(input_path)
    df = df[df["Sex"] == "Total"]

    def parse_age(age_str):
        try:
            if not isinstance(age_str, str):
                return None
            age_str = age_str.strip().replace("–", "-").replace(" ", "")
            if "All" in age_str or age_str == "":
                return None
            return int(age_str.split("-")[0].replace("+", "").strip())
        except:
            return None
    df["AgeStart"] = df["Age"].apply(parse_age)

    df = df[df["Country"].str.strip().str.lower() == region.strip().lower()]
    if df.empty:
        print(f"No data found for '{region}' after filtering. Please check spelling or data source.")
        return

    df_65 = df[df["AgeStart"] >= 65].groupby("Year")["Population"].sum().reset_index(name="Pop65plus")
    df_total = df.groupby("Year")["Population"].sum().reset_index(name="PopTotal")
    df_combined = pd.merge(df_65, df_total, on="Year")
    df_combined["Percent65plus"] = df_combined["Pop65plus"] / df_combined["PopTotal"] * 100

    if df_combined["Percent65plus"].dropna().shape[0] < 2:
        print("Insufficient data (<2 rows) to build model.")
        return

    df_model = df_combined[["Year", "Percent65plus"]].rename(columns={"Year": "ds", "Percent65plus": "y"})
    df_model["ds"] = pd.to_datetime(df_model["ds"], format="%Y")

    model, forecast, forecast_smooth = build_prophet_forecast(
        df_model, end_year=end_year,
        cps=0.04, n_changepoints=8, roll_k=5,
        use_logistic=True, cap_value=30.0, floor_value=0.0
    )

    fig, ax = plt.subplots()
    ax.plot(forecast_smooth["ds"], forecast_smooth["yhat_smooth"], label="Prophet (smoothed)")
    ax.set_title(f"Prophet Forecast of 65+ Population % in {region}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Percentage (%)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Forecast chart saved: {output_path}")

    merged = pd.merge(df_model, forecast, on="ds")
    mae, rmse, mape = evaluate_forecast(merged["y"], merged["yhat"])
    print(f"Prophet evaluation ({region}):")
    print(f"    MAE = {mae:.4f}, RMSE = {rmse:.4f}, MAPE = {mape:.2f}%")


def forecast_all_regions(input_path, output_path, regions=["England", "Wales", "Scotland"], end_year=2070):
    """
    Run Prophet forecasts for multiple UK regions and plot them together.

    Parameters:
    ----------
    input_path : str
        Path to population CSV.
    output_path : str
        Path to save combined forecast chart.
    regions : list of str
        Regions to process.
    end_year : int
        Forecast horizon end year.

    Output:
    -------
    Saves a PNG showing smoothed Prophet forecasts for all regions.
    """
    df = pd.read_csv(input_path)
    df = df[df["Sex"] == "Total"]

    def parse_age(age_str):
        try:
            if not isinstance(age_str, str):
                return None
            age_str = age_str.strip().replace("–", "-").replace(" ", "")
            if "All" in age_str or age_str == "":
                return None
            return int(age_str.split("-")[0].replace("+", "").strip())
        except:
            return None

    df["AgeStart"] = df["Age"].apply(parse_age)
    results = {}

    for region in regions:
        df_r = df[df["Country"].str.strip().str.lower() == region.lower()]

        df_65 = df_r[df_r["AgeStart"] >= 65].groupby("Year")["Population"].sum().reset_index(name="Pop65plus")
        df_total = df_r.groupby("Year")["Population"].sum().reset_index(name="PopTotal")
        df_combined = pd.merge(df_65, df_total, on="Year")
        df_combined["Percent65plus"] = df_combined["Pop65plus"] / df_combined["PopTotal"] * 100

        df_model_full = df_combined[["Year", "Percent65plus"]].rename(columns={"Year": "ds", "Percent65plus": "y"})
        df_model_full["ds"] = pd.to_datetime(df_model_full["ds"], format="%Y")

        df_train = df_model_full[df_model_full["ds"].dt.year <= 2023].copy()
        last_year = int(df_train["ds"].dt.year.max())

        model, forecast, forecast_smooth = build_prophet_forecast(
            df_train, end_year=end_year,
            cps=0.04, n_changepoints=8, roll_k=5,
            use_logistic=True, cap_value=30.0, floor_value=0.0
        )

        forecast_future = forecast[forecast["ds"].dt.year > last_year]
        year_min = int(forecast_future["ds"].dt.year.min())
        year_max = int(forecast_future["ds"].dt.year.max())

        print(f"\nProcessing region: {region}")
        print(f"{region} historical range: {int(df_train['ds'].dt.year.min())} → {last_year}")
        print(f"{region} forecast range (future only): {year_min} → {year_max}")

        results[region] = forecast_smooth

        merged = pd.merge(df_train, forecast[["ds", "yhat"]], on="ds")
        mae, rmse, mape = evaluate_forecast(merged["y"], merged["yhat"])
        print(f"Evaluation ({region}) → MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")

    print("Plotting multi-region Prophet forecast...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10.colors
    
    # Shaded uncertainty intervals
    for i, (region, fdf) in enumerate(results.items()):
        lb = fdf["yhat_lower"].rolling(3, center=True, min_periods=1).mean().clip(lower=0)
        ub = fdf["yhat_upper"].rolling(3, center=True, min_periods=1).mean()
        plt.fill_between(fdf["ds"], lb, ub, color=colors[i], alpha=0.18)
    
    # Smoothed forecast lines
    for i, (region, fdf) in enumerate(results.items()):
        plt.plot(fdf["ds"], fdf["yhat_smooth"], label=region, color=colors[i], linewidth=2)
    
    plt.ylim(bottom=0)
    plt.xlabel("Year")
    plt.ylabel("Percentage (%)")
    plt.title("Prophet Forecast: 65+ Population Ratio by Region")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Multi-region forecast chart saved: {output_path}")

    plot_forecast_comparison(results, output_path="output/forecast_comparison_regions.png")