import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_and_prepare_data(input_path):
    """
    Load and prepare a CSV file into Prophet-compatible time series format.

    Parameters
    ----------
    input_path : str
        Path to the input CSV file (must contain 'ds' and 'y' columns).

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing 'ds' (datetime) and 'y' (numeric value) columns.
    """
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()  # Remove any whitespace in column names
    print("Original column names:", df.columns.tolist())

    # Convert 'ds' column to datetime format
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def forecast_with_prophet(df, forecast_years=50):
    """
    Forecast future values using the Prophet model.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'ds' (datetime) and 'y' (numeric value).
    forecast_years : int, optional (default=50)
        Number of years to forecast into the future.

    Returns
    -------
    forecast : pandas.DataFrame
        DataFrame with predicted values and confidence intervals, including:
        'ds', 'yhat', 'yhat_lower', and 'yhat_upper'.
    """
    model = Prophet()
    model.fit(df)

    # Create a future dataframe with yearly frequency (end of each year)
    future = model.make_future_dataframe(periods=forecast_years, freq="YE")
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def export_forecast_to_csv(forecast_df, output_path):
    """
    Save forecast results to a CSV file.

    Parameters
    ----------
    forecast_df : pandas.DataFrame
        Forecast results from Prophet.
    output_path : str
        Path to save the CSV file.
    """
    forecast_df.to_csv(output_path, index=False)
    print(f"Forecast results exported to: {output_path}")


def run_forecast_pipeline(region_name, input_path, output_path):
    """
    Run the full Prophet forecasting pipeline:
    Load data → Fit model → Forecast → Export results → Evaluate performance.

    Parameters
    ----------
    region_name : str
        Name of the region (for logging purposes, e.g., "England").
    input_path : str
        Path to the input CSV file (must contain 'ds' and 'y').
    output_path : str
        Path to save the forecast CSV file.
    """
    print(f"Processing {region_name} ...")

    # Load and prepare data
    df = load_and_prepare_data(input_path)

    # Forecast with Prophet
    forecast = forecast_with_prophet(df)

    # Save forecast results
    export_forecast_to_csv(forecast, output_path)

    # Evaluate model performance on overlapping years
    merged = pd.merge(df[["ds", "y"]], forecast, on="ds", how="inner")
    mae, rmse, mape = evaluate_forecast(merged["y"], merged["yhat"])

    print(f"Prophet evaluation for {region_name}:")
    print(f"   MAE = {mae:.4f}, RMSE = {rmse:.4f}, MAPE = {mape:.2f}%")


def evaluate_forecast(y_true, y_pred):
    """
    Evaluate forecast accuracy using MAE, RMSE, and MAPE.

    Parameters
    ----------
    y_true : array-like
        Actual observed values.
    y_pred : array-like
        Predicted values from the model.

    Returns
    -------
    mae : float
        Mean Absolute Error.
    rmse : float
        Root Mean Squared Error.
    mape : float
        Mean Absolute Percentage Error (in %).
    """
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)

    # Root Mean Squared Error
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    # Mean Absolute Percentage Error (avoid division by zero)
    mape = (abs((y_true - y_pred) / y_true)).mean() * 100

    return mae, rmse, mape