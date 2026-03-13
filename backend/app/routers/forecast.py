from fastapi import APIRouter, Query
from app.services.forecast_service import (
    get_prophet_forecast,
    get_arima_forecast,
    get_metrics,
)

router = APIRouter()


@router.get("/forecast/prophet")
def prophet_forecast(region: str = Query("England", description="Region name")):
    """Return historical + Prophet forecast series for a region."""
    return get_prophet_forecast(region)


@router.get("/forecast/arima")
def arima_forecast(region: str = Query("England", description="Region name")):
    """Return historical + ARIMA forecast series for a region."""
    return get_arima_forecast(region)


@router.get("/metrics")
def metrics():
    """Return Prophet vs ARIMA model evaluation metrics for all regions."""
    return get_metrics()
