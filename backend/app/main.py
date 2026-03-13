"""
FastAPI backend for the UK Ageing Trends Forecasting Platform.
Exposes REST API endpoints consumed by the React frontend.
"""
import sys
import os

# Add project root to Python path so src.* modules are importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import data, forecast, cluster

app = FastAPI(
    title="UK Ageing Trends API",
    description="REST API for UK population ageing trend analysis and forecasting",
    version="1.0.0",
)

# Allow requests from the React dev server and built frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router, prefix="/api", tags=["Data"])
app.include_router(forecast.router, prefix="/api", tags=["Forecast"])
app.include_router(cluster.router, prefix="/api", tags=["Cluster"])


@app.get("/", tags=["Health"])
def root():
    return {"message": "UK Ageing Trends API is running", "docs": "/docs"}
