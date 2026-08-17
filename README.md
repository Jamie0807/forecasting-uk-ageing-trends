# UK Ageing Trends Forecasting Platform

[English](README.md) | [中文](README_CN.md)

A full-stack data analytics and forecasting platform built with **Python + React**. It uses official UK Office for National Statistics (ONS) population data to analyse and visualise long-term ageing trends across UK regions, focusing on the share of people aged 65 and over.

| Layer | Stack |
|---|---|
| **Backend** | Python, FastAPI, Prophet, ARIMA, scikit-learn |
| **Frontend** | React 18, Vite, Recharts, Tailwind CSS |
| **Data** | Official ONS Excel / XLS datasets |

---

## Project Structure

```text
forecasting-uk-ageing-trends/
├── backend/                         # FastAPI backend
│   ├── app/
│   │   ├── main.py                  # App entry, CORS and OpenAPI setup
│   │   ├── routers/
│   │   │   ├── data.py              # /api/ageing-ratio, /regions, /overview
│   │   │   ├── forecast.py          # /api/forecast/prophet, /forecast/arima, /metrics
│   │   │   └── cluster.py           # /api/cluster
│   │   └── services/
│   │       ├── data_service.py      # Historical ageing ratio data access
│   │       ├── forecast_service.py  # Prophet / ARIMA forecast result access
│   │       └── cluster_service.py   # KMeans clustering
│   └── requirements.txt
├── frontend/                        # React frontend
│   ├── src/
│   │   ├── App.jsx                  # React Router configuration
│   │   ├── api/client.js            # Axios API client
│   │   ├── components/Navbar.jsx    # Top navigation
│   │   └── pages/
│   │       ├── Dashboard.jsx        # Historical trend overview and stat cards
│   │       ├── Forecast.jsx         # Interactive Prophet / ARIMA forecast view
│   │       └── Cluster.jsx          # Regional clustering analysis
│   ├── vite.config.js               # Vite proxy configuration
│   └── package.json
├── src/                             # Python analysis modules
│   ├── preprocess*.py               # Regional data cleaning scripts
│   ├── merge_projection_data.py     # Historical + projection data merge
│   ├── model_prophet.py             # Prophet forecasting
│   ├── model_arima.py               # ARIMA modelling and comparison
│   ├── cluster_analysis.py          # KMeans clustering
│   └── plot_*.py                    # Offline visualisation scripts
├── data/
│   ├── raw/                         # Original ONS Excel / XLS files
│   └── processed/                   # Cleaned CSV files
├── output/                          # Forecast outputs and generated charts
│   └── multi_compare/               # Multi-model comparison outputs
├── main.py                          # Offline batch pipeline entry
└── requirements.txt                 # Python dependencies
```

---

## Overview

| Item | Description |
|---|---|
| **Research focus** | 65+ population share in England, Wales and Scotland |
| **Forecast horizon** | 2020-2150 |
| **Methods** | Prophet, ARIMA, KMeans clustering |
| **Data source** | UK Office for National Statistics (ONS) |

---

## Core Features

### 1. Data Preprocessing (`src/preprocess*.py`)
- Extracts population data from official ONS Excel / XLS files.
- Cleans and reshapes data by age, year and region.
- Supports England, Wales, Scotland and UK-level processing.

### 2. Data Integration (`src/merge_projection_data.py`)
- Merges historical observations with official population projections.
- Computes the 65+ population share for each region.

### 3. Time-Series Forecasting
- **Prophet**: trend modelling with changepoint detection and smoothing.
- **ARIMA**: automatic parameter search with AIC-based model selection.

### 4. Cluster Analysis (`src/cluster_analysis.py`)
- Applies KMeans to identify regions with similar ageing trajectories.
- Uses standardised time-series features for fair comparison.

### 5. Interactive Web Platform
- **Dashboard**: historical trend chart and regional summary cards.
- **Forecast**: interactive region and model switching with evaluation metrics.
- **Cluster**: adjustable cluster count with grouped regional trend views.

---

## Quick Start

### Prerequisites

- Python 3.8+; a conda environment is recommended.
- Node.js 18+.

### 1. Generate analysis data

```bash
pip install -r requirements.txt
python main.py
```

### 2. Start the FastAPI backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API documentation: http://localhost:8000/docs

### 3. Start the React frontend

```bash
cd frontend
npm install
npm run dev
```

Open the app at http://localhost:5173.

### Pages

| Path | Page | Description |
|---|---|---|
| `/dashboard` | Dashboard | Historical ageing trend overview and stat cards |
| `/forecast` | Forecast | Interactive Prophet / ARIMA forecast chart and metrics table |
| `/cluster` | Cluster | KMeans clustering analysis and regional grouping |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/ageing-ratio` | Historical ageing ratio by region and year |
| GET | `/api/regions` | Available regions |
| GET | `/api/overview` | Per-region summary statistics |
| GET | `/api/forecast/prophet?region=England` | Historical + Prophet forecast series |
| GET | `/api/forecast/arima?region=England` | Historical + ARIMA forecast series |
| GET | `/api/metrics` | Prophet vs ARIMA metrics: MAE, RMSE and MAPE |
| GET | `/api/cluster?n_clusters=3` | KMeans cluster assignments and trend data |

---

## Tech Stack

### Backend

| Technology | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **FastAPI** | REST API framework |
| **uvicorn** | ASGI server |
| **Pandas / NumPy** | Data processing and numerical computing |
| **Prophet** | Time-series forecasting |
| **pmdarima / Statsmodels** | ARIMA modelling and diagnostics |
| **scikit-learn** | KMeans clustering and preprocessing |

### Frontend

| Technology | Purpose |
|---|---|
| **React 18** | Component-based UI |
| **Vite** | Frontend build tool and dev server |
| **Recharts** | Interactive React charts |
| **Tailwind CSS** | Utility-first styling |
| **React Router v6** | Client-side routing |
| **Axios** | HTTP client |

---

## Offline Pipeline Configuration

The batch pipeline parameters can be adjusted in `main.py`:

```python
CONFIG = {
    "regions": ["England", "Wales", "Scotland"],
    "end_year": 2070,
    "test_year_start": 2030,
    "horizon": 30,
    "n_clusters": 3,
    "random_state": 42
}
```

---

## Model Notes

### Prophet
- Piecewise trend modelling with changepoint detection.
- Forecast smoothing for long-term trend readability.

### ARIMA
- Automatic `(p, d, q)` parameter search via `pmdarima`.
- AIC-based model selection for stationary or differenced time series.

### KMeans
- Standardises regional ageing trajectories before clustering.
- Groups regions by similar long-term ageing patterns.

---

## Project Highlights

- **Full-stack architecture**: FastAPI backend with a React frontend.
- **Official data workflow**: ONS raw data, cleaned CSV outputs and generated forecasts.
- **Multi-model comparison**: Prophet vs ARIMA using MAE, RMSE and MAPE.
- **Interactive visualisation**: Region, model and cluster controls in the browser.
- **End-to-end ETL pipeline**: From raw data to processed data, forecasts and charts.
- **Reproducible analysis**: Fixed random seed and committed processed outputs.

---

## Optimization Roadmap

### 1. Engineering and Reproducibility
- Add a `Makefile` or task runner for common commands such as data generation, backend startup, frontend startup and tests.
- Add `.env.example` and move configurable paths, ports and API settings out of hard-coded code.
- Add Docker or `docker-compose` so the full application can be started with one command.
- Clarify which files are source data, processed data and generated outputs.

### 2. Backend API Quality
- Cache CSV reads to avoid loading the same processed files on every request.
- Validate `region` values and return clear 400 / 404 errors for unsupported regions.
- Add Pydantic response models for API contracts.
- Improve error handling for missing files, changed column names and empty datasets.
- Add a `/health` endpoint that reports backend and data availability.

### 3. Forecasting and Analytical Rigor
- Add confidence intervals to long-term forecasts where model outputs support them.
- Add a simple baseline model, such as naive or linear trend, to make Prophet / ARIMA gains easier to evaluate.
- Highlight the best-performing model per region in the metrics table.
- Document forecasting assumptions, especially where predictions extend beyond official projection horizons.

### 4. Frontend Experience
- Add key insight cards, such as fastest ageing region, highest latest 65+ share and largest historical change.
- Let users compare Prophet and ARIMA on the same chart.
- Add clearer empty, loading, error and retry states.
- Improve mobile handling for charts and metric tables.
- Add short interpretation notes so charts communicate conclusions, not just data.

### 5. Testing and Quality Gates
- Add `pytest` coverage for backend services and API endpoints.
- Add frontend smoke tests or component tests for the main pages.
- Add formatting and linting with tools such as `ruff`, ESLint and Prettier.
- Add GitHub Actions or another CI workflow to run tests and frontend builds.

### 6. Portfolio and Deployment
- Add screenshots or a short demo GIF to the README.
- Add a deployment target for the frontend and backend.
- Include a concise architecture diagram and data-flow diagram.
- Expand the README with project challenges, decisions and trade-offs for interview use.

---

## License

This project is intended for academic and research use.
