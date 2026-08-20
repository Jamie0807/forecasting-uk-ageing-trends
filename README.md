# UK Ageing Trends Forecasting Platform

[English](README.md) | [中文](README_CN.md)

A full-stack data analytics and forecasting platform built with **Python + React**. It uses UK Office for National Statistics (ONS) population datasets to clean regional population projections, calculate the 65+ population share, compare Prophet and ARIMA forecasts, run KMeans clustering, and expose the results through a FastAPI + React dashboard.

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
| **Pipeline forecast end year** | 2070 (`CONFIG["end_year"]`) |
| **Model comparison horizon** | 30 years from the test split (`CONFIG["horizon"]`) |
| **Methods** | ONS data cleaning, Prophet, ARIMA, KMeans clustering |
| **Data source** | UK Office for National Statistics (ONS) |

---

## Core Features

### 1. Data Preprocessing (`src/preprocess*.py`)
- Extracts population data from ONS Excel / XLS files in `data/raw/`.
- Cleans and reshapes historical and projection data into CSV files under `data/processed/`.
- Supports England, Wales, Scotland and UK-level projection inputs.

### 2. Data Integration (`src/merge_projection_data.py`, `src/plot_ageing.py`)
- Merges England, Wales and Scotland projection outputs.
- Computes the 65+ population share and writes `data/processed/ageing_ratio_per_region.csv`.
- Generates the ageing trend chart used by the offline analysis outputs.

### 3. Time-Series Forecasting
- **Prophet**: generates England and multi-region forecast outputs through the batch pipeline.
- **ARIMA**: compares against Prophet for England and for multiple regions.
- Forecast CSVs and metrics are read by the FastAPI service from `output/` and `output/multi_compare/`.

### 4. Cluster Analysis (`src/cluster_analysis.py`)
- Applies KMeans to identify regions with similar ageing trajectories.
- Uses standardised time-series features for fair comparison.

### 5. Interactive Web Platform
- **Dashboard**: historical trend chart and regional summary cards.
- **Forecast**: interactive region and single-model switching between Prophet and ARIMA, plus evaluation metrics.
- **Cluster**: adjustable cluster count with grouped regional trend views.

### 6. FastAPI Backend
- Serves processed data and forecast outputs through `/api/*` endpoints.
- Hosts Swagger UI assets locally at `/docs` to avoid relying on a public CDN.
- Provides a root health response at `/`.

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

This runs the 8-step offline pipeline in `main.py`: raw data cleaning, regional merge, ageing-ratio generation, Prophet forecasts, England forecast export, ARIMA comparison, multi-region Prophet/ARIMA comparison and clustering.

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
| GET | `/` | Backend health response and docs pointer |

---

## Tech Stack

### Backend

| Technology | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **FastAPI** | REST API framework |
| **uvicorn** | ASGI server |
| **Pandas / NumPy** | Data processing and numerical computing |
| **Prophet** | Offline time-series forecasting in the analysis pipeline |
| **pmdarima / Statsmodels** | Offline ARIMA modelling and diagnostics |
| **scikit-learn** | KMeans clustering and preprocessing |
| **swagger-ui-bundle** | Local Swagger UI assets for backend docs |

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

The root `requirements.txt` is for the offline analysis pipeline. `backend/requirements.txt` contains the lighter FastAPI runtime dependencies. `frontend/package.json` contains the React/Vite dependencies.

---

## Generated Artifacts

- `data/processed/` and `output/` contain processed datasets, forecast CSVs and generated charts used by the demo.
- `frontend/dist/` is a Vite build artifact and is ignored by `.gitignore`; it should not be committed.
- Rebuild frontend assets with `cd frontend && npm run build` when needed.

---

## Model Notes

### Prophet
- Used by `src/model_prophet.py` and `src/multi_region_compare.py`.
- Produces forecast charts and CSV outputs consumed by the dashboard backend.

### ARIMA
- Used by `src/model_arima.py` and `src/multi_region_compare.py`.
- Produces comparison charts, forecast CSVs and MAE / RMSE / MAPE metrics.

### KMeans
- Used by `src/cluster_analysis.py` and the backend cluster service.
- Standardises regional ageing trajectories before assigning cluster labels.

---

## Project Highlights

- **Full-stack architecture**: FastAPI backend with a React frontend.
- **Official data workflow**: ONS raw data, cleaned CSV outputs and generated forecasts.
- **Multi-model comparison**: Prophet vs ARIMA using MAE, RMSE and MAPE outputs from `output/multi_compare/`.
- **Interactive visualisation**: Region, model and cluster controls in the browser.
- **End-to-end ETL pipeline**: From raw data to processed data, forecasts and charts.
- **Reproducible analysis**: Fixed random seed, explicit pipeline configuration and committed processed outputs.

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
