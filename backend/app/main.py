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
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from swagger_ui_bundle import swagger_ui_path
from app.routers import data, forecast, cluster

app = FastAPI(
    title="UK Ageing Trends API",
    description="REST API for UK population ageing trend analysis and forecasting",
    version="1.0.0",
    docs_url=None,        # 禁用默认 CDN 版 /docs
    redoc_url=None,
)

# 降级为 OpenAPI 3.0.3（swagger-ui 4.x 不支持 3.1.0）
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    app.openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        openapi_version="3.0.3",
    )
    return app.openapi_schema

app.openapi = custom_openapi

# 本地托管 swagger-ui 静态文件（避免 jsdelivr CDN 被屏蔽）
app.mount("/swagger-ui-static", StaticFiles(directory=swagger_ui_path), name="swagger-ui-static")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="UK Ageing Trends API",
        swagger_js_url="/swagger-ui-static/swagger-ui-bundle.js",
        swagger_css_url="/swagger-ui-static/swagger-ui.css",
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
