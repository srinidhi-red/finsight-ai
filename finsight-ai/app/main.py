"""
FinSight AI — AWS-Native Fintech RAG Platform
Entry point for the FastAPI application.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.routes import retrieval, ingest, risk, health
from app.core.config import settings
from app.core.logging import configure_logging
from app.core.telemetry import configure_telemetry

configure_logging()
configure_telemetry(service_name="finsight-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown


app = FastAPI(
    title="FinSight AI",
    description="AWS-native RAG platform for fintech compliance and fraud intelligence",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(retrieval.router, prefix="/api/v1/retrieve", tags=["retrieval"])
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingestion"])
app.include_router(risk.router, prefix="/api/v1/risk", tags=["risk"])
