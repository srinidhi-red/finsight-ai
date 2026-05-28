"""Health check endpoint."""

from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()


@router.get("/")
async def health():
    return {"status": "ok", "env": settings.ENV, "version": "1.0.0"}


@router.get("/ready")
async def ready():
    return {"ready": True}
