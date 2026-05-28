"""Ingestion endpoint."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.ingestion.pipeline import IngestionPipeline

router = APIRouter()
_pipeline = IngestionPipeline()


class IngestRequest(BaseModel):
    s3_key: str | None = None
    s3_prefix: str | None = None


@router.post("/")
async def ingest(request: IngestRequest):
    """Trigger ingestion for a single S3 key or all docs under a prefix."""
    try:
        if request.s3_key:
            return _pipeline.ingest_s3_key(request.s3_key)
        if request.s3_prefix is not None:
            return _pipeline.ingest_prefix(request.s3_prefix)
        raise HTTPException(status_code=400, detail="Provide s3_key or s3_prefix")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
