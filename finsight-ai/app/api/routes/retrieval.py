"""Retrieval endpoint — semantic search over compliance documents."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.retrieval_engine import RAGRetrievalEngine

router = APIRouter()
_engine = RAGRetrievalEngine()


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=1000, example="PCI DSS requirements for card-not-present transactions")
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalResponse(BaseModel):
    query: str
    chunks: list[dict]
    top_confidence: float
    latency_ms: float
    grounded: bool
    met_slo: bool


@router.post("/", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest):
    """
    Semantic retrieval over indexed compliance documents.
    Returns top-K chunks with confidence scores.
    Enforces 200ms SLO; grounded=False signals low-confidence results.
    """
    try:
        result = _engine.retrieve(query=request.query, top_k=request.top_k)
        return RetrievalResponse(
            query=result.query,
            chunks=result.chunks,
            top_confidence=result.top_confidence,
            latency_ms=result.latency_ms,
            grounded=result.grounded,
            met_slo=result.met_slo,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
