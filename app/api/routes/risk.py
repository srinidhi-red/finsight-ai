"""Risk scoring endpoint — real-time fraud risk analysis."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.risk_engine import RiskScoringEngine, RiskLevel

router = APIRouter()
_engine = RiskScoringEngine()


class RiskRequest(BaseModel):
    transaction_id: str = Field(..., example="txn_abc123")
    merchant_id: str = Field(..., example="mch_xyz789")
    amount: float = Field(..., gt=0, example=15000.00)
    country: str = Field(default="US", example="US")


class RiskResponse(BaseModel):
    transaction_id: str
    merchant_id: str
    risk_score: float
    risk_level: RiskLevel
    contributing_factors: list[str]
    rag_context_used: bool
    retrieval_confidence: float
    reasoning: str


@router.post("/score", response_model=RiskResponse)
async def score_transaction(request: RiskRequest):
    """
    Compute real-time fraud risk score for a transaction.
    Combines merchant anomaly profile (DynamoDB) with RAG compliance context.
    Returns 0.0–1.0 score + risk level + explainable contributing factors.
    """
    try:
        result = _engine.score(
            transaction_id=request.transaction_id,
            merchant_id=request.merchant_id,
            amount=request.amount,
            country=request.country,
        )
        return RiskResponse(
            transaction_id=result.transaction_id,
            merchant_id=result.merchant_id,
            risk_score=result.risk_score,
            risk_level=result.risk_level,
            contributing_factors=result.contributing_factors,
            rag_context_used=result.rag_context_used,
            retrieval_confidence=result.retrieval_confidence,
            reasoning=result.reasoning,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
