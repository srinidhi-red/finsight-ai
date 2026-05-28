"""
Transaction Risk Analysis Engine
Combines RAG-retrieved compliance context with DynamoDB-backed
merchant anomaly profiles to generate real-time fraud risk scores.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import boto3
from boto3.dynamodb.conditions import Key
from opentelemetry import trace

from app.core.config import settings
from app.core.retrieval_engine import RAGRetrievalEngine, RetrievalResult
from app.core.telemetry import get_tracer, get_meter

logger = logging.getLogger(__name__)
tracer = get_tracer()
meter = get_meter()

risk_score_histogram = meter.create_histogram(
    "risk.score",
    description="Distribution of fraud risk scores (0.0–1.0)",
)


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class MerchantProfile:
    merchant_id: str
    anomaly_score: float          # 0.0–1.0 from historical analysis
    flagged_transactions: int
    avg_transaction_amount: float
    high_risk_country: bool
    velocity_spike: bool


@dataclass
class RiskAnalysisResult:
    merchant_id: str
    transaction_id: str
    risk_score: float             # 0.0–1.0
    risk_level: RiskLevel
    contributing_factors: list[str]
    rag_context_used: bool
    retrieval_confidence: float
    reasoning: str


class RiskScoringEngine:
    """
    Two-signal fraud risk scoring:
      1. Merchant anomaly profile from DynamoDB
      2. RAG-retrieved compliance/KYC-AML context
    Outputs a composite 0–1 risk score with explainability.
    """

    RISK_THRESHOLDS = {
        RiskLevel.LOW: (0.0, 0.35),
        RiskLevel.MEDIUM: (0.35, 0.60),
        RiskLevel.HIGH: (0.60, 0.80),
        RiskLevel.CRITICAL: (0.80, 1.01),
    }

    def __init__(self):
        self._dynamodb = boto3.resource("dynamodb", region_name=settings.AWS_REGION)
        self._table = self._dynamodb.Table(settings.DYNAMODB_TABLE_MERCHANT)
        self._rag = RAGRetrievalEngine()

    def _get_merchant_profile(self, merchant_id: str) -> Optional[MerchantProfile]:
        try:
            response = self._table.query(
                KeyConditionExpression=Key("merchant_id").eq(merchant_id)
            )
            items = response.get("Items", [])
            if not items:
                return None
            item = items[0]
            return MerchantProfile(
                merchant_id=merchant_id,
                anomaly_score=float(item.get("anomaly_score", 0.0)),
                flagged_transactions=int(item.get("flagged_transactions", 0)),
                avg_transaction_amount=float(item.get("avg_transaction_amount", 0.0)),
                high_risk_country=item.get("high_risk_country", False),
                velocity_spike=item.get("velocity_spike", False),
            )
        except Exception as e:
            logger.error("DynamoDB lookup failed", extra={"merchant_id": merchant_id, "error": str(e)})
            return None

    def _classify_risk(self, score: float) -> RiskLevel:
        for level, (low, high) in self.RISK_THRESHOLDS.items():
            if low <= score < high:
                return level
        return RiskLevel.CRITICAL

    def _build_rag_query(self, merchant_id: str, amount: float, country: str) -> str:
        return (
            f"KYC AML compliance rules for merchant transactions. "
            f"Merchant {merchant_id}, transaction amount ${amount:.2f}, "
            f"originating country {country}. Applicable risk thresholds and flagging criteria."
        )

    def score(
        self,
        transaction_id: str,
        merchant_id: str,
        amount: float,
        country: str = "US",
    ) -> RiskAnalysisResult:
        """
        Compute composite fraud risk score for a transaction.
        Combines merchant DynamoDB profile + RAG compliance context.
        """
        with tracer.start_as_current_span("risk.score") as span:
            span.set_attribute("transaction.id", transaction_id)
            span.set_attribute("merchant.id", merchant_id)
            span.set_attribute("transaction.amount", amount)

            factors: list[str] = []
            base_score = 0.0

            # --- Signal 1: Merchant anomaly profile ---
            profile = self._get_merchant_profile(merchant_id)
            if profile:
                base_score += profile.anomaly_score * 0.45
                if profile.velocity_spike:
                    base_score += 0.10
                    factors.append("Velocity spike detected on merchant account")
                if profile.high_risk_country:
                    base_score += 0.08
                    factors.append("Merchant associated with high-risk jurisdiction")
                if profile.flagged_transactions > 5:
                    base_score += 0.07
                    factors.append(f"{profile.flagged_transactions} prior flagged transactions")
            else:
                # Unknown merchant — moderate prior
                base_score += 0.20
                factors.append("Merchant profile not found — unknown entity risk")

            # --- Signal 2: RAG compliance context ---
            rag_query = self._build_rag_query(merchant_id, amount, country)
            retrieval: RetrievalResult = self._rag.retrieve(rag_query, top_k=3)
            rag_boost = 0.0

            if retrieval.grounded:
                # Heuristic: high-amount transactions against KYC/AML thresholds
                if amount > 10_000:
                    rag_boost += 0.12
                    factors.append("Transaction exceeds KYC reporting threshold ($10K)")
                if amount > 50_000:
                    rag_boost += 0.08
                    factors.append("Transaction exceeds enhanced due diligence threshold ($50K)")

            final_score = min(base_score + rag_boost, 1.0)
            risk_level = self._classify_risk(final_score)

            reasoning = (
                f"Composite score {final_score:.3f} derived from merchant anomaly profile "
                f"(weight 0.45) and RAG compliance signal (weight 0.20). "
                f"Retrieval confidence: {retrieval.top_confidence:.3f}. "
                f"Risk level: {risk_level}."
            )

            span.set_attribute("risk.score", final_score)
            span.set_attribute("risk.level", risk_level)
            risk_score_histogram.record(final_score)

            return RiskAnalysisResult(
                merchant_id=merchant_id,
                transaction_id=transaction_id,
                risk_score=final_score,
                risk_level=risk_level,
                contributing_factors=factors,
                rag_context_used=retrieval.grounded,
                retrieval_confidence=retrieval.top_confidence,
                reasoning=reasoning,
            )
