"""
Unit tests for RiskScoringEngine.
Uses mocked DynamoDB and RAG retrieval to test scoring logic in isolation.
"""

import pytest
from unittest.mock import MagicMock, patch
from app.core.risk_engine import RiskScoringEngine, RiskLevel, MerchantProfile


def _make_engine(profile: MerchantProfile | None = None, rag_grounded: bool = True):
    engine = RiskScoringEngine.__new__(RiskScoringEngine)
    engine._dynamodb = MagicMock()
    engine._table = MagicMock()
    engine._rag = MagicMock()

    engine._get_merchant_profile = MagicMock(return_value=profile)
    engine._rag.retrieve = MagicMock(return_value=MagicMock(
        grounded=rag_grounded,
        top_confidence=0.85 if rag_grounded else 0.40,
        chunks=[{"text": "KYC AML threshold exceeded for large transactions", "source": "kyc_policy.pdf"}],
    ))
    return engine


class TestRiskClassification:
    def test_low_risk(self):
        engine = RiskScoringEngine.__new__(RiskScoringEngine)
        assert engine._classify_risk(0.10) == RiskLevel.LOW
        assert engine._classify_risk(0.34) == RiskLevel.LOW

    def test_medium_risk(self):
        engine = RiskScoringEngine.__new__(RiskScoringEngine)
        assert engine._classify_risk(0.35) == RiskLevel.MEDIUM
        assert engine._classify_risk(0.59) == RiskLevel.MEDIUM

    def test_high_risk(self):
        engine = RiskScoringEngine.__new__(RiskScoringEngine)
        assert engine._classify_risk(0.60) == RiskLevel.HIGH
        assert engine._classify_risk(0.79) == RiskLevel.HIGH

    def test_critical_risk(self):
        engine = RiskScoringEngine.__new__(RiskScoringEngine)
        assert engine._classify_risk(0.80) == RiskLevel.CRITICAL
        assert engine._classify_risk(1.00) == RiskLevel.CRITICAL


class TestScoring:
    def test_unknown_merchant_gets_moderate_base_score(self):
        engine = _make_engine(profile=None, rag_grounded=False)
        result = engine.score("txn_001", "mch_unknown", 500.0)
        assert result.risk_score >= 0.20
        assert any("unknown" in f.lower() for f in result.contributing_factors)

    def test_high_anomaly_merchant_raises_score(self):
        profile = MerchantProfile(
            merchant_id="mch_bad",
            anomaly_score=0.9,
            flagged_transactions=10,
            avg_transaction_amount=5000,
            high_risk_country=True,
            velocity_spike=True,
        )
        engine = _make_engine(profile=profile)
        result = engine.score("txn_002", "mch_bad", 1000.0)
        assert result.risk_score >= 0.60
        assert result.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_large_transaction_triggers_kyc_factor(self):
        profile = MerchantProfile(
            merchant_id="mch_ok",
            anomaly_score=0.1,
            flagged_transactions=0,
            avg_transaction_amount=200,
            high_risk_country=False,
            velocity_spike=False,
        )
        engine = _make_engine(profile=profile, rag_grounded=True)
        result = engine.score("txn_003", "mch_ok", 15000.0)
        assert any("10K" in f or "threshold" in f.lower() for f in result.contributing_factors)

    def test_score_capped_at_1(self):
        profile = MerchantProfile(
            merchant_id="mch_worst",
            anomaly_score=1.0,
            flagged_transactions=100,
            avg_transaction_amount=0,
            high_risk_country=True,
            velocity_spike=True,
        )
        engine = _make_engine(profile=profile)
        result = engine.score("txn_004", "mch_worst", 100000.0)
        assert result.risk_score <= 1.0

    def test_rag_context_flag(self):
        engine = _make_engine(rag_grounded=True)
        result = engine.score("txn_005", "mch_any", 1000.0)
        assert result.rag_context_used is True

    def test_low_confidence_rag_not_used(self):
        engine = _make_engine(rag_grounded=False)
        result = engine.score("txn_006", "mch_any", 1000.0)
        assert result.rag_context_used is False
