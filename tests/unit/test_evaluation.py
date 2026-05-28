"""
Unit tests for the RAGEvaluator harness.
Validates metric calculations and summary aggregation.
"""

import pytest
from unittest.mock import MagicMock
from app.evaluation.ragas_harness import (
    RAGEvaluator, EvalCase, _token_overlap,
    _faithfulness, _context_precision, _context_recall
)


CHUNKS = [
    {"text": "PCI DSS requires encryption of cardholder data at rest and in transit", "source": "pci_dss.pdf"},
    {"text": "KYC regulations mandate identity verification for transactions over $10K", "source": "kyc_policy.pdf"},
]


class TestMetrics:
    def test_token_overlap_identical(self):
        assert _token_overlap("hello world", "hello world") == 1.0

    def test_token_overlap_disjoint(self):
        assert _token_overlap("hello", "world") == 0.0

    def test_token_overlap_partial(self):
        score = _token_overlap("hello world", "hello there")
        assert 0.0 < score < 1.0

    def test_faithfulness_high_when_answer_from_context(self):
        answer = "PCI DSS requires encryption of cardholder data"
        score = _faithfulness(answer, CHUNKS)
        assert score > 0.3

    def test_faithfulness_zero_no_chunks(self):
        assert _faithfulness("any answer", []) == 0.0

    def test_context_precision_all_relevant(self):
        relevant_chunks = [{"text": "PCI DSS encryption rules for payment data"}]
        score = _context_precision("PCI DSS encryption", relevant_chunks)
        assert score == 1.0

    def test_context_recall_all_sources_found(self):
        score = _context_recall(["pci_dss.pdf"], CHUNKS)
        assert score == 1.0

    def test_context_recall_missing_source(self):
        score = _context_recall(["missing_doc.pdf"], CHUNKS)
        assert score == 0.0

    def test_context_recall_empty_ground_truth(self):
        assert _context_recall([], CHUNKS) == 1.0


class TestEvaluator:
    def _make_evaluator(self):
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = MagicMock(
            chunks=CHUNKS,
            grounded=True,
            top_confidence=0.88,
        )
        evaluator = RAGEvaluator(
            retrieval_engine=mock_rag,
            generation_fn=lambda q, chunks: chunks[0]["text"] if chunks else "",
        )
        return evaluator

    def test_evaluate_single_case(self):
        evaluator = self._make_evaluator()
        case = EvalCase(
            query="PCI DSS encryption requirements",
            expected_answer="PCI DSS requires encryption of cardholder data",
            ground_truth_sources=["pci_dss.pdf"],
        )
        result = evaluator.evaluate_case(case)
        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.context_precision <= 1.0
        assert result.context_recall == 1.0
        assert result.grounded is True

    def test_run_produces_summary(self):
        evaluator = self._make_evaluator()
        cases = [
            EvalCase(
                query=f"compliance query {i}",
                expected_answer="some compliant answer",
                ground_truth_sources=["pci_dss.pdf"],
            )
            for i in range(5)
        ]
        summary = evaluator.run(cases)
        assert summary.total_cases == 5
        assert 0.0 <= summary.mean_faithfulness <= 1.0
        assert 0.0 <= summary.grounded_rate <= 1.0
        assert summary.p95_latency_ms >= 0
