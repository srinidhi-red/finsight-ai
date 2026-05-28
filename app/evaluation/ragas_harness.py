"""
RAG Evaluation Harness (RAGAS-inspired)
Measures: faithfulness, answer relevance, context precision, context recall.
Runs across a test dataset of (query, expected_answer, ground_truth_contexts).
Used to validate guardrail thresholds and retrieval configurations.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import statistics

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    query: str
    expected_answer: str
    ground_truth_sources: list[str]
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    case: EvalCase
    retrieved_chunks: list[dict]
    generated_answer: str
    faithfulness: float       # Is the answer grounded in retrieved context?
    answer_relevance: float   # Does the answer address the query?
    context_precision: float  # Are retrieved chunks relevant?
    context_recall: float     # Were ground-truth sources retrieved?
    latency_ms: float
    grounded: bool


@dataclass
class EvalSummary:
    total_cases: int
    mean_faithfulness: float
    mean_answer_relevance: float
    mean_context_precision: float
    mean_context_recall: float
    mean_latency_ms: float
    grounded_rate: float
    p95_latency_ms: float
    failed_cases: list[str]


def _token_overlap(a: str, b: str) -> float:
    """Jaccard token overlap as a lightweight similarity proxy."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _faithfulness(answer: str, chunks: list[dict]) -> float:
    """
    Heuristic faithfulness: fraction of answer tokens supported by retrieved context.
    In production, replace with an LLM-as-judge call.
    """
    if not chunks:
        return 0.0
    combined_context = " ".join(c["text"] for c in chunks)
    return _token_overlap(answer, combined_context)


def _answer_relevance(query: str, answer: str) -> float:
    """Token overlap between query and answer as a relevance proxy."""
    return _token_overlap(query, answer)


def _context_precision(query: str, chunks: list[dict], threshold: float = 0.10) -> float:
    """Fraction of retrieved chunks that are relevant to the query."""
    if not chunks:
        return 0.0
    relevant = sum(1 for c in chunks if _token_overlap(query, c["text"]) >= threshold)
    return relevant / len(chunks)


def _context_recall(ground_truth_sources: list[str], chunks: list[dict]) -> float:
    """Fraction of ground-truth sources present in retrieved chunks."""
    if not ground_truth_sources:
        return 1.0
    retrieved_sources = {c.get("source", "") for c in chunks}
    matched = sum(1 for s in ground_truth_sources if s in retrieved_sources)
    return matched / len(ground_truth_sources)


class RAGEvaluator:
    """
    Offline evaluation harness for the FinSight RAG pipeline.
    Loads test cases from a JSONL file and scores each against
    faithfulness, relevance, precision, recall, and latency.
    """

    def __init__(self, retrieval_engine, generation_fn=None):
        """
        Args:
            retrieval_engine: Instance of RAGRetrievalEngine
            generation_fn: Callable(query, chunks) -> str
                           Defaults to a simple extractive baseline.
        """
        self._rag = retrieval_engine
        self._generate = generation_fn or self._extractive_baseline

    @staticmethod
    def _extractive_baseline(query: str, chunks: list[dict]) -> str:
        """Return the most relevant chunk text as a simple baseline answer."""
        if not chunks:
            return "Insufficient context to answer."
        return chunks[0]["text"][:500]

    def load_test_cases(self, path: str) -> list[EvalCase]:
        """Load eval cases from a JSONL file."""
        cases = []
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            cases.append(EvalCase(
                query=data["query"],
                expected_answer=data["expected_answer"],
                ground_truth_sources=data.get("ground_truth_sources", []),
                metadata=data.get("metadata", {}),
            ))
        logger.info("Loaded %d eval cases from %s", len(cases), path)
        return cases

    def evaluate_case(self, case: EvalCase) -> EvalResult:
        start = time.monotonic()
        retrieval = self._rag.retrieve(case.query)
        answer = self._generate(case.query, retrieval.chunks)
        latency_ms = (time.monotonic() - start) * 1000

        return EvalResult(
            case=case,
            retrieved_chunks=retrieval.chunks,
            generated_answer=answer,
            faithfulness=_faithfulness(answer, retrieval.chunks),
            answer_relevance=_answer_relevance(case.query, answer),
            context_precision=_context_precision(case.query, retrieval.chunks),
            context_recall=_context_recall(case.ground_truth_sources, retrieval.chunks),
            latency_ms=latency_ms,
            grounded=retrieval.grounded,
        )

    def run(self, test_cases: list[EvalCase], verbose: bool = False) -> EvalSummary:
        """Run all test cases and return aggregated metrics."""
        results: list[EvalResult] = []
        failed: list[str] = []

        for i, case in enumerate(test_cases):
            try:
                result = self.evaluate_case(case)
                results.append(result)
                if verbose:
                    print(
                        f"[{i+1}/{len(test_cases)}] "
                        f"faith={result.faithfulness:.2f} "
                        f"rel={result.answer_relevance:.2f} "
                        f"prec={result.context_precision:.2f} "
                        f"rec={result.context_recall:.2f} "
                        f"lat={result.latency_ms:.0f}ms "
                        f"grounded={result.grounded}"
                    )
            except Exception as e:
                failed.append(f"{case.query[:60]}: {e}")
                logger.error("Eval case failed", extra={"query": case.query, "error": str(e)})

        if not results:
            raise ValueError("No eval cases succeeded.")

        latencies = sorted(r.latency_ms for r in results)
        p95_idx = int(len(latencies) * 0.95)

        return EvalSummary(
            total_cases=len(results),
            mean_faithfulness=statistics.mean(r.faithfulness for r in results),
            mean_answer_relevance=statistics.mean(r.answer_relevance for r in results),
            mean_context_precision=statistics.mean(r.context_precision for r in results),
            mean_context_recall=statistics.mean(r.context_recall for r in results),
            mean_latency_ms=statistics.mean(r.latency_ms for r in results),
            grounded_rate=sum(r.grounded for r in results) / len(results),
            p95_latency_ms=latencies[min(p95_idx, len(latencies) - 1)],
            failed_cases=failed,
        )

    def report(self, summary: EvalSummary) -> None:
        """Print a formatted evaluation report."""
        print("\n" + "="*60)
        print("  FinSight AI — RAG Evaluation Report")
        print("="*60)
        print(f"  Total cases evaluated : {summary.total_cases}")
        print(f"  Faithfulness          : {summary.mean_faithfulness:.3f}")
        print(f"  Answer relevance      : {summary.mean_answer_relevance:.3f}")
        print(f"  Context precision     : {summary.mean_context_precision:.3f}")
        print(f"  Context recall        : {summary.mean_context_recall:.3f}")
        print(f"  Grounded rate         : {summary.grounded_rate:.1%}")
        print(f"  Mean latency          : {summary.mean_latency_ms:.1f}ms")
        print(f"  p95 latency           : {summary.p95_latency_ms:.1f}ms")
        if summary.failed_cases:
            print(f"\n  ⚠ Failed cases ({len(summary.failed_cases)}):")
            for f in summary.failed_cases[:5]:
                print(f"    - {f}")
        print("="*60 + "\n")
