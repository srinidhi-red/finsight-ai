"""
RAG Retrieval Engine
Handles semantic search over indexed compliance documents via
OpenSearch Serverless + Amazon Titan embeddings.
Enforces sub-200ms SLO with confidence-based guardrails.
"""

import time
import json
import logging
from dataclasses import dataclass
from typing import Optional

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from opentelemetry import trace

from app.core.config import settings
from app.core.telemetry import get_tracer, get_meter

logger = logging.getLogger(__name__)
tracer = get_tracer()
meter = get_meter()

retrieval_latency = meter.create_histogram(
    "retrieval.latency_ms",
    description="End-to-end retrieval latency in milliseconds",
    unit="ms",
)
retrieval_confidence = meter.create_histogram(
    "retrieval.confidence_score",
    description="Cosine similarity of top retrieved chunk",
)


@dataclass
class RetrievalResult:
    chunks: list[dict]
    top_confidence: float
    latency_ms: float
    query: str
    met_slo: bool
    grounded: bool  # False if confidence below threshold → triggers fallback


class RAGRetrievalEngine:
    """
    Semantic retrieval over OpenSearch Serverless.
    Embeddings via Amazon Titan; guardrails via confidence thresholding.
    """

    def __init__(self):
        self._os_client: Optional[OpenSearch] = None
        self._bedrock = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)

    def _get_opensearch_client(self) -> OpenSearch:
        if self._os_client:
            return self._os_client

        session = boto3.Session()
        credentials = session.get_credentials()
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            settings.AWS_REGION,
            "aoss",
            session_token=credentials.token,
        )
        self._os_client = OpenSearch(
            hosts=[{"host": settings.OPENSEARCH_ENDPOINT, "port": 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=5,
        )
        return self._os_client

    def _embed(self, text: str) -> list[float]:
        """Embed query text using Amazon Titan Embeddings v1."""
        body = json.dumps({"inputText": text})
        response = self._bedrock.invoke_model(
            modelId=settings.EMBEDDING_MODEL,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        return json.loads(response["body"].read())["embedding"]

    def retrieve(self, query: str, top_k: int = None) -> RetrievalResult:
        """
        Retrieve top-K semantically similar chunks for a query.
        Applies confidence guardrails and records OTel spans + metrics.
        """
        k = top_k or settings.TOP_K_RETRIEVAL
        start = time.monotonic()

        with tracer.start_as_current_span("rag.retrieve") as span:
            span.set_attribute("query.length", len(query))
            span.set_attribute("retrieval.top_k", k)

            query_vector = self._embed(query)

            os_client = self._get_opensearch_client()
            response = os_client.search(
                index=settings.OPENSEARCH_INDEX,
                body={
                    "size": k,
                    "query": {
                        "knn": {
                            "embedding": {
                                "vector": query_vector,
                                "k": k,
                            }
                        }
                    },
                    "_source": ["text", "source", "doc_type", "chunk_id"],
                },
            )

            hits = response["hits"]["hits"]
            latency_ms = (time.monotonic() - start) * 1000

            chunks = [
                {
                    "text": h["_source"]["text"],
                    "source": h["_source"]["source"],
                    "doc_type": h["_source"].get("doc_type", "unknown"),
                    "score": h["_score"],
                }
                for h in hits
            ]

            top_confidence = chunks[0]["score"] if chunks else 0.0
            grounded = top_confidence >= settings.MIN_CONFIDENCE_THRESHOLD
            met_slo = latency_ms <= settings.RETRIEVAL_LATENCY_SLO_MS

            span.set_attribute("retrieval.top_confidence", top_confidence)
            span.set_attribute("retrieval.latency_ms", latency_ms)
            span.set_attribute("retrieval.grounded", grounded)
            span.set_attribute("retrieval.met_slo", met_slo)

            retrieval_latency.record(latency_ms)
            retrieval_confidence.record(top_confidence)

            if not grounded:
                logger.warning(
                    "Low-confidence retrieval",
                    extra={"query": query[:80], "score": top_confidence},
                )

            if not met_slo:
                logger.warning(
                    "Retrieval SLO breach",
                    extra={"latency_ms": latency_ms, "slo_ms": settings.RETRIEVAL_LATENCY_SLO_MS},
                )

            return RetrievalResult(
                chunks=chunks,
                top_confidence=top_confidence,
                latency_ms=latency_ms,
                query=query,
                met_slo=met_slo,
                grounded=grounded,
            )
