"""
Document Ingestion Pipeline
Pulls compliance documents from S3, chunks them, embeds via
Amazon Titan, and indexes into OpenSearch Serverless.
Supports PDF, TXT, and JSON document types.
"""

import io
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Iterator

import boto3
from opentelemetry import trace

from app.core.config import settings
from app.core.retrieval_engine import RAGRetrievalEngine
from app.core.telemetry import get_tracer, get_meter

logger = logging.getLogger(__name__)
tracer = get_tracer()
meter = get_meter()

docs_ingested = meter.create_counter(
    "ingestion.documents_total",
    description="Total documents successfully ingested",
)
chunks_indexed = meter.create_counter(
    "ingestion.chunks_total",
    description="Total chunks indexed into OpenSearch",
)


CHUNK_SIZE = 512        # tokens (approximate, character-based here)
CHUNK_OVERLAP = 64


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    doc_type: str
    embedding: list[float]


def _sliding_window_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Iterator[str]:
    """Yield overlapping text chunks from a document."""
    step = size - overlap
    for i in range(0, max(len(text) - overlap, 1), step):
        chunk = text[i: i + size].strip()
        if chunk:
            yield chunk


class IngestionPipeline:
    """
    S3 → chunk → embed → OpenSearch ingestion pipeline.
    Designed for compliance documents: KYC/AML policies, payment regulations.
    """

    def __init__(self):
        self._s3 = boto3.client("s3", region_name=settings.AWS_REGION)
        self._rag = RAGRetrievalEngine()

    def _read_s3_object(self, key: str) -> str:
        response = self._s3.get_object(Bucket=settings.S3_BUCKET_NAME, Key=key)
        content = response["Body"].read()

        if key.endswith(".pdf"):
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    return "\n".join(p.extract_text() or "" for p in pdf.pages)
            except ImportError:
                logger.warning("pdfplumber not available, returning raw bytes as text")
                return content.decode("utf-8", errors="ignore")

        if key.endswith(".json"):
            data = json.loads(content)
            if isinstance(data, list):
                return "\n".join(str(item) for item in data)
            return json.dumps(data, indent=2)

        return content.decode("utf-8", errors="ignore")

    def _detect_doc_type(self, key: str) -> str:
        key_lower = key.lower()
        if "kyc" in key_lower or "aml" in key_lower:
            return "kyc_aml"
        if "pci" in key_lower:
            return "pci_dss"
        if "regulation" in key_lower or "reg" in key_lower:
            return "regulation"
        if "runbook" in key_lower:
            return "runbook"
        return "compliance"

    def ingest_s3_key(self, s3_key: str) -> dict:
        """
        Ingest a single S3 document into the OpenSearch index.
        Returns ingestion summary with chunk count and status.
        """
        with tracer.start_as_current_span("ingestion.ingest_document") as span:
            span.set_attribute("s3.key", s3_key)

            text = self._read_s3_object(s3_key)
            doc_type = self._detect_doc_type(s3_key)
            os_client = self._rag._get_opensearch_client()

            chunk_count = 0
            bulk_body = []

            for chunk_text in _sliding_window_chunks(text):
                embedding = self._rag._embed(chunk_text)
                chunk_id = str(uuid.uuid4())

                bulk_body.append({"index": {"_index": settings.OPENSEARCH_INDEX, "_id": chunk_id}})
                bulk_body.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "source": s3_key,
                    "doc_type": doc_type,
                    "embedding": embedding,
                })
                chunk_count += 1

                # Flush every 50 chunks
                if len(bulk_body) >= 100:
                    os_client.bulk(body=bulk_body)
                    bulk_body = []

            if bulk_body:
                os_client.bulk(body=bulk_body)

            docs_ingested.add(1, {"doc_type": doc_type})
            chunks_indexed.add(chunk_count, {"doc_type": doc_type})

            span.set_attribute("ingestion.chunk_count", chunk_count)
            logger.info("Document ingested", extra={"key": s3_key, "chunks": chunk_count})

            return {"s3_key": s3_key, "doc_type": doc_type, "chunks_indexed": chunk_count, "status": "ok"}

    def ingest_prefix(self, prefix: str = "") -> dict:
        """Ingest all documents under an S3 prefix."""
        paginator = self._s3.get_paginator("list_objects_v2")
        results = []

        for page in paginator.paginate(Bucket=settings.S3_BUCKET_NAME, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith((".pdf", ".txt", ".json")):
                    try:
                        result = self.ingest_s3_key(key)
                        results.append(result)
                    except Exception as e:
                        logger.error("Failed to ingest document", extra={"key": key, "error": str(e)})
                        results.append({"s3_key": key, "status": "error", "error": str(e)})

        total_chunks = sum(r.get("chunks_indexed", 0) for r in results)
        return {"documents": len(results), "total_chunks": total_chunks, "results": results}
