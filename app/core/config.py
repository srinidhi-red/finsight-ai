"""
Application configuration — loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    # App
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    ALLOWED_ORIGINS: List[str] = ["*"]

    # AWS
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    S3_BUCKET_NAME: str = "finsight-documents"
    OPENSEARCH_ENDPOINT: str = ""
    OPENSEARCH_INDEX: str = "finsight-docs"
    DYNAMODB_TABLE_MERCHANT: str = "finsight-merchant-profiles"

    # SageMaker (LLaMA 3.1 fine-tuned endpoint)
    SAGEMAKER_ENDPOINT_NAME: str = "finsight-llama31-qlora"
    SAGEMAKER_REGION: str = "us-east-1"

    # RAG
    EMBEDDING_MODEL: str = "amazon.titan-embed-text-v1"
    TOP_K_RETRIEVAL: int = 5
    RETRIEVAL_LATENCY_SLO_MS: int = 200
    MIN_CONFIDENCE_THRESHOLD: float = 0.72

    # OpenTelemetry
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"
    OTEL_SERVICE_NAME: str = "finsight-api"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
