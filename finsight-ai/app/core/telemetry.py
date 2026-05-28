"""
OpenTelemetry instrumentation for distributed tracing and metrics.
Traces export to an OTLP-compatible backend (e.g., AWS X-Ray, Grafana Tempo).
"""

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from app.core.config import settings


def configure_telemetry(service_name: str) -> None:
    """Bootstrap OTel tracing + metrics for the service."""

    # --- Tracing ---
    tracer_provider = TracerProvider()
    otlp_span_exporter = OTLPSpanExporter(
        endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
        insecure=True,
    )
    tracer_provider.add_span_processor(BatchSpanProcessor(otlp_span_exporter))
    trace.set_tracer_provider(tracer_provider)

    # --- Metrics ---
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT, insecure=True),
        export_interval_millis=10_000,
    )
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # --- Auto-instrumentation ---
    FastAPIInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()


def get_tracer(name: str = "finsight") -> trace.Tracer:
    return trace.get_tracer(name)


def get_meter(name: str = "finsight") -> metrics.Meter:
    return metrics.get_meter(name)
