"""
OpenTelemetry SDK initialization for vmo-pipecat (MELT Phase).

Initialises:
  - TracerProvider + BatchSpanProcessor → OTLP gRPC exporter (traces)
  - MeterProvider + PeriodicExportingMetricReader → OTLP gRPC exporter (metrics)
  - LoggingInstrumentor → injects trace_id/span_id into stdlib logging

Fail-safe: BatchSpanProcessor + async metric reader. If otel-collector is
unreachable, spans/metrics are dropped silently — never blocks the pipeline.

Usage (in runtime.py):
    from .observability.otel import init_otel
    init_otel()
"""

from __future__ import annotations

import os
import logging

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor

_logger = logging.getLogger(__name__)

_TRACER: trace.Tracer | None = None
_METER: metrics.Meter | None = None


def _build_resource() -> Resource:
    return Resource.create({
        SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "vmo-pipecat"),
    })


def _init_tracing(resource: Resource) -> trace.Tracer:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
    span_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("vmo-pipecat")


def _init_metrics(resource: Resource) -> metrics.Meter:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
    metric_exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
    reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=15_000)
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)
    return metrics.get_meter("vmo-pipecat")


def init_otel() -> None:
    """Initialise OpenTelemetry SDK: tracing, metrics, and logging instrumentation.

    Idempotent: safe to call multiple times.
    """
    global _TRACER, _METER
    if _TRACER is not None:
        return

    resource = _build_resource()

    _TRACER = _init_tracing(resource)
    _METER = _init_metrics(resource)

    # Inject trace_id / span_id into stdlib log records
    LoggingInstrumentor().instrument(set_logging_format=True)

    _logger.info("OpenTelemetry SDK initialised — traces + metrics via OTLP gRPC")


def get_tracer() -> trace.Tracer:
    """Return the configured OTel tracer (lazy-init if needed)."""
    global _TRACER
    if _TRACER is None:
        init_otel()
    assert _TRACER is not None
    return _TRACER


def get_meter() -> metrics.Meter:
    """Return the configured OTel meter (lazy-init if needed)."""
    global _METER
    if _METER is None:
        init_otel()
    assert _METER is not None
    return _METER
