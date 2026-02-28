"""OpenTelemetry-compatible tracing for ragbot agent pipeline.

Provides span management for agent stages (route, tool calls, synthesize,
verify, finalize) with timing and metadata. Falls back to a no-op tracer
when OpenTelemetry SDK is not installed.

Environment variables:
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (e.g., http://localhost:4317)
    OTEL_SERVICE_NAME: Service name (default: ragbot-api)
    RAGBOT_TRACING_ENABLED: Set to "true" to enable tracing (default: false)
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

_tracer = None
_USE_OTEL = False


def _init_otel() -> bool:
    """Try to initialize OpenTelemetry. Returns True if successful."""
    global _tracer, _USE_OTEL
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=endpoint)
        else:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            exporter = ConsoleSpanExporter()

        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("ragbot", "0.5.0")
        _USE_OTEL = True
        logger.info("OpenTelemetry tracing initialized (endpoint=%s)", endpoint or "console")
        return True
    except ImportError:
        logger.debug("OpenTelemetry SDK not installed; using built-in tracing")
        return False


def setup_tracing() -> None:
    """Initialize tracing if enabled via environment."""
    enabled = os.getenv("RAGBOT_TRACING_ENABLED", "false").lower() in ("true", "1", "yes")
    if enabled:
        _init_otel()


@dataclass
class Span:
    """A lightweight trace span with timing and attributes."""

    name: str
    trace_id: str = ""
    parent_id: str = ""
    span_id: str = ""
    start_time_ms: int = 0
    end_time_ms: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"  # ok | error
    children: list = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return self.end_time_ms - self.start_time_ms


@dataclass
class TraceRecord:
    """Complete trace for one agent request."""

    trace_id: str
    request_id: str
    spans: list = field(default_factory=list)

    @property
    def total_duration_ms(self) -> int:
        if not self.spans:
            return 0
        return self.spans[-1].end_time_ms - self.spans[0].start_time_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "total_duration_ms": self.total_duration_ms,
            "spans": [
                {
                    "name": s.name,
                    "duration_ms": s.duration_ms,
                    "attributes": s.attributes,
                    "status": s.status,
                }
                for s in self.spans
            ],
        }


class RequestTracer:
    """Per-request tracer that collects spans for one agent invocation.

    Usage::

        tracer = RequestTracer(request_id="abc")
        with tracer.span("route") as s:
            s.attributes["route"] = "doc_rag"
        with tracer.span("retrieve", tool="qdrant") as s:
            ...
        record = tracer.finish()
    """

    def __init__(self, request_id: str) -> None:
        import uuid
        self.request_id = request_id
        self.trace_id = uuid.uuid4().hex
        self._spans: list[Span] = []
        self._otel_span_ctx = None

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Generator[Span, None, None]:
        """Create a timed span. Attributes can be added during the span."""
        s = Span(
            name=name,
            trace_id=self.trace_id,
            start_time_ms=_now_ms(),
            attributes=dict(attrs),
        )

        otel_span = None
        if _USE_OTEL and _tracer:
            otel_span = _tracer.start_span(name, attributes=attrs)

        try:
            yield s
        except Exception as exc:
            s.status = "error"
            s.attributes["error"] = str(exc)
            if otel_span:
                otel_span.set_status(
                    _otel_status_error(str(exc))
                )
            raise
        finally:
            s.end_time_ms = _now_ms()
            self._spans.append(s)
            if otel_span:
                for k, v in s.attributes.items():
                    if isinstance(v, (str, int, float, bool)):
                        otel_span.set_attribute(k, v)
                otel_span.end()

    def finish(self) -> TraceRecord:
        """Finalize and return the complete trace record."""
        return TraceRecord(
            trace_id=self.trace_id,
            request_id=self.request_id,
            spans=list(self._spans),
        )


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _otel_status_error(description: str):
    """Create an OpenTelemetry error status (import-safe)."""
    try:
        from opentelemetry.trace import StatusCode, Status
        return Status(StatusCode.ERROR, description)
    except ImportError:
        return None
