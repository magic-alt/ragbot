from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_meter = None
_agent_requests = None
_agent_duration = None
_retrieval_duration = None
_tool_calls = None
_tool_duration = None
_feedback = None
_initialized = False


def setup_otel_metrics() -> None:
    """Initialize optional OTLP metrics export.

    This is independent from Prometheus scraping. Enable it with
    ``RAGBOT_OTEL_METRICS_ENABLED=true`` and configure
    ``OTEL_EXPORTER_OTLP_ENDPOINT``. Missing observability extras degrade to a
    no-op instead of preventing the API from starting.
    """
    global _meter, _agent_requests, _agent_duration, _retrieval_duration
    global _tool_calls, _tool_duration, _feedback, _initialized

    if _initialized:
        return
    _initialized = True
    enabled = os.getenv("RAGBOT_OTEL_METRICS_ENABLED", "false").lower() in ("true", "1", "yes")
    if not enabled:
        return

    try:
        from opentelemetry import metrics
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        exporter = OTLPMetricExporter(endpoint=endpoint) if endpoint else OTLPMetricExporter()
        reader = PeriodicExportingMetricReader(exporter)
        provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(provider)
        _meter = metrics.get_meter("ragbot", "0.5.0")
        _agent_requests = _meter.create_counter(
            "ragbot.agent.requests",
            unit="1",
            description="Completed Ragbot Agent requests",
        )
        _agent_duration = _meter.create_histogram(
            "ragbot.agent.request.duration",
            unit="s",
            description="End-to-end Ragbot Agent request duration",
        )
        _retrieval_duration = _meter.create_histogram(
            "ragbot.agent.retrieval.duration",
            unit="s",
            description="Ragbot retrieval/tool evidence acquisition duration",
        )
        _tool_calls = _meter.create_counter(
            "ragbot.agent.tool.calls",
            unit="1",
            description="Ragbot Agent tool calls",
        )
        _tool_duration = _meter.create_histogram(
            "ragbot.agent.tool.duration",
            unit="s",
            description="Ragbot Agent tool call duration",
        )
        _feedback = _meter.create_counter(
            "ragbot.agent.feedback",
            unit="1",
            description="Accepted Ragbot request feedback events",
        )
        logger.info("OpenTelemetry metrics initialized (endpoint=%s)", endpoint or "SDK default")
    except ImportError:
        logger.warning("OpenTelemetry metrics requested but observability dependencies are not installed")
    except Exception:
        logger.exception("OpenTelemetry metrics initialization failed; continuing without OTLP metrics")


def observe_agent(metrics: Any) -> None:
    if _agent_requests is None:
        return
    route = str(getattr(metrics, "route", "") or "unknown")
    confidence = str(getattr(metrics, "confidence", "") or "unknown")
    attrs = {
        "route": route,
        "confidence": confidence,
        "cited": bool(getattr(metrics, "has_citations", False)),
    }
    _agent_requests.add(1, attrs)
    _agent_duration.record(max(0.0, float(getattr(metrics, "total_duration_ms", 0))) / 1000.0, {"route": route})
    _retrieval_duration.record(
        max(0.0, float(getattr(metrics, "retrieval_duration_ms", 0))) / 1000.0,
        {"route": route},
    )
    for tool_call in list(getattr(metrics, "tool_calls", ()) or ()):
        tool = str(tool_call.get("name") or "unknown")
        status = "success" if bool(tool_call.get("ok", False)) else "failure"
        tool_attrs = {"tool": tool, "status": status}
        _tool_calls.add(1, tool_attrs)
        _tool_duration.record(
            max(0.0, float(tool_call.get("duration_ms") or 0)) / 1000.0,
            {"tool": tool},
        )


def observe_feedback(feedback: str) -> None:
    if _feedback is not None:
        _feedback.add(1, {"feedback": str(feedback)})
