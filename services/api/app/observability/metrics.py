"""Request diagnostics plus production metric event emission for Ragbot.

The bounded in-memory history exists for request diagnostics, feedback lookup,
and the admin history surface. It is intentionally *not* the production metrics
backend. Every completed request/tool event is also emitted immediately to
Prometheus and, when configured, OpenTelemetry/OTLP so replicas can be
aggregated by the monitoring system.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics/diagnostics for a single request."""

    request_id: str
    tenant_id: str
    user_id: str
    timestamp: float = field(default_factory=time.time)
    route: str = ""
    total_duration_ms: int = 0
    route_duration_ms: int = 0
    retrieval_duration_ms: int = 0
    synthesis_duration_ms: int = 0
    verify_duration_ms: int = 0
    citation_count: int = 0
    evidence_count: int = 0
    confidence: str = ""
    has_citations: bool = False
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_success_count: int = 0
    tool_failure_count: int = 0
    iterations: int = 0
    feedback: Optional[str] = None


@dataclass
class AggregateMetrics:
    """Process-local diagnostic aggregation over the bounded request history."""

    window_start: float = 0.0
    window_end: float = 0.0
    total_requests: int = 0
    citation_coverage: float = 0.0
    avg_evidence_count: float = 0.0
    avg_citation_count: float = 0.0
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    retrieval_hit_rate: float = 0.0
    avg_retrieval_ms: float = 0.0
    tool_failure_rate: float = 0.0
    tool_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    positive_feedback: int = 0
    negative_feedback: int = 0
    feedback_score: float = 0.0
    avg_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    avg_iterations: float = 0.0


class MetricsCollector:
    """Bounded thread-safe request history for diagnostics and feedback lookup."""

    def __init__(self, max_history: int = 10000) -> None:
        self._lock = threading.Lock()
        self._history: List[RequestMetrics] = []
        self._max_history = max_history

    def record(self, metrics: RequestMetrics) -> None:
        with self._lock:
            self._history.append(metrics)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
        _emit_request_metrics(metrics)

    def record_feedback(self, request_id: str, feedback: str) -> bool:
        found = False
        with self._lock:
            for item in reversed(self._history):
                if item.request_id == request_id:
                    item.feedback = feedback
                    found = True
                    break
        if found:
            _emit_feedback(feedback)
        return found

    def aggregate(self, last_n: Optional[int] = None) -> AggregateMetrics:
        with self._lock:
            history = list(self._history)

        if last_n:
            history = history[-last_n:]
        if not history:
            return AggregateMetrics()

        total = len(history)
        agg = AggregateMetrics(
            window_start=history[0].timestamp,
            window_end=history[-1].timestamp,
            total_requests=total,
        )
        with_citations = sum(1 for item in history if item.has_citations)
        agg.citation_coverage = with_citations / total if total else 0.0
        agg.avg_citation_count = sum(item.citation_count for item in history) / total
        agg.avg_evidence_count = sum(item.evidence_count for item in history) / total

        for item in history:
            if item.confidence:
                agg.confidence_distribution[item.confidence] = (
                    agg.confidence_distribution.get(item.confidence, 0) + 1
                )

        retrieval_calls = []
        tool_total = 0
        tool_fails = 0
        tool_stats: Dict[str, Dict[str, int]] = {}
        for item in history:
            for tool_call in item.tool_calls:
                tool_name = tool_call.get("name", "unknown")
                ok = tool_call.get("ok", True)
                tool_total += 1
                if not ok:
                    tool_fails += 1
                if tool_name not in tool_stats:
                    tool_stats[tool_name] = {"success": 0, "failure": 0}
                if ok:
                    tool_stats[tool_name]["success"] += 1
                else:
                    tool_stats[tool_name]["failure"] += 1
                if tool_name == "retrieve":
                    retrieval_calls.append(ok)

        agg.tool_failure_rate = tool_fails / tool_total if tool_total else 0.0
        agg.tool_stats = tool_stats
        agg.retrieval_hit_rate = (
            sum(1 for result in retrieval_calls if result) / len(retrieval_calls)
            if retrieval_calls else 0.0
        )
        retrieval_durations = [
            item.retrieval_duration_ms for item in history if item.retrieval_duration_ms > 0
        ]
        agg.avg_retrieval_ms = (
            sum(retrieval_durations) / len(retrieval_durations) if retrieval_durations else 0.0
        )

        positive = sum(1 for item in history if item.feedback == "positive")
        negative = sum(1 for item in history if item.feedback == "negative")
        agg.positive_feedback = positive
        agg.negative_feedback = negative
        agg.feedback_score = positive / (positive + negative) if (positive + negative) else 0.0

        durations = [item.total_duration_ms for item in history]
        agg.avg_duration_ms = sum(durations) / len(durations) if durations else 0.0
        sorted_durations = sorted(durations)
        p95_idx = int(len(sorted_durations) * 0.95)
        agg.p95_duration_ms = (
            sorted_durations[min(p95_idx, len(sorted_durations) - 1)] if sorted_durations else 0.0
        )
        agg.avg_iterations = sum(item.iterations for item in history) / total
        return agg

    def get_history(self, last_n: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            history = list(self._history[-last_n:])
        return [
            {
                "request_id": item.request_id,
                "tenant_id": item.tenant_id,
                "user_id": item.user_id,
                "timestamp": item.timestamp,
                "route": item.route,
                "total_duration_ms": item.total_duration_ms,
                "confidence": item.confidence,
                "citation_count": item.citation_count,
                "evidence_count": item.evidence_count,
                "tool_success_count": item.tool_success_count,
                "tool_failure_count": item.tool_failure_count,
                "iterations": item.iterations,
                "feedback": item.feedback,
            }
            for item in history
        ]

    def reset(self) -> None:
        with self._lock:
            self._history.clear()


_collector: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    global _collector
    with _collector_lock:
        if _collector is None:
            _collector = MetricsCollector()
        return _collector


def _emit_request_metrics(metrics: RequestMetrics) -> None:
    try:
        from .prometheus import observe_agent as observe_prometheus
        observe_prometheus(metrics)
    except Exception:
        logger.exception("Failed to emit Prometheus Agent metrics")
    try:
        from .otel_metrics import observe_agent as observe_otel
        observe_otel(metrics)
    except Exception:
        logger.exception("Failed to emit OpenTelemetry Agent metrics")


def _emit_feedback(feedback: str) -> None:
    try:
        from .prometheus import observe_feedback as observe_prometheus_feedback
        observe_prometheus_feedback(feedback)
    except Exception:
        logger.exception("Failed to emit Prometheus feedback metric")
    try:
        from .otel_metrics import observe_feedback as observe_otel_feedback
        observe_otel_feedback(feedback)
    except Exception:
        logger.exception("Failed to emit OpenTelemetry feedback metric")


def build_request_metrics(state, trace_record=None) -> RequestMetrics:
    metrics = RequestMetrics(
        request_id=state.request_id,
        tenant_id=state.tenant_id,
        user_id=state.user_id,
        route=state.route or "",
        evidence_count=len(state.evidence),
        iterations=state.iteration,
    )

    if state.final:
        metrics.confidence = state.final.confidence
        metrics.citation_count = len(state.final.citations)
        metrics.has_citations = len(state.final.citations) > 0

    for tool_call in state.tool_calls:
        metrics.tool_calls.append(
            {
                "name": tool_call.name,
                "ok": tool_call.ok,
                "duration_ms": tool_call.ended_at_ms - tool_call.started_at_ms,
            }
        )
        if tool_call.ok:
            metrics.tool_success_count += 1
        else:
            metrics.tool_failure_count += 1

    if trace_record:
        metrics.total_duration_ms = trace_record.total_duration_ms
        for span in trace_record.spans:
            if span.name == "route":
                metrics.route_duration_ms = span.duration_ms
            elif span.name in ("retrieve", "sql_query", "code_search"):
                metrics.retrieval_duration_ms += span.duration_ms
            elif span.name == "synthesize":
                metrics.synthesis_duration_ms = span.duration_ms
            elif span.name == "verify":
                metrics.verify_duration_ms = span.duration_ms

    return metrics
