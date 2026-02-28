"""Quality metrics collection for ragbot.

Tracks:
- Citation coverage: fraction of answers with >= 1 citation
- Retrieval hit rate: fraction of retrieval calls returning results
- Tool failure rate: per-tool success/failure counts
- User feedback: thumbs up/down per request
- Cost: token usage and tool call counts per request

Metrics are collected in-memory with thread-safe counters.
Can be exported via /admin/metrics endpoint or pushed to external systems.
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
    """Metrics for a single request."""

    request_id: str
    tenant_id: str
    user_id: str
    timestamp: float = field(default_factory=time.time)

    # Route
    route: str = ""

    # Timing
    total_duration_ms: int = 0
    route_duration_ms: int = 0
    retrieval_duration_ms: int = 0
    synthesis_duration_ms: int = 0
    verify_duration_ms: int = 0

    # Quality
    citation_count: int = 0
    evidence_count: int = 0
    confidence: str = ""
    has_citations: bool = False

    # Tool calls
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_success_count: int = 0
    tool_failure_count: int = 0

    # Cost
    iterations: int = 0

    # Feedback
    feedback: Optional[str] = None  # "positive" | "negative" | None


@dataclass
class AggregateMetrics:
    """Aggregated metrics over a time window."""

    window_start: float = 0.0
    window_end: float = 0.0

    total_requests: int = 0

    # Quality
    citation_coverage: float = 0.0  # % of requests with citations
    avg_evidence_count: float = 0.0
    avg_citation_count: float = 0.0
    confidence_distribution: Dict[str, int] = field(default_factory=dict)

    # Retrieval
    retrieval_hit_rate: float = 0.0  # % of retrieval calls returning results
    avg_retrieval_ms: float = 0.0

    # Tools
    tool_failure_rate: float = 0.0  # % of tool calls that failed
    tool_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Feedback
    positive_feedback: int = 0
    negative_feedback: int = 0
    feedback_score: float = 0.0  # positive / (positive + negative)

    # Performance
    avg_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    avg_iterations: float = 0.0


class MetricsCollector:
    """Thread-safe in-memory metrics collector.

    Stores per-request metrics and provides aggregation.

    Usage::

        collector = MetricsCollector()
        # After each request:
        collector.record(request_metrics)
        # Export:
        agg = collector.aggregate()
    """

    def __init__(self, max_history: int = 10000) -> None:
        self._lock = threading.Lock()
        self._history: List[RequestMetrics] = []
        self._max_history = max_history

    def record(self, metrics: RequestMetrics) -> None:
        """Record metrics for a completed request."""
        with self._lock:
            self._history.append(metrics)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

    def record_feedback(self, request_id: str, feedback: str) -> bool:
        """Record user feedback for a request."""
        with self._lock:
            for m in reversed(self._history):
                if m.request_id == request_id:
                    m.feedback = feedback
                    return True
            return False

    def aggregate(self, last_n: Optional[int] = None) -> AggregateMetrics:
        """Compute aggregate metrics over recent requests."""
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

        # Citation coverage
        with_citations = sum(1 for m in history if m.has_citations)
        agg.citation_coverage = with_citations / total if total else 0.0
        agg.avg_citation_count = sum(m.citation_count for m in history) / total
        agg.avg_evidence_count = sum(m.evidence_count for m in history) / total

        # Confidence distribution
        for m in history:
            if m.confidence:
                agg.confidence_distribution[m.confidence] = (
                    agg.confidence_distribution.get(m.confidence, 0) + 1
                )

        # Retrieval hit rate
        retrieval_calls = []
        tool_total = 0
        tool_fails = 0
        tool_stats: Dict[str, Dict[str, int]] = {}

        for m in history:
            for tc in m.tool_calls:
                tool_name = tc.get("name", "unknown")
                ok = tc.get("ok", True)
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
            sum(1 for r in retrieval_calls if r) / len(retrieval_calls)
            if retrieval_calls else 0.0
        )

        # Retrieval timing
        retrieval_durations = [m.retrieval_duration_ms for m in history if m.retrieval_duration_ms > 0]
        agg.avg_retrieval_ms = sum(retrieval_durations) / len(retrieval_durations) if retrieval_durations else 0.0

        # Feedback
        positive = sum(1 for m in history if m.feedback == "positive")
        negative = sum(1 for m in history if m.feedback == "negative")
        agg.positive_feedback = positive
        agg.negative_feedback = negative
        agg.feedback_score = positive / (positive + negative) if (positive + negative) else 0.0

        # Performance
        durations = [m.total_duration_ms for m in history]
        agg.avg_duration_ms = sum(durations) / len(durations) if durations else 0.0
        sorted_durations = sorted(durations)
        p95_idx = int(len(sorted_durations) * 0.95)
        agg.p95_duration_ms = sorted_durations[min(p95_idx, len(sorted_durations) - 1)] if sorted_durations else 0.0
        agg.avg_iterations = sum(m.iterations for m in history) / total

        return agg

    def get_history(self, last_n: int = 100) -> List[Dict[str, Any]]:
        """Return recent request metrics as dicts."""
        with self._lock:
            history = list(self._history[-last_n:])
        return [
            {
                "request_id": m.request_id,
                "tenant_id": m.tenant_id,
                "user_id": m.user_id,
                "timestamp": m.timestamp,
                "route": m.route,
                "total_duration_ms": m.total_duration_ms,
                "confidence": m.confidence,
                "citation_count": m.citation_count,
                "evidence_count": m.evidence_count,
                "tool_success_count": m.tool_success_count,
                "tool_failure_count": m.tool_failure_count,
                "iterations": m.iterations,
                "feedback": m.feedback,
            }
            for m in history
        ]

    def reset(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._history.clear()


# Global singleton
_collector: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _collector
    with _collector_lock:
        if _collector is None:
            _collector = MetricsCollector()
        return _collector


def build_request_metrics(state, trace_record=None) -> RequestMetrics:
    """Build RequestMetrics from a completed AgentState and optional TraceRecord."""
    from contracts.types import AgentState

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

    # Tool call stats
    for tc in state.tool_calls:
        metrics.tool_calls.append({
            "name": tc.name,
            "ok": tc.ok,
            "duration_ms": tc.ended_at_ms - tc.started_at_ms,
        })
        if tc.ok:
            metrics.tool_success_count += 1
        else:
            metrics.tool_failure_count += 1

    # Timing from trace
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
