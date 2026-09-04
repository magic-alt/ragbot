from __future__ import annotations

from typing import Any

try:
    from prometheus_client import Counter, Gauge, Histogram, CONTENT_TYPE_LATEST, generate_latest
except ImportError:  # pragma: no cover - dependency guard for minimal installs
    Counter = Gauge = Histogram = None  # type: ignore[assignment]
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    generate_latest = None


if Counter is not None:
    HTTP_REQUESTS = Counter(
        "ragbot_http_requests_total",
        "HTTP requests handled by this Ragbot process.",
        ("method", "path", "status"),
    )
    HTTP_LATENCY = Histogram(
        "ragbot_http_request_duration_seconds",
        "HTTP request latency in seconds.",
        ("method", "path"),
    )
    QUEUE_JOBS = Gauge(
        "ragbot_ingestion_jobs",
        "Current durable ingestion jobs by status.",
        ("status",),
    )
    QUEUE_OLDEST_PENDING = Gauge(
        "ragbot_ingestion_oldest_pending_age_seconds",
        "Age of the oldest pending ingestion job.",
    )
    QUEUE_STALE_LEASES = Gauge(
        "ragbot_ingestion_stale_running_leases",
        "Running ingestion jobs whose lease is expired.",
    )
    SOURCE_COUNT = Gauge(
        "ragbot_sources",
        "Current Sources by state.",
        ("state",),
    )

    # Agent metrics are emitted at request completion rather than reconstructed
    # from process-local rolling history during scrape. Prometheus can aggregate
    # these counters/histograms correctly across replicas.
    AGENT_REQUESTS = Counter(
        "ragbot_agent_requests_total",
        "Completed Agent requests.",
        ("route", "confidence", "cited"),
    )
    AGENT_DURATION = Histogram(
        "ragbot_agent_request_duration_seconds",
        "End-to-end Agent request duration.",
        ("route",),
    )
    AGENT_RETRIEVAL_DURATION = Histogram(
        "ragbot_agent_retrieval_duration_seconds",
        "Agent retrieval/tool evidence acquisition duration.",
        ("route",),
    )
    AGENT_CITATIONS = Histogram(
        "ragbot_agent_citations",
        "Citation count per completed Agent request.",
        ("route",),
        buckets=(0, 1, 2, 3, 5, 8, 13, 21),
    )
    AGENT_EVIDENCE = Histogram(
        "ragbot_agent_evidence_items",
        "Evidence item count per completed Agent request.",
        ("route",),
        buckets=(0, 1, 2, 3, 5, 8, 13, 21, 34),
    )
    AGENT_ITERATIONS = Histogram(
        "ragbot_agent_iterations",
        "Agent loop iterations per completed request.",
        ("route",),
        buckets=(0, 1, 2, 3, 4, 5, 8, 13),
    )
    TOOL_CALLS = Counter(
        "ragbot_agent_tool_calls_total",
        "Agent tool calls by tool and outcome.",
        ("tool", "status"),
    )
    TOOL_DURATION = Histogram(
        "ragbot_agent_tool_duration_seconds",
        "Agent tool call duration.",
        ("tool",),
    )
    FEEDBACK = Counter(
        "ragbot_agent_feedback_total",
        "Accepted request feedback events.",
        ("feedback",),
    )
else:  # pragma: no cover
    HTTP_REQUESTS = HTTP_LATENCY = QUEUE_JOBS = QUEUE_OLDEST_PENDING = None
    QUEUE_STALE_LEASES = SOURCE_COUNT = None
    AGENT_REQUESTS = AGENT_DURATION = AGENT_RETRIEVAL_DURATION = None
    AGENT_CITATIONS = AGENT_EVIDENCE = AGENT_ITERATIONS = None
    TOOL_CALLS = TOOL_DURATION = FEEDBACK = None


def observe_http(method: str, path: str, status: int, duration_seconds: float) -> None:
    if HTTP_REQUESTS is None:
        return
    HTTP_REQUESTS.labels(method=method, path=path, status=str(status)).inc()
    HTTP_LATENCY.labels(method=method, path=path).observe(max(0.0, duration_seconds))


def observe_agent(metrics: Any) -> None:
    """Emit one completed Agent request into replica-aggregatable metrics."""
    if AGENT_REQUESTS is None:
        return
    route = str(getattr(metrics, "route", "") or "unknown")
    confidence = str(getattr(metrics, "confidence", "") or "unknown")
    cited = "true" if bool(getattr(metrics, "has_citations", False)) else "false"
    AGENT_REQUESTS.labels(route=route, confidence=confidence, cited=cited).inc()
    AGENT_DURATION.labels(route=route).observe(max(0.0, float(getattr(metrics, "total_duration_ms", 0))) / 1000.0)
    retrieval_ms = max(0.0, float(getattr(metrics, "retrieval_duration_ms", 0)))
    AGENT_RETRIEVAL_DURATION.labels(route=route).observe(retrieval_ms / 1000.0)
    AGENT_CITATIONS.labels(route=route).observe(max(0, int(getattr(metrics, "citation_count", 0))))
    AGENT_EVIDENCE.labels(route=route).observe(max(0, int(getattr(metrics, "evidence_count", 0))))
    AGENT_ITERATIONS.labels(route=route).observe(max(0, int(getattr(metrics, "iterations", 0))))

    for tool_call in list(getattr(metrics, "tool_calls", ()) or ()):
        tool = str(tool_call.get("name") or "unknown")
        status = "success" if bool(tool_call.get("ok", False)) else "failure"
        TOOL_CALLS.labels(tool=tool, status=status).inc()
        TOOL_DURATION.labels(tool=tool).observe(max(0.0, float(tool_call.get("duration_ms") or 0)) / 1000.0)


def observe_feedback(feedback: str) -> None:
    if FEEDBACK is not None:
        FEEDBACK.labels(feedback=str(feedback)).inc()


def render_prometheus(repo: Any) -> tuple[bytes, str]:
    if generate_latest is None:
        raise RuntimeError("prometheus-client is required for the Prometheus metrics endpoint")

    # Durable control-plane gauges are refreshed from the shared repository at
    # scrape time. Request/tool counters and histograms above are emitted as the
    # events happen and are naturally aggregated by Prometheus across replicas.
    from ..routes.control_plane import build_overview

    overview = build_overview(repo, None)
    queue = overview["queue"]
    for status in ("pending", "running", "failed", "dead_lettered", "completed"):
        QUEUE_JOBS.labels(status=status).set(float((queue.get("by_status") or {}).get(status, 0)))
    QUEUE_OLDEST_PENDING.set(float(queue.get("oldest_pending_age_seconds") or 0.0))
    QUEUE_STALE_LEASES.set(float(queue.get("stale_running_leases") or 0.0))

    sources = overview["sources"]
    for state in ("total", "active", "paused", "scheduled"):
        SOURCE_COUNT.labels(state=state).set(float(sources.get(state) or 0.0))

    return generate_latest(), CONTENT_TYPE_LATEST
