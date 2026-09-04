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
    AGENT_QUALITY = Gauge(
        "ragbot_agent_quality_ratio",
        "Process-local rolling Agent quality ratios.",
        ("metric",),
    )
    AGENT_LATENCY = Gauge(
        "ragbot_agent_duration_milliseconds",
        "Process-local rolling Agent latency statistics.",
        ("stat",),
    )
else:  # pragma: no cover
    HTTP_REQUESTS = HTTP_LATENCY = QUEUE_JOBS = QUEUE_OLDEST_PENDING = None
    QUEUE_STALE_LEASES = SOURCE_COUNT = AGENT_QUALITY = AGENT_LATENCY = None


def observe_http(method: str, path: str, status: int, duration_seconds: float) -> None:
    if HTTP_REQUESTS is None:
        return
    HTTP_REQUESTS.labels(method=method, path=path, status=str(status)).inc()
    HTTP_LATENCY.labels(method=method, path=path).observe(max(0.0, duration_seconds))


def render_prometheus(repo: Any) -> tuple[bytes, str]:
    if generate_latest is None:
        raise RuntimeError("prometheus-client is required for the Prometheus metrics endpoint")

    # Import lazily to avoid an API/router import cycle during module loading.
    from ..routes.control_plane import build_overview
    from .metrics import get_metrics_collector

    overview = build_overview(repo, None)
    queue = overview["queue"]
    for status in ("pending", "running", "failed", "dead_lettered", "completed"):
        QUEUE_JOBS.labels(status=status).set(float((queue.get("by_status") or {}).get(status, 0)))
    QUEUE_OLDEST_PENDING.set(float(queue.get("oldest_pending_age_seconds") or 0.0))
    QUEUE_STALE_LEASES.set(float(queue.get("stale_running_leases") or 0.0))

    sources = overview["sources"]
    for state in ("total", "active", "paused", "scheduled"):
        SOURCE_COUNT.labels(state=state).set(float(sources.get(state) or 0.0))

    aggregate = get_metrics_collector().aggregate()
    AGENT_QUALITY.labels(metric="citation_coverage").set(float(aggregate.citation_coverage))
    AGENT_QUALITY.labels(metric="retrieval_hit_rate").set(float(aggregate.retrieval_hit_rate))
    AGENT_QUALITY.labels(metric="tool_failure_rate").set(float(aggregate.tool_failure_rate))
    AGENT_QUALITY.labels(metric="feedback_score").set(float(aggregate.feedback_score))
    AGENT_LATENCY.labels(stat="average").set(float(aggregate.avg_duration_ms))
    AGENT_LATENCY.labels(stat="p95").set(float(aggregate.p95_duration_ms))

    return generate_latest(), CONTENT_TYPE_LATEST
