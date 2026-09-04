from __future__ import annotations

from typing import Optional


class SourceFenceError(RuntimeError):
    """Raised when a job no longer belongs to the current Source lifecycle."""


def source_generation(source) -> str:
    """Return the durable lifecycle token for a Source.

    ``created_at`` is immutable for one active Source lifecycle. Quick Import
    assigns a new ``created_at`` when restoring a tombstoned deterministic
    Source, so old queued/running Jobs cannot publish into the restored Source.
    Legacy fixtures without timestamps fall back to a stable source-id token.
    """
    created_at = getattr(source, "created_at", None)
    return str(created_at) if created_at else f"legacy:{source.source_id}"


def job_source_generation(job) -> Optional[str]:
    stats = dict(getattr(job, "stats", {}) or {})
    value = stats.get("source_generation")
    return str(value) if value not in (None, "") else None


def job_stats_for_source(source, stats: Optional[dict] = None) -> dict:
    result = dict(stats or {})
    result["source_generation"] = source_generation(source)
    return result


def assert_source_fence(source, repo, expected_generation: Optional[str] = None):
    """Fail closed if the Source was deleted/recreated while this Job runs."""
    current = repo.get_source(source.source_id)
    if current is None:
        raise SourceFenceError(f"Source disappeared during ingestion: {source.source_id}")
    if current.tenant_id != source.tenant_id:
        raise SourceFenceError(f"Source tenant changed during ingestion: {source.source_id}")
    if current.status == "deleted":
        raise SourceFenceError(f"Source is deleted during ingestion: {source.source_id}")

    expected = expected_generation or source_generation(source)
    actual = source_generation(current)
    if actual != expected:
        raise SourceFenceError(
            "Source lifecycle generation changed during ingestion: "
            f"source={source.source_id} expected={expected} actual={actual}"
        )
    return current
