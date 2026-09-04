from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


class SourceFenceError(RuntimeError):
    """Raised when a job no longer belongs to the current Source lifecycle."""


def _canonical_timestamp(value) -> Optional[str]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value).strip()
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return raw
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def source_generation(source) -> str:
    """Return the durable lifecycle token for a Source.

    ``updated_at`` changes whenever the Source definition/lifecycle is mutated;
    ``created_at`` is the fallback for older rows. Timestamps are normalized so
    an API-side ISO string and the same PostgreSQL ``datetime`` compare equal.
    """
    updated_at = _canonical_timestamp(getattr(source, "updated_at", None))
    created_at = _canonical_timestamp(getattr(source, "created_at", None))
    if updated_at:
        return updated_at
    if created_at:
        return created_at
    return f"legacy:{source.source_id}"


def job_source_generation(job) -> Optional[str]:
    stats = dict(getattr(job, "stats", {}) or {})
    value = stats.get("source_generation")
    if value in (None, ""):
        return None
    normalized = _canonical_timestamp(value)
    return normalized or str(value)


def job_stats_for_source(source, stats: Optional[dict] = None) -> dict:
    result = dict(stats or {})
    result["source_generation"] = source_generation(source)
    return result


def assert_source_fence(source, repo, expected_generation: Optional[str] = None):
    """Fail closed if the Source changed while this Job runs."""
    current = repo.get_source(source.source_id)
    if current is None:
        raise SourceFenceError(f"Source disappeared during ingestion: {source.source_id}")
    if current.tenant_id != source.tenant_id:
        raise SourceFenceError(f"Source tenant changed during ingestion: {source.source_id}")
    if current.status == "deleted":
        raise SourceFenceError(f"Source is deleted during ingestion: {source.source_id}")

    expected = _canonical_timestamp(expected_generation) if expected_generation else source_generation(source)
    expected = expected or str(expected_generation)
    actual = source_generation(current)
    if actual != expected:
        raise SourceFenceError(
            "Source lifecycle generation changed during ingestion: "
            f"source={source.source_id} expected={expected} actual={actual}"
        )
    return current
