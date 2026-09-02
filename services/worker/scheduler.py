"""Durable recurring Source scheduler shared by all ingestion workers.

Every worker may scan for due Sources. A deterministic schedule Job ID plus the
repository's atomic ``add_job_if_absent`` operation guarantees that one schedule
window produces at most one durable ingestion Job across worker replicas.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Optional

from services.api.app.storage.models import IngestionJob, Source

MIN_SYNC_INTERVAL_SECONDS = 60


def schedule_due_sources(repo, *, now: Optional[datetime] = None, limit: int = 100) -> dict[str, int]:
    current = now or datetime.now(timezone.utc)
    due = _due_sources(repo, current, limit)
    stats = {"scanned": len(due), "enqueued": 0, "already_enqueued": 0, "blocked_active": 0}

    for source in due:
        interval = int(source.sync_interval_seconds or 0)
        if interval < MIN_SYNC_INTERVAL_SECONDS or not source.sync_next_at:
            continue
        due_at = _parse_time(source.sync_next_at)
        job_id = scheduled_job_id(source.source_id, due_at)

        active = _latest_active_job(repo, source)
        if active is not None and active.job_id != job_id:
            stats["blocked_active"] += 1
            continue

        existing = repo.get_job(job_id)
        inserted = False
        if existing is None:
            job = IngestionJob(
                job_id=job_id,
                tenant_id=source.tenant_id,
                source_id=source.source_id,
                source_type=source.source_type,
                source_config=deepcopy(source.config),
                status="pending",
                created_at=current.isoformat(),
                available_at=current.isoformat(),
                stats={"trigger": "scheduled", "scheduled_for": due_at.isoformat()},
            )
            inserted = bool(repo.add_job_if_absent(job))
            if inserted:
                stats["enqueued"] += 1
            else:
                stats["already_enqueued"] += 1
        else:
            stats["already_enqueued"] += 1

        # Advance the Source even when another worker won the insert race or a
        # previous process crashed after inserting the deterministic Job but
        # before updating Source scheduling state.
        next_at = _next_future_window(due_at, interval, current)
        repo.update_source(
            source.source_id,
            sync_next_at=next_at.isoformat(),
            sync_last_enqueued_at=current.isoformat(),
        )

    return stats


def configure_source_sync(
    repo,
    source: Source,
    *,
    enabled: bool,
    interval_seconds: Optional[int],
    run_immediately: bool = False,
    now: Optional[datetime] = None,
) -> Source:
    current = now or datetime.now(timezone.utc)
    if enabled:
        if interval_seconds is None or interval_seconds < MIN_SYNC_INTERVAL_SECONDS:
            raise ValueError(f"sync interval must be >= {MIN_SYNC_INTERVAL_SECONDS} seconds")
        next_at = current if run_immediately else current + timedelta(seconds=interval_seconds)
        updated = repo.update_source(
            source.source_id,
            sync_enabled=True,
            sync_interval_seconds=interval_seconds,
            sync_next_at=next_at.isoformat(),
        )
    else:
        updated = repo.update_source(
            source.source_id,
            sync_enabled=False,
            sync_interval_seconds=None,
            sync_next_at=None,
        )
    if updated is None:
        raise ValueError(f"Source not found: {source.source_id}")
    return updated


def scheduled_job_id(source_id: str, due_at: datetime) -> str:
    normalized = due_at.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    digest = hashlib.sha256(f"scheduled-sync\0{source_id}\0{normalized}".encode("utf-8")).hexdigest()
    return f"sync-{digest[:27]}"


def _due_sources(repo, now: datetime, limit: int) -> list[Source]:
    optimized = getattr(repo, "list_due_sources", None)
    if callable(optimized):
        return list(optimized(now.isoformat(), limit=limit))
    result = []
    for source in repo.list_sources():
        if source.status != "active" or not source.sync_enabled or not source.sync_next_at:
            continue
        if _parse_time(source.sync_next_at) <= now:
            result.append(source)
    result.sort(key=lambda item: (_parse_time(item.sync_next_at), item.source_id))
    return result[:limit]


def _latest_active_job(repo, source: Source):
    active = [
        job for job in repo.list_jobs(tenant_id=source.tenant_id, source_id=source.source_id)
        if job.status in {"pending", "running"}
    ]
    if not active:
        return None
    return max(active, key=lambda item: (str(item.created_at or ""), item.job_id))


def _next_future_window(due_at: datetime, interval_seconds: int, now: datetime) -> datetime:
    next_at = due_at + timedelta(seconds=interval_seconds)
    if next_at > now:
        return next_at
    missed = int((now - next_at).total_seconds() // interval_seconds) + 1
    return next_at + timedelta(seconds=missed * interval_seconds)


def _parse_time(value) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
