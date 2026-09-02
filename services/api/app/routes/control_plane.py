"""Product control-plane APIs for Source catalog and ingestion operations."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Query

from ..auth.principal import allowed_tenants, authorize_tenant, require_admin


def create_control_plane_router(get_services: Callable, auth_dep: Any) -> APIRouter:
    router = APIRouter(tags=["control-plane"])

    @router.get("/catalog/overview")
    async def catalog_overview(
        tenant_id: Optional[str] = None,
        _key: Optional[str] = Depends(auth_dep),
    ):
        services = get_services()
        tenant_scope = _resolve_tenant_scope(_key, tenant_id)
        return build_overview(services.repo, tenant_scope)

    @router.get("/catalog/sources")
    async def source_catalog(
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        source_type: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = Query(default=100, ge=1, le=500),
        _key: Optional[str] = Depends(auth_dep),
    ):
        services = get_services()
        tenant_scope = _resolve_tenant_scope(_key, tenant_id)
        sources = _scoped_sources(services.repo, tenant_scope)
        jobs = _scoped_jobs(services.repo, tenant_scope)
        by_source: dict[str, list] = {}
        for job in jobs:
            by_source.setdefault(job.source_id, []).append(job)
        for items in by_source.values():
            items.sort(key=_job_sort_key, reverse=True)

        needle = (q or "").strip().lower()
        result = []
        for source in sources:
            if source.status == "deleted":
                continue
            if status and source.status != status:
                continue
            if source_type and source.source_type != source_type:
                continue
            if needle and needle not in " ".join(
                [source.name, source.source_id, source.tenant_id, source.source_type, *source.tags]
            ).lower():
                continue
            source_jobs = by_source.get(source.source_id, [])
            latest = source_jobs[0] if source_jobs else None
            result.append(_source_catalog_item(source, latest, source_jobs))

        result.sort(key=lambda item: (item["tenant_id"], item["name"].lower(), item["source_id"]))
        return {"total": len(result), "sources": result[:limit]}

    @router.get("/catalog/jobs")
    async def job_catalog(
        tenant_id: Optional[str] = None,
        source_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = Query(default=100, ge=1, le=500),
        _key: Optional[str] = Depends(auth_dep),
    ):
        services = get_services()
        tenant_scope = _resolve_tenant_scope(_key, tenant_id)
        jobs = _scoped_jobs(services.repo, tenant_scope)
        if source_id:
            jobs = [job for job in jobs if job.source_id == source_id]
        if status:
            jobs = [job for job in jobs if job.status == status]
        jobs.sort(key=_job_sort_key, reverse=True)
        return {"total": len(jobs), "jobs": [_job_item(job) for job in jobs[:limit]]}

    @router.get("/admin/overview")
    async def admin_overview(_key: Optional[str] = Depends(auth_dep)):
        require_admin(_key)
        return build_overview(get_services().repo, None)

    @router.get("/admin/queue/metrics")
    async def admin_queue_metrics(_key: Optional[str] = Depends(auth_dep)):
        require_admin(_key)
        overview = build_overview(get_services().repo, None)
        return {
            "generated_at": overview["generated_at"],
            "queue": overview["queue"],
            "scheduled_sources": overview["sources"]["scheduled"],
            "next_sync_at": overview["sources"]["next_sync_at"],
        }

    return router


def build_overview(repo, tenant_scope: Optional[set[str] | frozenset[str]]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    sources = [source for source in _scoped_sources(repo, tenant_scope) if source.status != "deleted"]
    jobs = _scoped_jobs(repo, tenant_scope)

    by_status: dict[str, int] = {}
    for job in jobs:
        by_status[job.status] = by_status.get(job.status, 0) + 1

    pending = [job for job in jobs if job.status == "pending"]
    running = [job for job in jobs if job.status == "running"]
    failed = [job for job in jobs if job.status == "failed"]
    completed = [job for job in jobs if job.status == "completed"]
    scheduled = [source for source in sources if source.sync_enabled]

    oldest_pending_age = 0.0
    if pending:
        ages = [max(0.0, (now - _parse_time(job.created_at or job.available_at)).total_seconds()) for job in pending]
        oldest_pending_age = max(ages)

    stale_running = sum(
        1 for job in running
        if job.lease_expires_at and _parse_time(job.lease_expires_at) <= now
    )

    completed_24h = sum(
        1 for job in completed
        if job.completed_at and (now - _parse_time(job.completed_at)).total_seconds() <= 86400
    )
    failed_24h = sum(
        1 for job in failed
        if job.completed_at and (now - _parse_time(job.completed_at)).total_seconds() <= 86400
    )

    latest_completed_by_source = {}
    for job in sorted(completed, key=_job_sort_key, reverse=True):
        latest_completed_by_source.setdefault(job.source_id, job)
    indexed_docs = sum(int(job.doc_count or 0) for job in latest_completed_by_source.values())
    indexed_chunks = sum(
        int((job.stats or {}).get("chunks_total", job.chunk_count or 0))
        for job in latest_completed_by_source.values()
    )

    next_syncs = [_parse_time(source.sync_next_at) for source in scheduled if source.sync_next_at]
    return {
        "generated_at": now.isoformat(),
        "sources": {
            "total": len(sources),
            "active": sum(1 for source in sources if source.status == "active"),
            "paused": sum(1 for source in sources if source.status == "paused"),
            "scheduled": len(scheduled),
            "next_sync_at": min(next_syncs).isoformat() if next_syncs else None,
        },
        "queue": {
            "by_status": by_status,
            "pending": len(pending),
            "running": len(running),
            "failed": len(failed),
            "oldest_pending_age_seconds": round(oldest_pending_age, 3),
            "stale_running_leases": stale_running,
            "completed_24h": completed_24h,
            "failed_24h": failed_24h,
        },
        "knowledge": {"documents": indexed_docs, "chunks": indexed_chunks},
        "recent_failures": [_job_item(job) for job in sorted(failed, key=_job_sort_key, reverse=True)[:10]],
    }


def _resolve_tenant_scope(api_key: Optional[str], tenant_id: Optional[str]):
    if tenant_id:
        authorize_tenant(api_key, tenant_id)
        return {tenant_id}
    scope = allowed_tenants(api_key)
    return set(scope) if scope is not None else None


def _scoped_sources(repo, tenant_scope):
    sources = repo.list_sources()
    if tenant_scope is None:
        return sources
    return [source for source in sources if source.tenant_id in tenant_scope]


def _scoped_jobs(repo, tenant_scope):
    jobs = repo.list_jobs()
    if tenant_scope is None:
        return jobs
    return [job for job in jobs if job.tenant_id in tenant_scope]


def _source_catalog_item(source, latest, jobs) -> dict[str, Any]:
    latest_completed = next((job for job in jobs if job.status == "completed"), None)
    return {
        "source_id": source.source_id,
        "tenant_id": source.tenant_id,
        "source_type": source.source_type,
        "name": source.name,
        "status": source.status,
        "tags": list(source.tags),
        "location": _safe_location(source),
        "sync": {
            "enabled": source.sync_enabled,
            "interval_seconds": source.sync_interval_seconds,
            "next_at": _iso(source.sync_next_at),
            "last_enqueued_at": _iso(source.sync_last_enqueued_at),
        },
        "latest_job": _job_item(latest) if latest else None,
        "last_index": {
            "documents": int(latest_completed.doc_count or 0) if latest_completed else 0,
            "chunks": int((latest_completed.stats or {}).get("chunks_total", latest_completed.chunk_count or 0)) if latest_completed else 0,
            "completed_at": _iso(latest_completed.completed_at) if latest_completed else None,
        },
        "created_at": _iso(source.created_at),
        "updated_at": _iso(source.updated_at),
    }


def _job_item(job) -> dict[str, Any]:
    data = asdict(job)
    data.pop("source_config", None)
    for key in ("created_at", "started_at", "completed_at", "available_at", "lease_expires_at", "heartbeat_at"):
        data[key] = _iso(data.get(key))
    return data


def _safe_location(source) -> Optional[str]:
    """Return a useful location without exposing connector credentials/config."""
    config = source.config or {}
    if source.source_type == "web":
        value = config.get("url")
        return str(value) if value else None
    if source.source_type == "s3":
        bucket = config.get("bucket")
        prefix = str(config.get("prefix") or "").strip("/")
        return f"s3://{bucket}/{prefix}" if bucket and prefix else (f"s3://{bucket}" if bucket else None)
    if source.source_type == "gdrive":
        folder_id = config.get("folder_id")
        return f"gdrive://{folder_id}" if folder_id else None
    if source.source_type == "notion":
        page_id = config.get("page_id")
        return f"notion://{page_id}" if page_id else None
    if source.source_type == "confluence":
        base_url = str(config.get("base_url") or "")
        host = urlsplit(base_url).hostname
        space = config.get("space_key")
        return f"confluence://{host}/{space}" if host and space else None
    value = config.get("path")
    return str(value) if value else None


def _job_sort_key(job):
    return (_time_sort_value(job.created_at), job.job_id)


def _time_sort_value(value) -> float:
    try:
        return _parse_time(value).timestamp()
    except Exception:
        return 0.0


def _parse_time(value) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    else:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value):
    if value is None:
        return None
    return _parse_time(value).isoformat()
