"""Product control-plane APIs for Source catalog and ingestion operations."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query

from ..auth.principal import (
    CAP_CATALOG_READ,
    ROLE_CAPABILITIES,
    allowed_tenants,
    authorize_tenant,
    capabilities_for_principal,
    get_api_principal,
    require_admin,
    require_capability,
)
from ..storage.query_support import ensure_query_repository


def create_control_plane_router(get_services: Callable, auth_dep: Any) -> APIRouter:
    router = APIRouter(tags=["control-plane"])

    @router.get("/catalog/overview")
    async def catalog_overview(
        tenant_id: Optional[str] = None,
        _key: Optional[str] = Depends(auth_dep),
    ):
        require_capability(_key, CAP_CATALOG_READ)
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
        cursor: Optional[str] = Query(default=None),
        _key: Optional[str] = Depends(auth_dep),
    ):
        require_capability(_key, CAP_CATALOG_READ)
        services = get_services()
        repo = ensure_query_repository(services.repo)
        tenant_scope = _resolve_tenant_scope(_key, tenant_id)
        try:
            page = repo.page_sources(
                tenant_ids=tenant_scope,
                status=status,
                source_type=source_type,
                q=q,
                cursor=cursor,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        summaries = repo.source_job_summaries(source.source_id for source in page.items)
        items = []
        for source in page.items:
            summary = summaries.get(source.source_id) or {}
            items.append(
                _source_catalog_item(
                    source,
                    summary.get("latest"),
                    summary.get("latest_completed"),
                )
            )
        return {
            "total": page.total,
            "next_cursor": page.next_cursor,
            "sources": items,
        }

    @router.get("/catalog/jobs")
    async def job_catalog(
        tenant_id: Optional[str] = None,
        source_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = Query(default=100, ge=1, le=500),
        cursor: Optional[str] = Query(default=None),
        _key: Optional[str] = Depends(auth_dep),
    ):
        require_capability(_key, CAP_CATALOG_READ)
        services = get_services()
        repo = ensure_query_repository(services.repo)
        tenant_scope = _resolve_tenant_scope(_key, tenant_id)
        try:
            page = repo.page_jobs(
                tenant_ids=tenant_scope,
                source_id=source_id,
                status=status,
                cursor=cursor,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "total": page.total,
            "next_cursor": page.next_cursor,
            "jobs": [_job_item(job) for job in page.items],
        }

    @router.get("/catalog/documents")
    async def document_catalog(
        tenant_id: Optional[str] = None,
        source_id: Optional[str] = None,
        limit: int = Query(default=100, ge=1, le=500),
        cursor: Optional[str] = Query(default=None),
        _key: Optional[str] = Depends(auth_dep),
    ):
        require_capability(_key, CAP_CATALOG_READ)
        services = get_services()
        repo = ensure_query_repository(services.repo)
        tenant_scope = _resolve_tenant_scope(_key, tenant_id)
        try:
            page = repo.page_documents(
                tenant_ids=tenant_scope,
                source_id=source_id,
                cursor=cursor,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "total": page.total,
            "next_cursor": page.next_cursor,
            "documents": [asdict(document) for document in page.items],
        }

    @router.get("/catalog/generations")
    async def generation_catalog(
        tenant_id: Optional[str] = None,
        source_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = Query(default=100, ge=1, le=500),
        cursor: Optional[str] = Query(default=None),
        _key: Optional[str] = Depends(auth_dep),
    ):
        require_capability(_key, CAP_CATALOG_READ)
        services = get_services()
        repo = ensure_query_repository(services.repo)
        tenant_scope = _resolve_tenant_scope(_key, tenant_id)
        try:
            page = repo.page_generations(
                tenant_ids=tenant_scope,
                source_id=source_id,
                status=status,
                cursor=cursor,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "total": page.total,
            "next_cursor": page.next_cursor,
            "generations": [asdict(generation) for generation in page.items],
        }

    @router.get("/catalog/session")
    async def catalog_session(_key: Optional[str] = Depends(auth_dep)):
        """Return non-secret RBAC capability metadata for the current principal."""
        principal = get_api_principal(_key)
        effective = capabilities_for_principal(principal)
        role_matrix = {role: sorted(values) for role, values in ROLE_CAPABILITIES.items()}
        if principal is None:
            return {
                "principal_mode": "development",
                "admin": True,
                "roles": ["owner"],
                "tenant_ids": [],
                "capabilities": {"read": True, "operate": True, "admin": True},
                "effective_capabilities": sorted(effective),
                "role_capability_matrix": role_matrix,
            }

        roles = sorted({role.strip().lower() for role in principal.roles if role.strip()})
        operate = any(
            capability in effective
            for capability in ("source.create", "source.update", "ingestion.run", "ingestion.retry")
        )
        return {
            "principal_mode": "scoped",
            "admin": principal.admin,
            "roles": roles,
            "tenant_ids": sorted(principal.tenant_ids),
            "capabilities": {
                "read": CAP_CATALOG_READ in effective,
                "operate": operate,
                "admin": principal.admin,
            },
            "effective_capabilities": sorted(effective),
            "role_capability_matrix": role_matrix,
        }

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

    @router.get("/admin/database/metrics")
    async def admin_database_metrics(_key: Optional[str] = Depends(auth_dep)):
        require_admin(_key)
        metrics = getattr(get_services().repo, "database_runtime_metrics", None)
        if not callable(metrics):
            raise HTTPException(
                status_code=501,
                detail="Configured repository does not expose database runtime metrics",
            )
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **metrics(),
        }

    @router.get("/admin/cache", deprecated=True)
    async def retired_cache_status(_key: Optional[str] = Depends(auth_dep)):
        require_admin(_key)
        return {
            "enabled": False,
            "retired": True,
            "reason": (
                "Process-local retrieval/embedding caches are not part of the runtime; "
                "a future cache requires shared generation-aware invalidation."
            ),
            "retrieval": {"supported": False, "runtime_wired": False},
            "embedding": {"supported": False, "runtime_wired": False},
        }

    @router.post("/admin/queue/reconcile")
    async def admin_queue_reconcile(
        max_attempts: int = Query(default=3, ge=1, le=100),
        _key: Optional[str] = Depends(auth_dep),
    ):
        require_admin(_key)
        reconcile = getattr(get_services().repo, "reconcile_ingestion_jobs", None)
        if not callable(reconcile):
            raise HTTPException(status_code=501, detail="Repository does not support queue reconciliation")
        repaired = reconcile(max_attempts=max_attempts)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "max_attempts": max_attempts,
            "repaired": repaired,
        }

    return router


def build_overview(repo, tenant_scope: Optional[set[str] | frozenset[str]]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    bounded = ensure_query_repository(repo)
    raw = bounded.control_plane_overview(tenant_scope)
    sources = raw.get("sources") or {}
    queue = raw.get("queue") or {}
    oldest_pending_at = queue.get("oldest_pending_at")
    oldest_pending_age = 0.0
    if oldest_pending_at:
        oldest_pending_age = max(
            0.0,
            (now - _parse_time(oldest_pending_at)).total_seconds(),
        )
    by_status = {
        key: int(queue.get(key, 0) or 0)
        for key in ("pending", "running", "completed", "failed", "dead_lettered")
        if int(queue.get(key, 0) or 0) > 0
    }
    knowledge = raw.get("knowledge") or {}
    return {
        "generated_at": now.isoformat(),
        "sources": {
            "total": int(sources.get("total", 0) or 0),
            "active": int(sources.get("active", 0) or 0),
            "paused": int(sources.get("paused", 0) or 0),
            "scheduled": int(sources.get("scheduled", 0) or 0),
            "next_sync_at": _iso(sources.get("next_sync_at")),
        },
        "queue": {
            "by_status": by_status,
            "pending": int(queue.get("pending", 0) or 0),
            "running": int(queue.get("running", 0) or 0),
            "failed": int(queue.get("failed", 0) or 0),
            "dead_lettered": int(queue.get("dead_lettered", 0) or 0),
            "oldest_pending_age_seconds": round(oldest_pending_age, 3),
            "stale_running_leases": int(queue.get("stale_running", 0) or 0),
            "completed_24h": int(queue.get("completed_24h", 0) or 0),
            "failed_24h": int(queue.get("failed_24h", 0) or 0),
            "dead_lettered_24h": int(queue.get("dead_lettered_24h", 0) or 0),
        },
        "knowledge": {
            "documents": int(knowledge.get("documents", 0) or 0),
            "chunks": int(knowledge.get("chunks", 0) or 0),
        },
        "recent_failures": [_job_item(job) for job in raw.get("recent_failures") or []],
    }


def _resolve_tenant_scope(api_key: Optional[str], tenant_id: Optional[str]):
    if tenant_id:
        authorize_tenant(api_key, tenant_id)
        return {tenant_id}
    scope = allowed_tenants(api_key)
    return set(scope) if scope is not None else None


def _source_catalog_item(source, latest, latest_completed) -> dict[str, Any]:
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
    for key in (
        "created_at", "started_at", "completed_at", "available_at", "lease_expires_at",
        "heartbeat_at", "dead_lettered_at",
    ):
        data[key] = _iso(data.get(key))
    return data


def _safe_location(source) -> Optional[str]:
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