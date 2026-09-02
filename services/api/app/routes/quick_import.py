"""High-level product API for creating/reusing sources and starting ingestion.

The lower-level ``/sources`` + ``/ingest/jobs`` APIs remain available for
advanced orchestration. Quick import collapses the common product workflow into
one idempotent call and provides a batch surface suitable for manifests.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional
from urllib.parse import urlsplit, urlunsplit

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth.principal import authorize_tenant
from ..storage.models import Source
from .ingest import assert_source_ingestible, enqueue_ingestion_job, latest_active_ingestion_job
from .sources import _validate_source_config, _validate_source_type

SourceType = Literal["local_fs", "pdf", "web", "repo"]


class QuickSourceSpec(BaseModel):
    location: str = Field(min_length=1)
    source_type: Optional[SourceType] = None
    name: Optional[str] = Field(default=None, min_length=1)
    tags: Optional[List[str]] = None
    acl_policy_id: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    reuse_source: bool = True
    sync_source_metadata: bool = True
    dedupe_active_job: bool = True
    idempotency_key: Optional[str] = Field(default=None, min_length=1, max_length=200)


class QuickIngestRequest(QuickSourceSpec):
    tenant_id: str = Field(min_length=1)


class BatchQuickIngestRequest(BaseModel):
    tenant_id: str = Field(min_length=1)
    sources: List[QuickSourceSpec] = Field(min_length=1, max_length=100)


def infer_source_type(location: str) -> SourceType:
    """Infer the connector from a local path or remote URL."""
    value = location.strip()
    parsed = urlsplit(value)
    if parsed.scheme.lower() in {"http", "https"}:
        host = (parsed.hostname or "").lower()
        path = parsed.path.lower().rstrip("/")
        if path.endswith(".pdf"):
            return "pdf"
        if path.endswith(".git") or host in {"github.com", "gitlab.com", "bitbucket.org"}:
            return "repo"
        return "web"

    lowered = value.lower().rstrip("/\\")
    if lowered.endswith(".pdf"):
        return "pdf"
    if lowered.endswith(".git"):
        return "repo"
    return "local_fs"


def canonical_location(location: str) -> str:
    """Normalize locations enough for stable source identity without resolving I/O."""
    value = location.strip()
    parsed = urlsplit(value)
    if parsed.scheme.lower() in {"http", "https"}:
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/") or "/"
        return urlunsplit((scheme, netloc, path, parsed.query, ""))
    normalized = value.rstrip("/\\")
    return normalized or value


def build_source_config(source_type: SourceType, location: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = dict(extra or {})
    key = "url" if source_type == "web" else "path"
    config[key] = location.strip()
    return config


def deterministic_source_id(tenant_id: str, source_type: str, location: str) -> str:
    identity = f"{tenant_id}\0{source_type}\0{canonical_location(location)}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def deterministic_job_id(source_id: str, idempotency_key: str) -> str:
    identity = f"quick-ingest\0{source_id}\0{idempotency_key}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def _source_location(source: Source) -> Optional[str]:
    key = "url" if source.source_type == "web" else "path"
    value = source.config.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _find_existing_source(repo, *, tenant_id: str, source_type: str, location: str) -> Optional[Source]:
    stable_id = deterministic_source_id(tenant_id, source_type, location)
    direct = repo.get_source(stable_id)
    if direct and direct.tenant_id == tenant_id and direct.source_type == source_type:
        return direct

    target = canonical_location(location)
    for source in repo.list_sources(tenant_id=tenant_id):
        if source.source_type != source_type:
            continue
        existing_location = _source_location(source)
        if existing_location and canonical_location(existing_location) == target:
            return source
    return None


def _upsert_source(repo, *, tenant_id: str, spec: QuickSourceSpec, source_type: SourceType, config: Dict[str, Any]) -> tuple[Source, bool]:
    existing = _find_existing_source(
        repo,
        tenant_id=tenant_id,
        source_type=source_type,
        location=spec.location,
    ) if spec.reuse_source else None

    now = datetime.now(timezone.utc).isoformat()
    if existing is not None:
        if existing.status == "deleted":
            restored = Source(
                source_id=existing.source_id,
                tenant_id=tenant_id,
                source_type=source_type,
                name=spec.name or existing.name or spec.location,
                config=config,
                status="active",
                acl_policy_id=spec.acl_policy_id if spec.acl_policy_id is not None else existing.acl_policy_id,
                tags=spec.tags if spec.tags is not None else existing.tags,
                created_at=existing.created_at or now,
                updated_at=now,
            )
            repo.add_source(restored)
            return restored, True

        if spec.sync_source_metadata:
            updates: Dict[str, Any] = {"config": config, "updated_at": now}
            if spec.name is not None:
                updates["name"] = spec.name
            if spec.tags is not None:
                updates["tags"] = spec.tags
            if spec.acl_policy_id is not None:
                updates["acl_policy_id"] = spec.acl_policy_id
            existing = repo.update_source(existing.source_id, **updates) or existing
        return existing, True

    source = Source(
        source_id=(
            deterministic_source_id(tenant_id, source_type, spec.location)
            if spec.reuse_source
            else uuid.uuid4().hex
        ),
        tenant_id=tenant_id,
        source_type=source_type,
        name=spec.name or spec.location,
        config=config,
        status="active",
        acl_policy_id=spec.acl_policy_id,
        tags=spec.tags or [],
        created_at=now,
        updated_at=now,
    )
    repo.add_source(source)
    return source, False


def _run_quick_import(*, tenant_id: str, spec: QuickSourceSpec, services) -> Dict[str, Any]:
    source_type = spec.source_type or infer_source_type(spec.location)
    _validate_source_type(source_type)
    config = build_source_config(source_type, spec.location, spec.config)
    _validate_source_config(source_type, config)

    source, source_reused = _upsert_source(
        services.repo,
        tenant_id=tenant_id,
        spec=spec,
        source_type=source_type,
        config=config,
    )
    assert_source_ingestible(source)

    if spec.idempotency_key:
        job_id = deterministic_job_id(source.source_id, spec.idempotency_key)
        existing_job = services.repo.get_job(job_id)
        if existing_job is not None:
            if existing_job.tenant_id != tenant_id or existing_job.source_id != source.source_id:
                raise HTTPException(status_code=409, detail="Idempotency key collision")
            return {
                "status": "idempotent_replay",
                "source_id": source.source_id,
                "source_type": source.source_type,
                "source_reused": source_reused,
                "job_id": existing_job.job_id,
                "job_status": existing_job.status,
                "job_reused": True,
            }
    else:
        job_id = None

    if spec.dedupe_active_job:
        active_job = latest_active_ingestion_job(
            services.repo,
            tenant_id=tenant_id,
            source_id=source.source_id,
        )
        if active_job is not None:
            return {
                "status": "already_queued",
                "source_id": source.source_id,
                "source_type": source.source_type,
                "source_reused": source_reused,
                "job_id": active_job.job_id,
                "job_status": active_job.status,
                "job_reused": True,
            }

    job = enqueue_ingestion_job(source, services, job_id=job_id)
    return {
        "status": "accepted",
        "source_id": source.source_id,
        "source_type": source.source_type,
        "source_reused": source_reused,
        "job_id": job.job_id,
        "job_status": job.status,
        "job_reused": False,
    }


def create_quick_import_router(get_services: Callable, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/ingest", tags=["ingest"])

    @router.post("/quick", status_code=202)
    async def quick_import(
        payload: QuickIngestRequest,
        _key: Optional[str] = Depends(auth_dep),
    ):
        authorize_tenant(_key, payload.tenant_id)
        services = get_services()
        spec = QuickSourceSpec(**payload.model_dump(exclude={"tenant_id"}))
        return _run_quick_import(tenant_id=payload.tenant_id, spec=spec, services=services)

    @router.post("/batch", status_code=202)
    async def batch_import(
        payload: BatchQuickIngestRequest,
        _key: Optional[str] = Depends(auth_dep),
    ):
        authorize_tenant(_key, payload.tenant_id)
        services = get_services()
        items = []
        failed = 0
        for spec in payload.sources:
            try:
                result = _run_quick_import(tenant_id=payload.tenant_id, spec=spec, services=services)
                items.append({"location": spec.location, **result})
            except HTTPException as exc:
                failed += 1
                items.append({
                    "location": spec.location,
                    "status": "error",
                    "status_code": exc.status_code,
                    "error": exc.detail,
                })
            except Exception as exc:
                failed += 1
                items.append({
                    "location": spec.location,
                    "status": "error",
                    "status_code": 500,
                    "error": str(exc),
                })
        return {
            "tenant_id": payload.tenant_id,
            "total": len(items),
            "accepted": len(items) - failed,
            "failed": failed,
            "items": items,
        }

    return router
