from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from services.worker.connectors.registry import connector_registry
from services.worker.pipeline import purge_source_knowledge
from services.worker.scheduler import configure_source_sync
from services.worker.uploads import is_upload_uri
from services.worker.uploads.lifecycle import (
    retire_uploaded_object_for_source,
    source_upload_object_id,
)

from ..auth.principal import (
    CAP_CATALOG_READ,
    CAP_SOURCE_CREATE,
    CAP_SOURCE_DELETE,
    CAP_SOURCE_SYNC,
    CAP_SOURCE_UPDATE,
    allowed_tenants,
    authorize_tenant,
    require_capability,
)
from ..storage.models import Source
from ..storage.query_support import ensure_query_repository


SOURCE_TYPE_VALUES = connector_registry().source_types()
VALID_SOURCE_TYPES = set(SOURCE_TYPE_VALUES)


class CreateSourceRequest(BaseModel):
    tenant_id: str = Field(min_length=1)
    source_type: str = Field(json_schema_extra={"enum": list(SOURCE_TYPE_VALUES)})
    name: str = Field(min_length=1)
    config: Dict[str, Any] = Field(default_factory=dict)
    acl_policy_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class UpdateSourceRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1)
    config: Optional[Dict[str, Any]] = None
    status: Optional[Literal["active", "paused"]] = None
    acl_policy_id: Optional[str] = None
    tags: Optional[List[str]] = None


class SourceSyncRequest(BaseModel):
    enabled: bool
    interval_seconds: Optional[int] = Field(default=None, ge=60)
    run_immediately: bool = False


def _validate_source_type(source_type: str) -> None:
    try:
        connector_registry().get(source_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid source_type: {source_type}") from exc


def _reject_direct_managed_upload_config(config: Dict[str, Any]) -> None:
    path = config.get("path")
    if isinstance(path, str) and is_upload_uri(path):
        raise HTTPException(
            status_code=422,
            detail=(
                "ragbot-upload URIs are server-managed; create uploaded PDF Sources through "
                "/ingest/upload/pdf instead of /sources"
            ),
        )


def _validate_source_config(source_type: str, config: Dict[str, Any]) -> None:
    try:
        connector_registry().validate_source_config(source_type, config)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def create_sources_router(get_services: Callable, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/sources", tags=["sources"])

    @router.post("", status_code=201)
    async def create_source(payload: CreateSourceRequest, _key: Optional[str] = Depends(auth_dep)):
        _validate_source_type(payload.source_type)
        _reject_direct_managed_upload_config(payload.config)
        _validate_source_config(payload.source_type, payload.config)
        authorize_tenant(_key, payload.tenant_id)
        require_capability(_key, CAP_SOURCE_CREATE)
        services = get_services()
        now = datetime.now(timezone.utc).isoformat()
        source = Source(
            source_id=uuid.uuid4().hex,
            tenant_id=payload.tenant_id,
            source_type=payload.source_type,
            name=payload.name,
            config=payload.config,
            acl_policy_id=payload.acl_policy_id,
            tags=payload.tags,
            created_at=now,
            updated_at=now,
        )
        services.repo.add_source(source)
        return asdict(source)

    @router.get("")
    async def list_sources(
        tenant_id: Optional[str] = None,
        limit: int = Query(default=100, ge=1, le=500),
        cursor: Optional[str] = Query(default=None),
        _key: Optional[str] = Depends(auth_dep),
    ):
        require_capability(_key, CAP_CATALOG_READ)
        services = get_services()
        repo = ensure_query_repository(services.repo)
        if tenant_id:
            authorize_tenant(_key, tenant_id)
            tenant_scope = {tenant_id}
        else:
            allowed = allowed_tenants(_key)
            tenant_scope = set(allowed) if allowed is not None else None
        try:
            page = repo.page_sources(
                tenant_ids=tenant_scope,
                cursor=cursor,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "total": page.total,
            "next_cursor": page.next_cursor,
            "sources": [asdict(source) for source in page.items],
        }

    @router.get("/{source_id}")
    async def get_source(source_id: str, _key: Optional[str] = Depends(auth_dep)):
        services = get_services()
        source = services.repo.get_source(source_id)
        if not source or source.status == "deleted":
            raise HTTPException(404, "Source not found")
        authorize_tenant(_key, source.tenant_id)
        require_capability(_key, CAP_CATALOG_READ)
        return asdict(source)

    @router.put("/{source_id}")
    async def update_source(
        source_id: str,
        payload: UpdateSourceRequest,
        _key: Optional[str] = Depends(auth_dep),
    ):
        services = get_services()
        source = services.repo.get_source(source_id)
        if not source or source.status == "deleted":
            raise HTTPException(404, "Source not found")
        authorize_tenant(_key, source.tenant_id)
        require_capability(_key, CAP_SOURCE_UPDATE)
        if payload.config is not None:
            managed_object_id = source_upload_object_id(source)
            if managed_object_id is not None and dict(payload.config) != dict(source.config or {}):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Server-managed uploaded Source content is immutable; upload a new document "
                        "instead of replacing config/path on the existing Source"
                    ),
                )
            if managed_object_id is None:
                _reject_direct_managed_upload_config(payload.config)
            _validate_source_config(source.source_type, payload.config)
        updates = {
            key: value
            for key, value in payload.model_dump(exclude_unset=True).items()
            if value is not None
        }
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        updated = services.repo.update_source(source_id, **updates)
        return asdict(updated)

    @router.put("/{source_id}/sync")
    async def update_source_sync(
        source_id: str,
        payload: SourceSyncRequest,
        _key: Optional[str] = Depends(auth_dep),
    ):
        services = get_services()
        source = services.repo.get_source(source_id)
        if not source or source.status == "deleted":
            raise HTTPException(404, "Source not found")
        authorize_tenant(_key, source.tenant_id)
        require_capability(_key, CAP_SOURCE_SYNC)
        if payload.enabled and payload.interval_seconds is None:
            raise HTTPException(status_code=422, detail="enabled sync requires interval_seconds")
        try:
            updated = configure_source_sync(
                services.repo,
                source,
                enabled=payload.enabled,
                interval_seconds=payload.interval_seconds,
                run_immediately=payload.run_immediately,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return asdict(updated)

    @router.delete("/{source_id}", status_code=204)
    async def delete_source(source_id: str, _key: Optional[str] = Depends(auth_dep)):
        services = get_services()
        source = services.repo.get_source(source_id)
        if not source or source.status == "deleted":
            raise HTTPException(404, "Source not found")
        authorize_tenant(_key, source.tenant_id)
        require_capability(_key, CAP_SOURCE_DELETE)

        tombstoned = services.repo.update_source(
            source_id,
            status="deleted",
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        if tombstoned is None:
            raise HTTPException(404, "Source not found")
        retire_uploaded_object_for_source(services.repo, tombstoned)
        purge_source_knowledge(tombstoned, services.repo, services.qdrant)
        return None

    return router