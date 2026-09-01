from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from contracts.types import SourceType
from services.worker.pipeline import purge_source_knowledge

from ..storage.models import Source


class CreateSourceRequest(BaseModel):
    tenant_id: str = Field(min_length=1)
    source_type: SourceType
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


VALID_SOURCE_TYPES = {"local_fs", "pdf", "web", "repo"}


def _validate_source_config(source_type: str, config: Dict[str, Any]) -> None:
    required_key = "url" if source_type == "web" else "path"
    value = config.get(required_key)
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(
            status_code=422,
            detail=f"source_type={source_type} requires non-empty config.{required_key}",
        )


def create_sources_router(get_services: Callable, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/sources", tags=["sources"])

    @router.post("", status_code=201)
    async def create_source(payload: CreateSourceRequest, _key=Depends(auth_dep)):
        _validate_source_config(payload.source_type, payload.config)
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
    async def list_sources(tenant_id: Optional[str] = None, _key=Depends(auth_dep)):
        services = get_services()
        sources = services.repo.list_sources(tenant_id=tenant_id)
        return {"sources": [asdict(s) for s in sources if s.status != "deleted"]}

    @router.get("/{source_id}")
    async def get_source(source_id: str, _key=Depends(auth_dep)):
        services = get_services()
        source = services.repo.get_source(source_id)
        if not source or source.status == "deleted":
            raise HTTPException(404, "Source not found")
        return asdict(source)

    @router.put("/{source_id}")
    async def update_source(source_id: str, payload: UpdateSourceRequest, _key=Depends(auth_dep)):
        services = get_services()
        source = services.repo.get_source(source_id)
        if not source or source.status == "deleted":
            raise HTTPException(404, "Source not found")
        if payload.config is not None:
            _validate_source_config(source.source_type, payload.config)
        updates = {
            key: value
            for key, value in payload.model_dump(exclude_unset=True).items()
            if value is not None
        }
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        updated = services.repo.update_source(source_id, **updates)
        return asdict(updated)

    @router.delete("/{source_id}", status_code=204)
    async def delete_source(source_id: str, _key=Depends(auth_dep)):
        services = get_services()
        source = services.repo.get_source(source_id)
        if not source or source.status == "deleted":
            raise HTTPException(404, "Source not found")

        # Purge indexed knowledge before tombstoning the Source. If vector or
        # repository cleanup fails, the source remains active and retryable.
        purge_source_knowledge(source, services.repo, services.qdrant)
        if not services.repo.delete_source(source_id):
            raise HTTPException(404, "Source not found")
        return None

    return router
