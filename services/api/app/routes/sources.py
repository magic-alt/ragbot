from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..storage.models import Source


class CreateSourceRequest(BaseModel):
    tenant_id: str
    source_type: str = Field(description="local_fs | pdf | web | repo | email | database")
    name: str
    config: Dict[str, Any] = Field(default_factory=dict)
    acl_policy_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class UpdateSourceRequest(BaseModel):
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    acl_policy_id: Optional[str] = None
    tags: Optional[List[str]] = None


VALID_SOURCE_TYPES = {"local_fs", "pdf", "web", "repo", "email", "database"}


def create_sources_router(get_services: Callable, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/sources", tags=["sources"])

    @router.post("", status_code=201)
    async def create_source(payload: CreateSourceRequest, _key=Depends(auth_dep)):
        if payload.source_type not in VALID_SOURCE_TYPES:
            raise HTTPException(400, f"Invalid source_type: {payload.source_type}")
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
        updates = {k: v for k, v in payload.model_dump(exclude_unset=True).items() if v is not None}
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        updated = services.repo.update_source(source_id, **updates)
        return asdict(updated)

    @router.delete("/{source_id}", status_code=204)
    async def delete_source(source_id: str, _key=Depends(auth_dep)):
        services = get_services()
        if not services.repo.delete_source(source_id):
            raise HTTPException(404, "Source not found")
        return None

    return router
