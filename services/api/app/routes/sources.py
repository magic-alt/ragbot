from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from services.worker.chunking import resolve_chunking_spec
from services.worker.connectors.credentials import validate_secret_ref
from services.worker.parsing import resolve_parser_spec
from services.worker.pipeline import purge_source_knowledge
from services.worker.scheduler import configure_source_sync
from services.worker.uploads.lifecycle import retire_uploaded_object_for_source

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


SOURCE_TYPE_VALUES = (
    "local_fs", "pdf", "web", "repo", "s3", "gdrive", "notion", "confluence"
)
VALID_SOURCE_TYPES = set(SOURCE_TYPE_VALUES)
_CLOUD_SECRET_SOURCE_TYPES = {"gdrive", "notion", "confluence"}
_PARSER_SOURCE_TYPES = {"local_fs", "pdf", "web", "s3", "gdrive"}
_INLINE_SECRET_MARKERS = (
    "access_token", "refresh_token", "api_key", "apikey", "password",
    "private_key", "client_secret", "secret_access_key",
)


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
    if source_type not in VALID_SOURCE_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid source_type: {source_type}")


def _require_string(config: Dict[str, Any], key: str, source_type: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(
            status_code=422,
            detail=f"source_type={source_type} requires non-empty config.{key}",
        )
    return value.strip()


def _reject_inline_secrets(config: Dict[str, Any]) -> None:
    offending = []
    for key, value in config.items():
        normalized = str(key).strip().lower()
        if normalized == "credential_ref":
            continue
        if any(marker in normalized for marker in _INLINE_SECRET_MARKERS) and value not in (None, ""):
            offending.append(str(key))
    if offending:
        raise HTTPException(
            status_code=422,
            detail=(
                "Cloud connector credentials must not be stored in Source.config; "
                f"use credential_ref=env:VARIABLE instead (inline fields: {', '.join(sorted(offending))})"
            ),
        )


def _validate_chunking_config(source_type: str, config: Dict[str, Any]) -> None:
    raw_chunking = config.get("chunking")
    if raw_chunking is not None and not isinstance(raw_chunking, dict):
        raise HTTPException(status_code=422, detail="config.chunking must be an object")
    default_size = 600 if source_type == "repo" else 800
    default_strategy = "structural" if source_type == "repo" else None
    try:
        resolve_chunking_spec(
            raw_chunking,
            chunk_size=int(config.get("chunk_size", default_size)),
            chunk_overlap=int(config.get("chunk_overlap", 100)),
            default_strategy=default_strategy,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid chunking configuration: {exc}") from exc


def _validate_parsing_config(source_type: str, config: Dict[str, Any]) -> None:
    raw_parsing = config.get("parsing")
    if raw_parsing is None:
        return
    if source_type not in _PARSER_SOURCE_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"source_type={source_type} does not accept config.parsing",
        )
    if not isinstance(raw_parsing, dict):
        raise HTTPException(status_code=422, detail="config.parsing must be an object")
    validation_name = {
        "pdf": "document.pdf",
        "web": "index.html",
        "local_fs": "document.txt",
        "s3": "document.txt",
        "gdrive": "document.txt",
    }[source_type]
    validation_media_type = "application/pdf" if source_type == "pdf" else "application/octet-stream"
    try:
        resolve_parser_spec(
            raw_parsing,
            name=validation_name,
            media_type=validation_media_type,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid parsing configuration: {exc}") from exc


def _validate_source_config(source_type: str, config: Dict[str, Any]) -> None:
    _validate_chunking_config(source_type, config)
    _validate_parsing_config(source_type, config)
    if source_type == "web":
        _require_string(config, "url", source_type)
        return
    if source_type == "s3":
        _require_string(config, "bucket", source_type)
        return
    if source_type == "gdrive":
        _reject_inline_secrets(config)
        _require_string(config, "folder_id", source_type)
        credential_ref = _require_string(config, "credential_ref", source_type)
        try:
            validate_secret_ref(credential_ref)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        credential_type = str(config.get("credential_type", "access_token")).strip().lower()
        if credential_type not in {"access_token", "google_json"}:
            raise HTTPException(status_code=422, detail="gdrive credential_type must be access_token or google_json")
        return
    if source_type == "notion":
        _reject_inline_secrets(config)
        _require_string(config, "page_id", source_type)
        credential_ref = _require_string(config, "credential_ref", source_type)
        try:
            validate_secret_ref(credential_ref)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return
    if source_type == "confluence":
        _reject_inline_secrets(config)
        _require_string(config, "base_url", source_type)
        _require_string(config, "space_key", source_type)
        credential_ref = _require_string(config, "credential_ref", source_type)
        try:
            validate_secret_ref(credential_ref)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        auth_type = str(config.get("auth_type", "basic")).strip().lower()
        if auth_type not in {"basic", "bearer"}:
            raise HTTPException(status_code=422, detail="confluence auth_type must be basic or bearer")
        if auth_type == "basic":
            _require_string(config, "email", source_type)
        return
    _require_string(config, "path", source_type)


def create_sources_router(get_services: Callable, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/sources", tags=["sources"])

    @router.post("", status_code=201)
    async def create_source(payload: CreateSourceRequest, _key: Optional[str] = Depends(auth_dep)):
        _validate_source_type(payload.source_type)
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
        _key: Optional[str] = Depends(auth_dep),
    ):
        require_capability(_key, CAP_CATALOG_READ)
        services = get_services()
        if tenant_id:
            authorize_tenant(_key, tenant_id)
            sources = services.repo.list_sources(tenant_id=tenant_id)
        else:
            tenant_scope = allowed_tenants(_key)
            sources = services.repo.list_sources()
            if tenant_scope is not None:
                sources = [source for source in sources if source.tenant_id in tenant_scope]
        return {"sources": [asdict(source) for source in sources if source.status != "deleted"]}

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