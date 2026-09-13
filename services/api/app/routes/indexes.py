from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth.principal import require_admin


class CreateIndexRequest(BaseModel):
    embedding_contract_id: str = Field(min_length=1)
    index_version_id: Optional[str] = None
    physical_collection: Optional[str] = None


class ShadowRequest(BaseModel):
    cases: list[dict[str, Any]] = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] = Field(default_factory=dict)


class ReadyRequest(BaseModel):
    evidence: dict[str, Any]
    approved: bool = False


class ActivationRequest(BaseModel):
    retention_seconds: int = Field(default=7 * 24 * 3600, ge=0)


def create_indexes_router(
    get_services: Callable,
    verify_api_key: Callable,
) -> APIRouter:
    router = APIRouter(prefix="/admin/indexes", tags=["index-lifecycle"])

    def lifecycle(_key: Optional[str]):
        require_admin(_key)
        service = getattr(get_services(), "index_lifecycle", None)
        if service is None:
            raise HTTPException(
                status_code=409,
                detail="Index lifecycle is unavailable for the configured vector/repository backend",
            )
        return service

    @router.get("")
    def list_indexes(_key: Optional[str] = Depends(verify_api_key)) -> dict[str, Any]:
        service = lifecycle(_key)
        versions = service.repo.list_index_versions(alias_name=service.alias_name)
        active_physical = service.vector_store.active_collection_name()
        return {
            "alias_name": service.alias_name,
            "active_physical_collection": active_physical,
            "items": [asdict(item) for item in versions],
        }

    @router.get("/contracts")
    def list_embedding_contracts(
        _key: Optional[str] = Depends(verify_api_key),
    ) -> dict[str, Any]:
        service = lifecycle(_key)
        return {"items": service.embedding_router.public_metadata()}

    @router.post("")
    def create_index(
        payload: CreateIndexRequest,
        _key: Optional[str] = Depends(verify_api_key),
    ) -> dict[str, Any]:
        service = lifecycle(_key)
        try:
            version = service.create_candidate(
                payload.embedding_contract_id,
                index_version_id=payload.index_version_id,
                physical_collection=payload.physical_collection,
            )
            return asdict(version)
        except (KeyError, ValueError, RuntimeError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/{index_version_id}")
    def get_index(
        index_version_id: str,
        _key: Optional[str] = Depends(verify_api_key),
    ) -> dict[str, Any]:
        service = lifecycle(_key)
        version = service.repo.get_index_version(index_version_id)
        if version is None:
            raise HTTPException(status_code=404, detail="IndexVersion not found")
        return asdict(version)

    @router.post("/{index_version_id}/shadow")
    def shadow_index(
        index_version_id: str,
        payload: ShadowRequest,
        _key: Optional[str] = Depends(verify_api_key),
    ) -> dict[str, Any]:
        service = lifecycle(_key)
        try:
            return service.shadow_compare(
                index_version_id,
                payload.cases,
                top_k=payload.top_k,
                filters=payload.filters,
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/{index_version_id}/ready")
    def mark_ready(
        index_version_id: str,
        payload: ReadyRequest,
        _key: Optional[str] = Depends(verify_api_key),
    ) -> dict[str, Any]:
        service = lifecycle(_key)
        try:
            return asdict(
                service.mark_ready(
                    index_version_id,
                    payload.evidence,
                    approved=payload.approved,
                )
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/{index_version_id}/activate")
    def activate_index(
        index_version_id: str,
        payload: ActivationRequest,
        _key: Optional[str] = Depends(verify_api_key),
    ) -> dict[str, Any]:
        service = lifecycle(_key)
        try:
            return asdict(
                service.activate(
                    index_version_id, retention_seconds=payload.retention_seconds
                )
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/{index_version_id}/rollback")
    def rollback_index(
        index_version_id: str,
        payload: ActivationRequest,
        _key: Optional[str] = Depends(verify_api_key),
    ) -> dict[str, Any]:
        service = lifecycle(_key)
        try:
            return asdict(
                service.rollback(
                    index_version_id, retention_seconds=payload.retention_seconds
                )
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/reconcile/run")
    def reconcile_indexes(
        _key: Optional[str] = Depends(verify_api_key),
    ) -> dict[str, Any]:
        service = lifecycle(_key)
        try:
            return service.reconcile()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/retention/prune")
    def prune_indexes(
        _key: Optional[str] = Depends(verify_api_key),
    ) -> dict[str, Any]:
        service = lifecycle(_key)
        return service.prune_retired()

    return router
