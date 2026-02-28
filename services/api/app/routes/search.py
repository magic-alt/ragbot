from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

router = APIRouter(tags=["search"])


class SearchFilters(BaseModel):
    source_types: Optional[List[str]] = None
    doc_ids: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    path_prefix: Optional[str] = None
    url_prefix: Optional[str] = None
    time_from: Optional[str] = None
    time_to: Optional[str] = None
    model_config = {"extra": "forbid"}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    tenant_id: str
    user_id: str
    top_k: int = Field(default=20, ge=1, le=100)
    filters: Optional[SearchFilters] = None


class ChunkResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    citations: List[str] = []
    metadata: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    request_id: str
    chunks: List[ChunkResult]
    total: int


def _build_retrieval_filters(
    tenant_id: str,
    user_id: str,
    filters: Optional[SearchFilters],
    services: Any,
) -> Dict[str, Any]:
    from ..auth.acl import compute_security_scope

    result: Dict[str, Any] = {"tenant_id": tenant_id}
    policies = services.repo.list_policies(tenant_id)
    acl_hashes = compute_security_scope(user_id, policies)
    if acl_hashes:
        result["security_scope"] = acl_hashes
    if not filters:
        return result
    if filters.source_types:
        result["source_types"] = filters.source_types
    if filters.doc_ids:
        result["doc_ids"] = filters.doc_ids
    if filters.tags:
        result["tags"] = filters.tags
    if filters.path_prefix:
        result["path_prefix"] = filters.path_prefix
    if filters.url_prefix:
        result["url_prefix"] = filters.url_prefix
    if filters.time_from or filters.time_to:
        result["time_range"] = {"start": filters.time_from, "end": filters.time_to}
    return result


def create_search_endpoint(get_services, verify_api_key):
    """Register the /search endpoint on the router."""

    @router.post("/search", response_model=SearchResponse)
    async def search_endpoint(
        payload: SearchRequest,
        _key: str = Depends(verify_api_key),
    ) -> SearchResponse:
        services = get_services()
        retrieval_filters = _build_retrieval_filters(
            payload.tenant_id,
            payload.user_id,
            payload.filters,
            services,
        )
        chunks = services.retriever.retrieve(
            payload.query,
            retrieval_filters,
            top_k=payload.top_k,
        )
        chunk_results = [
            ChunkResult(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                text=c.text,
                score=c.score,
                citations=c.citations if c.citations else [],
                metadata=c.metadata if c.metadata else {},
            )
            for c in chunks
        ]
        return SearchResponse(
            request_id=uuid.uuid4().hex,
            chunks=chunk_results,
            total=len(chunk_results),
        )

    return router
