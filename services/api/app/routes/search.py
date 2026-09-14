from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth.acl import compute_security_scope
from ..auth.principal import CAP_KNOWLEDGE_QUERY, authorize_identity, require_capability
from ..retrieval.contracts import (
    RetrievalDeadlineExceeded,
    RetrievalPlan,
    RetrievalRequest,
    UnsupportedRetrievalPlan,
    resolve_retrieval_plan,
)

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
    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    top_k: int = Field(default=20, ge=1, le=100)
    # Backward-compatible ablation name. `plan` is the stable #57 contract.
    mode: Literal["vector", "lexical", "hybrid"] = "hybrid"
    plan: Optional[Literal["dense", "lexical", "hybrid_rrf", "qdrant_dense_sparse"]] = None
    candidate_pool: Optional[int] = Field(default=None, ge=1, le=200)
    rerank: bool = True
    diversity: bool = False
    deadline_ms: Optional[int] = Field(default=None, ge=1, le=120000)
    explain: bool = False
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
    diagnostics: Dict[str, Any] = {}


def _build_retrieval_filters(
    tenant_id: str,
    user_id: str,
    groups: tuple[str, ...],
    roles: tuple[str, ...],
    filters: Optional[SearchFilters],
    services: Any,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"tenant_id": tenant_id}
    policies = services.repo.list_policies(tenant_id)
    acl_hashes = compute_security_scope(
        user_id,
        policies,
        groups=list(groups),
        roles=list(roles),
    )
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
    @router.post("/search", response_model=SearchResponse)
    async def search_endpoint(
        payload: SearchRequest,
        _key: Optional[str] = Depends(verify_api_key),
    ) -> SearchResponse:
        require_capability(_key, CAP_KNOWLEDGE_QUERY)
        services = get_services()
        trusted_user_id, groups, roles = authorize_identity(
            _key, payload.tenant_id, payload.user_id
        )
        retrieval_filters = _build_retrieval_filters(
            payload.tenant_id,
            trusted_user_id,
            groups,
            roles,
            payload.filters,
            services,
        )
        plan = resolve_retrieval_plan(payload.plan, legacy_mode=payload.mode)
        request = RetrievalRequest(
            query=payload.query,
            filters=retrieval_filters,
            top_k=payload.top_k,
            plan=plan,
            candidate_pool=payload.candidate_pool,
            rerank=payload.rerank,
            deadline_ms=payload.deadline_ms,
            diversity=payload.diversity,
        )
        try:
            retrieval = await services.retriever.query(request)
        except RetrievalDeadlineExceeded as exc:
            raise HTTPException(
                status_code=504,
                detail={"message": str(exc), "retrieval": exc.trace.as_dict()},
            ) from exc
        except UnsupportedRetrievalPlan as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        chunk_results = [
            ChunkResult(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                text=c.text,
                score=c.score,
                citations=c.citations if c.citations else [],
                metadata=c.metadata if c.metadata else {},
            )
            for c in retrieval.chunks
        ]
        diagnostics: Dict[str, Any] = {
            "retrieval_mode": payload.mode,
            "retrieval_plan": plan.value,
            "requested_candidate_pool": payload.candidate_pool,
            "reranker_requested": payload.rerank,
            "diversity_requested": payload.diversity,
            "deadline_ms": payload.deadline_ms,
            "explain": payload.explain,
            **retrieval.trace.as_dict(),
        }
        describe = getattr(services.retriever, "diagnostics", None)
        if callable(describe):
            diagnostics = describe(payload.query, diagnostics)
        return SearchResponse(
            request_id=uuid.uuid4().hex,
            chunks=chunk_results,
            total=len(chunk_results),
            diagnostics=diagnostics,
        )

    return router
