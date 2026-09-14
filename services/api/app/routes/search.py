from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth.acl import compute_security_scope
from ..auth.principal import CAP_KNOWLEDGE_QUERY, authorize_identity, require_capability
from ..quality.recorder import QualityRecorder
from ..retrieval.contracts import (
    RetrievalDeadlineExceeded,
    RetrievalPlan,
    RetrievalRequest,
    UnsupportedRetrievalPlan,
    resolve_retrieval_plan,
)

logger = logging.getLogger(__name__)
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
    diagnostics: Dict[str, Any]


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
        # The request ID exists before any retrieval work starts, so production
        # errors/timeouts can still be correlated with one durable lineage row.
        request_id = uuid.uuid4().hex
        started_monotonic = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        recorder = QualityRecorder(services.repo)
        try:
            retrieval = await services.retriever.query(request)
        except RetrievalDeadlineExceeded as exc:
            _record_quality_safe(
                recorder,
                services=services,
                request_id=request_id,
                tenant_id=payload.tenant_id,
                user_id=trusted_user_id,
                query=payload.query,
                request=request,
                response=SimpleNamespace(trace=exc.trace, chunks=[]),
                status="deadline_exceeded",
                started_monotonic=started_monotonic,
                started_at=started_at,
                error=exc,
            )
            raise HTTPException(
                status_code=504,
                detail={
                    "request_id": request_id,
                    "message": str(exc),
                    "retrieval": exc.trace.as_dict(),
                },
            ) from exc
        except UnsupportedRetrievalPlan as exc:
            _record_quality_safe(
                recorder,
                services=services,
                request_id=request_id,
                tenant_id=payload.tenant_id,
                user_id=trusted_user_id,
                query=payload.query,
                request=request,
                status="failed",
                started_monotonic=started_monotonic,
                started_at=started_at,
                error=exc,
            )
            raise HTTPException(
                status_code=409,
                detail={"request_id": request_id, "message": str(exc)},
            ) from exc
        except asyncio.CancelledError as exc:
            _record_quality_safe(
                recorder,
                services=services,
                request_id=request_id,
                tenant_id=payload.tenant_id,
                user_id=trusted_user_id,
                query=payload.query,
                request=request,
                status="cancelled",
                started_monotonic=started_monotonic,
                started_at=started_at,
                error=exc,
            )
            raise
        except Exception as exc:
            _record_quality_safe(
                recorder,
                services=services,
                request_id=request_id,
                tenant_id=payload.tenant_id,
                user_id=trusted_user_id,
                query=payload.query,
                request=request,
                status="failed",
                started_monotonic=started_monotonic,
                started_at=started_at,
                error=exc,
            )
            raise

        _record_quality_safe(
            recorder,
            services=services,
            request_id=request_id,
            tenant_id=payload.tenant_id,
            user_id=trusted_user_id,
            query=payload.query,
            request=request,
            response=retrieval,
            status="completed",
            started_monotonic=started_monotonic,
            started_at=started_at,
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
            request_id=request_id,
            chunks=chunk_results,
            total=len(chunk_results),
            diagnostics=diagnostics,
        )

    return router


def _record_quality_safe(recorder: QualityRecorder, **kwargs: Any) -> None:
    try:
        recorder.record_search(**kwargs)
    except Exception:
        # Observability is fail-open for serving traffic. The request ID still
        # remains available to logs/HTTP responses so storage outages are
        # diagnosable without converting a successful search into an outage.
        logger.exception("Failed to persist durable search quality record")
