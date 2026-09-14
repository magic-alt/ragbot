from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


RetrievalPlan = Literal["dense", "lexical", "hybrid_rrf", "qdrant_dense_sparse"]


class ErrorDetail(TypedDict):
    code: str
    message: str
    request_id: str | None
    retryable: bool
    details: Any


class SearchRequest(TypedDict):
    query: str
    tenant_id: str
    user_id: str
    top_k: NotRequired[int]
    plan: NotRequired[RetrievalPlan]
    candidate_pool: NotRequired[int]
    rerank: NotRequired[bool]
    diversity: NotRequired[bool]
    deadline_ms: NotRequired[int]
    explain: NotRequired[bool]
    filters: NotRequired[dict[str, Any]]


class SearchChunk(TypedDict):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    citations: list[str]
    metadata: dict[str, Any]


class SearchResponse(TypedDict):
    request_id: str
    chunks: list[SearchChunk]
    total: int
    diagnostics: dict[str, Any]


class ChatRequest(TypedDict):
    query: str
    tenant_id: str
    user_id: str
    session_id: NotRequired[str]
    constraints: NotRequired[dict[str, Any]]
    client_context: NotRequired[dict[str, Any]]


class ChatResponse(TypedDict):
    request_id: str
    answer: str
    citations: list[dict[str, Any]]
    confidence: str
    followups: list[str]
    debug: NotRequired[dict[str, Any]]


class Source(TypedDict, total=False):
    source_id: str
    tenant_id: str
    source_type: str
    name: str
    config: dict[str, Any]
    acl_policy_id: str | None
    tags: list[str]
    status: str
    created_at: str | None
    updated_at: str | None


class Job(TypedDict, total=False):
    job_id: str
    tenant_id: str
    source_id: str
    source_type: str
    status: str
    created_at: str | None
    started_at: str | None
    completed_at: str | None
    stats: dict[str, Any]


class SourcePage(TypedDict):
    total: int
    next_cursor: str | None
    sources: list[Source]


class JobPage(TypedDict):
    total: int
    next_cursor: str | None
    jobs: list[Job]


class SSEEvent(TypedDict):
    event: str
    data: dict[str, Any]
