from __future__ import annotations

from typing import Any, Dict, List

from ...quality.contracts import reranker_contract_id, retrieval_contract_id
from ..state import AgentState, Citation, EvidenceItem, ToolCallRecord, now_ms
from ..reliability import safe_tool_call


async def retrieve_node(state: AgentState, services: Any) -> AgentState:
    filters = _build_filters(state)
    args = {"query": state.query, "top_k": 30, "filters": filters, "plan": "hybrid_rrf"}
    start_ms = now_ms()
    try:
        chunks = await safe_tool_call(
            "retrieve",
            services.retriever.aretrieve,
            state.query,
            filters,
            top_k=30,
            plan="hybrid_rrf",
            deadline_ms=9500,
        )
        if chunks:
            citations = [_chunk_to_citation(chunk) for chunk in chunks[:12]]
            text = _format_chunks(chunks, limit=12)
            state.evidence.append(
                EvidenceItem(
                    kind="doc_chunk",
                    score=1.0,
                    text=text,
                    citations=citations,
                    metadata={"count": len(chunks)},
                )
            )
        record = ToolCallRecord(
            name="retrieve",
            args=args,
            ok=True,
            started_at_ms=start_ms,
            ended_at_ms=now_ms(),
            result_preview={
                "count": len(chunks),
                "quality": _quality_summary(services, chunks),
            },
        )
    except Exception as exc:
        record = ToolCallRecord(
            name="retrieve",
            args=args,
            ok=False,
            started_at_ms=start_ms,
            ended_at_ms=now_ms(),
            error=str(exc),
        )
    state.tool_calls.append(record)
    return state


def _quality_summary(services: Any, chunks: List[Any]) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    retrieved = []
    for chunk in chunks:
        metadata = dict(getattr(chunk, "metadata", None) or {})
        trace = metadata.get("_retrieval") if isinstance(metadata, dict) else None
        if isinstance(trace, dict) and not context:
            raw_context = trace.get("context")
            if isinstance(raw_context, dict):
                context = dict(raw_context)
        retrieved.append(
            {
                "chunk_id": getattr(chunk, "chunk_id", None),
                "doc_id": getattr(chunk, "doc_id", None),
                "score": float(getattr(chunk, "score", 0.0) or 0.0),
                "final_rank": trace.get("final_rank") if isinstance(trace, dict) else None,
            }
        )

    embedder = getattr(services, "embedder", None)
    embedding_id = str(getattr(embedder, "contract_id", "") or "") or None
    repo = getattr(services, "repo", None)
    qdrant = getattr(services, "qdrant", None)
    index_id = None
    alias = getattr(qdrant, "alias_name", None)
    getter = getattr(repo, "get_active_index_version", None)
    if alias and callable(getter):
        try:
            active = getter(alias)
        except Exception:
            active = None
        if active is not None:
            index_id = str(getattr(active, "index_version_id", "") or "") or None
            embedding_id = str(getattr(active, "embedding_contract_id", "") or "") or embedding_id

    return {
        "retrieval_plan": "hybrid_rrf",
        "retrieval_contract_id": retrieval_contract_id(
            plan="hybrid_rrf",
            top_k=30,
            candidate_pool=None,
            rerank=True,
            diversity=False,
        ),
        "embedding_contract_id": embedding_id,
        "index_version_id": index_id,
        "reranker_contract_id": reranker_contract_id(getattr(services, "reranker", None)),
        "retrieved": retrieved,
        "context": context,
    }


def _build_filters(state: AgentState) -> Dict[str, Any]:
    filters: Dict[str, Any] = {"tenant_id": state.tenant_id}
    constraints = state.constraints
    if constraints.source_types:
        filters["source_types"] = constraints.source_types
    if constraints.doc_ids:
        filters["doc_ids"] = constraints.doc_ids
    if constraints.tags:
        filters["tags"] = constraints.tags
    if constraints.path_prefix:
        filters["path_prefix"] = constraints.path_prefix
    if constraints.url_prefix:
        filters["url_prefix"] = constraints.url_prefix
    if constraints.time_from or constraints.time_to:
        filters["time_range"] = {"start": constraints.time_from, "end": constraints.time_to}
    security_scope = constraints.security_scope or {}
    acl_hashes = security_scope.get("acl_hashes") if isinstance(security_scope, dict) else security_scope
    if acl_hashes:
        filters["security_scope"] = acl_hashes
    return filters


def _chunk_to_citation(chunk: Any) -> Citation:
    meta = chunk.metadata or {}
    return Citation(
        kind="chunk",
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        page=meta.get("page"),
        section=meta.get("section"),
        url=meta.get("url"),
        path=meta.get("path"),
        title=meta.get("title"),
    )


def _format_chunks(chunks: List[Any], limit: int = 12) -> str:
    parts: List[str] = []
    for idx, chunk in enumerate(chunks[:limit], start=1):
        text = (chunk.text or "").strip().replace("\n", " ")
        if not text:
            continue
        parts.append(f"[{idx}] {text}")
    return " ".join(parts)
