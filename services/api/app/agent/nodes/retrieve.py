from __future__ import annotations

from typing import Any, Dict, List

from ..state import AgentState, Citation, EvidenceItem, ToolCallRecord, now_ms
from ..reliability import safe_tool_call


def retrieve_node(state: AgentState, services: Any) -> AgentState:
    filters = _build_filters(state)
    args = {"query": state.query, "top_k": 30, "filters": filters}
    start_ms = now_ms()
    try:
        chunks = safe_tool_call("retrieve", services.retriever.retrieve, state.query, filters, top_k=30)
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
            result_preview={"count": len(chunks)},
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

