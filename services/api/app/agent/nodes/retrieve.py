from __future__ import annotations

from typing import Any, Dict

from contracts.types import AgentState


def retrieve_node(state: AgentState, services: Any) -> AgentState:
    filters: Dict[str, Any] = {"tenant_id": state.tenant_id}
    constraints = state.constraints or {}
    if constraints.get("source_type"):
        filters["source_types"] = [constraints["source_type"]]
    if constraints.get("repo"):
        filters["path_prefix"] = constraints["repo"]
    filters["security_scope"] = services.security_scope(state.user_id, state.tenant_id)

    params = {"query": state.query, "top_k": 20, "filters": filters}
    state.add_tool_call("retrieve", params)
    chunks = services.retriever.retrieve(state.query, filters, top_k=20)
    for chunk in chunks:
        payload = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "score": chunk.score,
            "metadata": chunk.metadata,
        }
        state.add_evidence("chunk", payload, chunk.citations)
    return state

