from __future__ import annotations

from typing import List

from ..state import AgentState, Verification, ROUTE_CODE, ROUTE_DOC_RAG, ROUTE_MIXED, ROUTE_SQL


def verify_node(state: AgentState, services: object) -> AgentState:
    missing: List[str] = []
    if state.route == ROUTE_SQL:
        rows_evidence = [ev for ev in state.evidence if ev.kind == "sql_rows"]
        if not rows_evidence or rows_evidence[0].metadata.get("row_count", 0) == 0:
            missing.append("sql_rows")
    elif state.route == ROUTE_CODE:
        code_evidence = [ev for ev in state.evidence if ev.kind == "code_snippets"]
        if not code_evidence:
            missing.append("code_snippets")
    elif state.route in (ROUTE_DOC_RAG, ROUTE_MIXED):
        chunk_evidence = [ev for ev in state.evidence if ev.kind == "doc_chunk"]
        if not chunk_evidence:
            missing.append("doc_chunks")

    enough = len(missing) == 0
    next_action = None
    if not enough:
        if "doc_chunks" in missing:
            next_action = "retrieve"
        elif "sql_rows" in missing:
            next_action = "sql_query"
        elif "code_snippets" in missing:
            next_action = "code_search"

    state.verification = Verification(
        enough_evidence=enough,
        missing=missing,
        next_query=state.query,
        next_action=next_action,
    )
    return state

