from __future__ import annotations

import logging
from typing import List

from ..state import AgentState, Verification, ROUTE_CODE, ROUTE_DOC_RAG, ROUTE_MIXED, ROUTE_SQL

logger = logging.getLogger(__name__)


def verify_node(state: AgentState, services: object) -> AgentState:
    llm = getattr(services, "llm", None)
    if llm and getattr(llm, "enabled", False):
        try:
            state.verification = _llm_verify(state, llm)
            return state
        except Exception as exc:
            logger.warning("LLM verification failed: %s: %s", type(exc).__name__, exc)

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


def _llm_verify(state: AgentState, llm: object) -> Verification:
    system = (
        "You are a verifier. Determine whether the evidence supports the draft answer. "
        "Return JSON only."
    )
    evidence_text = "\n".join(
        f"{idx}. {ev.kind}: {ev.text}" for idx, ev in enumerate(state.evidence, start=1)
    )
    draft_text = state.draft.answer_text if state.draft else ""
    user = (
        "Question:\n"
        f"{state.query}\n\n"
        "Draft Answer (may include citations):\n"
        f"{draft_text}\n\n"
        "Evidence:\n"
        f"{evidence_text}\n\n"
        "Return JSON with fields: enough_evidence (boolean), missing (array of strings), "
        "next_action (retrieve/sql_query/code_search/web_search, optional), next_query (string, optional)."
    )
    schema = {
        "type": "object",
        "properties": {
            "enough_evidence": {"type": "boolean"},
            "missing": {"type": "array", "items": {"type": "string"}},
            "next_action": {
                "type": "string",
                "enum": ["retrieve", "sql_query", "code_search", "web_search"],
            },
            "next_query": {"type": "string"},
        },
        "required": ["enough_evidence", "missing"],
        "additionalProperties": False,
    }
    result = llm.chat_json(system=system, user=user, schema=schema)
    return Verification(
        enough_evidence=bool(result.get("enough_evidence")),
        missing=result.get("missing", []) or [],
        next_query=result.get("next_query"),
        next_action=result.get("next_action"),
    )

