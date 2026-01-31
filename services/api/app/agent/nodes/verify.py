from __future__ import annotations

from contracts.types import AgentState, AgentVerdict


def verify_node(state: AgentState, services: object) -> AgentState:
    if not state.evidence:
        state.verdict = AgentVerdict(enough_evidence=False, missing_what="no evidence")
        return state

    if state.route == "sql":
        rows_evidence = [ev for ev in state.evidence if ev.kind == "rows"]
        if not rows_evidence:
            state.verdict = AgentVerdict(enough_evidence=False, missing_what="sql rows")
            return state
        if rows_evidence and not rows_evidence[0].payload.get("rows"):
            state.verdict = AgentVerdict(enough_evidence=False, missing_what="sql rows")
            return state

    if state.route == "code":
        code_evidence = [ev for ev in state.evidence if ev.kind == "code"]
        if not code_evidence:
            state.verdict = AgentVerdict(enough_evidence=False, missing_what="code snippets")
            return state

    if state.route in ("doc_rag", "mixed"):
        chunk_evidence = [ev for ev in state.evidence if ev.kind == "chunk"]
        if not chunk_evidence:
            state.verdict = AgentVerdict(enough_evidence=False, missing_what="document chunks")
            return state

    state.verdict = AgentVerdict(enough_evidence=True)
    return state

