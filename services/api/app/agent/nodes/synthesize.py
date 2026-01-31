from __future__ import annotations

from typing import Any, List

from contracts.types import AgentState


def synthesize_node(state: AgentState, services: Any) -> AgentState:
    if not state.evidence:
        state.draft = "未找到可用证据。"
        return state
    lines: List[str] = []
    for evidence in state.evidence:
        snippet = _summarize_evidence(evidence)
        citations = _format_citations(evidence.citations)
        lines.append(f"- {snippet} {citations}")
    state.draft = "\n".join(lines)
    return state


def _summarize_evidence(evidence: Any) -> str:
    if evidence.kind == "chunk":
        text = evidence.payload.get("text", "")
        return text.strip().replace("\n", " ")[:160]
    if evidence.kind == "rows":
        rows = evidence.payload.get("rows", [])
        return f"SQL 返回 {len(rows)} 行。"
    if evidence.kind == "code":
        path = evidence.payload.get("path")
        return f"代码片段来自 {path}。"
    if evidence.kind == "sql_error":
        return f"SQL 错误: {evidence.payload.get('error')}"
    return "证据片段。"


def _format_citations(citations: List[str]) -> str:
    if not citations:
        return ""
    return "(cite: " + ", ".join(citations) + ")"

