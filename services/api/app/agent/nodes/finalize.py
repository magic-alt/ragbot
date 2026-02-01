from __future__ import annotations

from typing import List

from ..state import AgentState, FinalAnswer


def finalize_node(state: AgentState, services: object) -> AgentState:
    if state.draft and state.verification and state.verification.enough_evidence:
        confidence = "high" if not state.verification.missing else "medium"
        state.final = FinalAnswer(
            answer=state.draft.answer_text,
            citations=state.draft.used_citations,
            confidence=confidence,
            followups=[],
        )
        return state

    missing = state.verification.missing if state.verification else ["insufficient evidence"]
    answer = _render_degraded_answer(missing)
    citations = state.draft.used_citations if state.draft else []
    state.final = FinalAnswer(
        answer=answer,
        citations=citations,
        confidence="low",
        followups=_suggest_followups(missing),
    )
    return state


def _render_degraded_answer(missing: List[str]) -> str:
    detail = "、".join(missing)
    return f"当前证据不足，缺少: {detail}。建议补充相关文档或权限后重试。"


def _suggest_followups(missing: List[str]) -> List[str]:
    followups: List[str] = []
    if "doc_chunks" in missing:
        followups.append("请提供相关文档或放宽检索范围。")
    if "sql_rows" in missing:
        followups.append("请确认 SQL 可访问的表或字段，并提供更具体的筛选条件。")
    if "code_snippets" in missing:
        followups.append("请提供仓库路径或更具体的代码关键词。")
    return followups

