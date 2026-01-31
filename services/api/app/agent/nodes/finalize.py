from __future__ import annotations

from contracts.types import AgentState


def finalize_node(state: AgentState, services: object) -> AgentState:
    if state.verdict and not state.verdict.enough_evidence:
        missing = state.verdict.missing_what or "evidence"
        state.final = (
            "当前证据不足以完成回答。"
            f" 缺少: {missing}。"
            "建议补充相关文档或权限后重试。"
        )
        return state
    if state.draft:
        state.final = "基于以下证据整理:\n" + state.draft
    else:
        state.final = "未生成有效回答。"
    return state

