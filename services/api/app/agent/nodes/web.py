from __future__ import annotations

from typing import Any

from ..state import AgentState, EvidenceItem, ToolCallRecord, now_ms


def web_node(state: AgentState, services: Any) -> AgentState:
    args = {"query": state.query, "recency_days": 30}
    record = ToolCallRecord(
        name="web_search",
        args=args,
        ok=True,
        started_at_ms=now_ms(),
        ended_at_ms=now_ms(),
        result_preview={"count": 0},
    )
    state.tool_calls.append(record)
    state.evidence.append(
        EvidenceItem(
            kind="web_snippets",
            score=0.0,
            text="",
            citations=[],
            metadata={"count": 0},
        )
    )
    return state

