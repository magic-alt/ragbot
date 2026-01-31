from __future__ import annotations

from typing import Any

from contracts.types import AgentState


def web_node(state: AgentState, services: Any) -> AgentState:
    state.add_tool_call("web_search", {"query": state.query})
    return state

