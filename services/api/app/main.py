from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional

from .agent.graph import AgentServices, build_default_services, run_agent
from .agent.state import Constraints, FinalAnswer


def chat(
    query: str,
    tenant_id: str,
    user_id: str,
    services: Optional[AgentServices] = None,
    constraints: Optional[Constraints] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Dict[str, object]:
    services = services or build_default_services()
    state = run_agent(
        query=query,
        tenant_id=tenant_id,
        user_id=user_id,
        services=services,
        constraints=constraints,
        session_id=session_id,
        request_id=request_id,
    )
    final = state.final or FinalAnswer(answer="", citations=[], confidence="low")
    return {
        "request_id": state.request_id,
        "answer": final.answer,
        "citations": [asdict(cite) for cite in final.citations],
        "confidence": final.confidence,
        "followups": final.followups,
        "debug": {
            "route": state.route,
            "tool_calls": [asdict(call) for call in state.tool_calls],
        },
    }

