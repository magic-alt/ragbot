from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

from .agent.context import dedup_evidence, compress_evidence
from .agent.graph import AgentServices, build_default_services, run_agent
from .agent.state import Constraints, FinalAnswer
from contracts.types import EvidenceItem


def chat(
    query: str,
    tenant_id: str,
    user_id: str,
    services: Optional[AgentServices] = None,
    constraints: Optional[Constraints] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    initial_evidence: Optional[List[EvidenceItem]] = None,
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
        initial_evidence=initial_evidence,
    )
    # Post-process evidence: dedup + compress
    state.evidence = compress_evidence(dedup_evidence(state.evidence))
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
