from __future__ import annotations

from typing import Dict, Optional

from .agent.graph import AgentServices, build_default_services, run_agent


def chat(query: str, tenant_id: str, user_id: str, services: Optional[AgentServices] = None) -> Dict[str, object]:
    services = services or build_default_services()
    state = run_agent(query, tenant_id, user_id, services)
    citations = []
    for evidence in state.evidence:
        citations.extend(evidence.citations)
    return {
        "answer": state.final or "",
        "citations": citations,
        "confidence": "high" if state.verdict and state.verdict.enough_evidence else "low",
    }

