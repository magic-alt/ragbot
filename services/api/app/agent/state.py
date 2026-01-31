from __future__ import annotations

from contracts.types import AgentState

ROUTE_DOC_RAG = "doc_rag"
ROUTE_SQL = "sql"
ROUTE_CODE = "code"
ROUTE_MIXED = "mixed"
ROUTE_WEB = "web_fallback"


def build_initial_state(query: str, tenant_id: str, user_id: str) -> AgentState:
    return AgentState(query=query, tenant_id=tenant_id, user_id=user_id)

