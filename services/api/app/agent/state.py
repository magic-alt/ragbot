from __future__ import annotations

import time
import uuid
from typing import Optional

from contracts.types import (  # noqa: F401 - re-exported
    AgentState,
    Citation,
    Confidence,
    Constraints,
    Draft,
    EvidenceItem,
    FinalAnswer,
    RouteType,
    SourceType,
    ToolCallRecord,
    ToolName,
    Verification,
)

ROUTE_DOC_RAG = "doc_rag"
ROUTE_SQL = "sql"
ROUTE_CODE = "code"
ROUTE_MIXED = "mixed"
ROUTE_WEB = "web_fallback"


def now_ms() -> int:
    return int(time.time() * 1000)


def build_initial_state(
    query: str,
    tenant_id: str,
    user_id: str,
    session_id: Optional[str] = None,
    constraints: Optional[Constraints] = None,
    request_id: Optional[str] = None,
) -> AgentState:
    return AgentState(
        request_id=request_id or uuid.uuid4().hex,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        query=query,
        constraints=constraints or Constraints(),
    )
