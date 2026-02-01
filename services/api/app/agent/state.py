from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

SourceType = Literal["pdf", "web", "repo", "db_doc"]
RouteType = Literal["doc_rag", "sql", "code", "mixed", "web_fallback"]
Confidence = Literal["high", "medium", "low"]

ToolName = Literal[
    "retrieve",
    "sql_query",
    "code_search",
    "web_search",
    "web_fetch",
]


@dataclass
class Citation:
    kind: Literal["chunk", "row", "code", "web"]
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    url: Optional[str] = None
    path: Optional[str] = None
    ref: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    row_ref: Optional[str] = None
    title: Optional[str] = None


@dataclass
class EvidenceItem:
    kind: Literal["doc_chunk", "sql_rows", "code_snippets", "web_snippets"]
    score: float = 0.0
    text: str = ""
    citations: List[Citation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallRecord:
    name: ToolName
    args: Dict[str, Any]
    ok: bool
    started_at_ms: int
    ended_at_ms: int
    error: Optional[str] = None
    result_preview: Optional[Dict[str, Any]] = None


@dataclass
class Constraints:
    source_types: Optional[List[SourceType]] = None
    doc_ids: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    repo: Optional[str] = None
    ref: Optional[str] = None
    path_prefix: Optional[str] = None
    url_prefix: Optional[str] = None
    time_from: Optional[str] = None
    time_to: Optional[str] = None
    security_scope: Optional[Dict[str, Any]] = None


@dataclass
class Verification:
    enough_evidence: bool
    missing: List[str] = field(default_factory=list)
    next_query: Optional[str] = None
    next_action: Optional[Literal["retrieve", "sql_query", "code_search", "web_search"]] = None


@dataclass
class Draft:
    answer_outline: List[str] = field(default_factory=list)
    answer_text: str = ""
    used_citations: List[Citation] = field(default_factory=list)


@dataclass
class FinalAnswer:
    answer: str
    citations: List[Citation]
    confidence: Confidence
    followups: List[str] = field(default_factory=list)


@dataclass
class AgentState:
    request_id: str
    tenant_id: str
    user_id: str
    session_id: Optional[str] = None
    query: str = ""
    constraints: Constraints = field(default_factory=Constraints)
    route: Optional[RouteType] = None
    plan: List[str] = field(default_factory=list)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)
    draft: Optional[Draft] = None
    verification: Optional[Verification] = None
    iteration: int = 0
    max_iterations: int = 3
    hard_fail: bool = False
    final: Optional[FinalAnswer] = None


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

