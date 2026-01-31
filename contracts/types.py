from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class RetrievalChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    citations: List[str]
    metadata: Dict[str, Any]


@dataclass
class SqlResult:
    rows: List[Dict[str, Any]]
    columns: List[Dict[str, str]]
    stats: Dict[str, Any]


@dataclass
class CodeSnippet:
    path: str
    ref: str
    line_start: int
    line_end: int
    content: str


@dataclass
class Evidence:
    kind: str
    payload: Dict[str, Any]
    citations: List[str]


@dataclass
class AgentVerdict:
    enough_evidence: bool
    missing_what: Optional[str] = None


@dataclass
class AgentState:
    query: str
    tenant_id: str
    user_id: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    route: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    draft: Optional[str] = None
    verdict: Optional[AgentVerdict] = None
    final: Optional[str] = None

    def add_tool_call(self, name: str, params: Dict[str, Any]) -> None:
        self.tool_calls.append({"name": name, "params": params})

    def add_evidence(self, kind: str, payload: Dict[str, Any], citations: Sequence[str]) -> None:
        self.evidence.append(Evidence(kind=kind, payload=payload, citations=list(citations)))

