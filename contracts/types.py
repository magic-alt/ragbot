from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


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
class PatchResult:
    path: str
    diff: str
    original_lines: int
    modified_lines: int


# SourceType represents document/vector retrieval sources. Database access is a
# separate SQL tool configured by POSTGRES_DSN; it is not an ingestible source.
SourceType = Literal["pdf", "web", "repo", "local_fs", "s3"]
RouteType = Literal["doc_rag", "sql", "code", "mixed", "web_fallback"]
Confidence = Literal["high", "medium", "low"]
ToolName = Literal[
    "retrieve",
    "sql_query",
    "code_search",
    "web_search",
    "web_fetch",
    "open_file",
    "apply_patch",
    "explain_error",
]


@dataclass(eq=False)
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

    def _key(self) -> tuple:
        return (
            self.kind, self.chunk_id, self.doc_id, self.page,
            self.section, self.url, self.path, self.ref,
            self.line_start, self.line_end, self.row_ref, self.title,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Citation):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())


@dataclass
class EvidenceItem:
    kind: Literal[
        "doc_chunk",
        "sql_rows",
        "code_snippets",
        "web_snippets",
        "file_content",
        "patch",
        "error_analysis",
    ]
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
    next_action: Optional[
        Literal[
            "retrieve",
            "sql_query",
            "code_search",
            "web_search",
            "open_file",
            "apply_patch",
            "explain_error",
        ]
    ] = None


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
