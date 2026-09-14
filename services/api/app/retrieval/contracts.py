from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from contracts.types import RetrievalChunk


class RetrievalPlan(str, Enum):
    """Stable retrieval-plan identifiers exposed to API/SDK callers."""

    DENSE = "dense"
    LEXICAL = "lexical"
    HYBRID_RRF = "hybrid_rrf"
    QDRANT_DENSE_SPARSE = "qdrant_dense_sparse"


_LEGACY_MODE_TO_PLAN = {
    "vector": RetrievalPlan.DENSE,
    "dense": RetrievalPlan.DENSE,
    "lexical": RetrievalPlan.LEXICAL,
    "hybrid": RetrievalPlan.HYBRID_RRF,
    "hybrid_rrf": RetrievalPlan.HYBRID_RRF,
    "qdrant_dense_sparse": RetrievalPlan.QDRANT_DENSE_SPARSE,
}


def resolve_retrieval_plan(
    plan: str | RetrievalPlan | None = None,
    *,
    legacy_mode: str | None = None,
) -> RetrievalPlan:
    raw = plan.value if isinstance(plan, RetrievalPlan) else str(plan or legacy_mode or "hybrid_rrf")
    normalized = raw.strip().lower()
    try:
        return _LEGACY_MODE_TO_PLAN[normalized]
    except KeyError as exc:
        allowed = ", ".join(item.value for item in RetrievalPlan)
        raise ValueError(f"Unsupported retrieval plan={raw!r}; expected one of: {allowed}") from exc


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    filters: Dict[str, Any]
    top_k: int = 20
    plan: RetrievalPlan = RetrievalPlan.HYBRID_RRF
    candidate_pool: Optional[int] = None
    rerank: bool = True
    deadline_ms: Optional[int] = None
    diversity: bool = False

    def __post_init__(self) -> None:
        if not str(self.query).strip():
            raise ValueError("RetrievalRequest.query must not be empty")
        if int(self.top_k) <= 0:
            raise ValueError("RetrievalRequest.top_k must be > 0")
        if self.candidate_pool is not None and int(self.candidate_pool) <= 0:
            raise ValueError("RetrievalRequest.candidate_pool must be > 0 when set")
        if self.deadline_ms is not None and int(self.deadline_ms) <= 0:
            raise ValueError("RetrievalRequest.deadline_ms must be > 0 when set")


@dataclass
class Candidate:
    chunk_id: str
    score: float
    source: str
    rank: int
    payload: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass
class RetrievalTrace:
    plan: str
    deadline_ms: Optional[int]
    candidate_pool: int
    started_monotonic: float
    stage_ms: Dict[str, float] = field(default_factory=dict)
    candidate_counts: Dict[str, int] = field(default_factory=dict)
    fusion_method: Optional[str] = None
    fusion_policy: Dict[str, Any] = field(default_factory=dict)
    reranker_configured: bool = False
    reranker_requested: bool = False
    reranker_enabled: bool = False
    reranker_candidate_count: int = 0
    diversity_enabled: bool = False
    timed_out: bool = False
    cancelled: bool = False
    error_stage: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "retrieval_plan": self.plan,
            "deadline_ms": self.deadline_ms,
            "candidate_pool": self.candidate_pool,
            "stage_ms": {key: round(float(value), 3) for key, value in self.stage_ms.items()},
            "candidate_counts": dict(self.candidate_counts),
            "fusion_method": self.fusion_method,
            "fusion_policy": dict(self.fusion_policy),
            "reranker_configured": self.reranker_configured,
            "reranker_requested": self.reranker_requested,
            "reranker_enabled": self.reranker_enabled,
            "reranker_candidate_count": self.reranker_candidate_count,
            "diversity_enabled": self.diversity_enabled,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "error_stage": self.error_stage,
        }


@dataclass
class RetrievalResponse:
    chunks: List[RetrievalChunk]
    trace: RetrievalTrace


class RetrievalDeadlineExceeded(TimeoutError):
    def __init__(self, message: str, trace: RetrievalTrace) -> None:
        super().__init__(message)
        self.trace = trace


class UnsupportedRetrievalPlan(RuntimeError):
    pass
