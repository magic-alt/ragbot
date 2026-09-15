from __future__ import annotations

import hashlib
import json
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
class QdrantRrfFusionSpec:
    """Immutable Qdrant RRF query contract for controlled hybrid experiments.

    Prefetch order is dense first, sparse second, so weights follow the same
    order. The default Qdrant RRF constant is k=2. This object belongs to the
    retrieval experiment contract, not IndexVersion: changing weights does not
    change stored vector representations and therefore never requires reindexing.
    """

    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    k: int = 2
    provider_id: str = "qdrant"
    method: str = "rrf"

    def __post_init__(self) -> None:
        if float(self.dense_weight) <= 0 or float(self.sparse_weight) <= 0:
            raise ValueError("RRF dense/sparse weights must be > 0")
        if int(self.k) <= 0:
            raise ValueError("RRF k must be > 0")
        if self.provider_id != "qdrant" or self.method != "rrf":
            raise ValueError("QdrantRrfFusionSpec requires provider_id=qdrant and method=rrf")

    @property
    def weights(self) -> tuple[float, float]:
        return (float(self.dense_weight), float(self.sparse_weight))

    @property
    def contract_id(self) -> str:
        canonical = json.dumps(self.as_dict(include_contract=False), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
        return f"fusion-{digest}"

    @property
    def weighted(self) -> bool:
        return self.weights != (1.0, 1.0)

    def as_dict(self, *, include_contract: bool = True) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "provider_id": self.provider_id,
            "method": self.method,
            "k": int(self.k),
            "dense_weight": float(self.dense_weight),
            "sparse_weight": float(self.sparse_weight),
            "weights": [float(self.dense_weight), float(self.sparse_weight)],
            "prefetch_order": ["dense", "sparse"],
        }
        if include_contract:
            value["contract_id"] = self.contract_id
        return value


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
    # Internal evaluation selector. The public /v1/search contract does not
    # expose this field; it exists so a validating IndexVersion can be measured
    # before its Qdrant alias is activated.
    index_version_id: Optional[str] = None
    # Internal fusion experiment contract. Public online callers continue to
    # use the deployed/default fusion policy until evidence promotes a contract.
    fusion_spec: Optional[QdrantRrfFusionSpec] = None

    def __post_init__(self) -> None:
        if not str(self.query).strip():
            raise ValueError("RetrievalRequest.query must not be empty")
        if int(self.top_k) <= 0:
            raise ValueError("RetrievalRequest.top_k must be > 0")
        if self.candidate_pool is not None and int(self.candidate_pool) <= 0:
            raise ValueError("RetrievalRequest.candidate_pool must be > 0 when set")
        if self.deadline_ms is not None and int(self.deadline_ms) <= 0:
            raise ValueError("RetrievalRequest.deadline_ms must be > 0 when set")
        if self.index_version_id is not None and not str(self.index_version_id).strip():
            raise ValueError("RetrievalRequest.index_version_id must not be empty when set")
        if self.fusion_spec is not None and self.plan is not RetrievalPlan.QDRANT_DENSE_SPARSE:
            raise ValueError("fusion_spec is currently supported only by qdrant_dense_sparse")


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
    representation_contracts: Dict[str, Any] = field(default_factory=dict)
    index_version_id: Optional[str] = None
    reranker_configured: bool = False
    reranker_requested: bool = False
    reranker_enabled: bool = False
    reranker_candidate_count: int = 0
    diversity_enabled: bool = False
    timed_out: bool = False
    cancelled: bool = False
    error_stage: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        legacy_mode = {
            RetrievalPlan.DENSE.value: "vector",
            RetrievalPlan.LEXICAL.value: "lexical",
            RetrievalPlan.HYBRID_RRF.value: "hybrid",
            RetrievalPlan.QDRANT_DENSE_SPARSE.value: "qdrant_dense_sparse",
        }.get(self.plan, self.plan)
        dense_count = int(self.candidate_counts.get("dense", 0))
        lexical_count = int(self.candidate_counts.get("lexical", 0))
        return {
            "retrieval_plan": self.plan,
            "retrieval_mode": legacy_mode,
            "deadline_ms": self.deadline_ms,
            "candidate_pool": self.candidate_pool,
            "parallel_fanout": self.plan == RetrievalPlan.HYBRID_RRF.value,
            "vector_candidates": dense_count,
            "lexical_candidates": lexical_count,
            "stage_ms": {key: round(float(value), 3) for key, value in self.stage_ms.items()},
            "candidate_counts": dict(self.candidate_counts),
            "fusion_method": self.fusion_method,
            "fusion_policy": dict(self.fusion_policy),
            "representation_contracts": dict(self.representation_contracts),
            "index_version_id": self.index_version_id,
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
