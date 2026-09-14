from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional


def stable_contract_id(namespace: str, payload: Mapping[str, Any], *, version: int = 1) -> str:
    """Return a deterministic identifier for representation-affecting config.

    Contract IDs are intentionally content-derived. Runtime/run IDs remain
    separate so repeated evaluations of the same contract are independently
    auditable.
    """

    normalized = {
        "namespace": str(namespace).strip().lower(),
        "version": int(version),
        "payload": _jsonable(payload),
    }
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"{normalized['namespace']}:v{version}:{digest[:24]}"


def retrieval_contract_id(
    *,
    plan: str,
    top_k: int,
    candidate_pool: Optional[int],
    rerank: bool,
    diversity: bool,
) -> str:
    return stable_contract_id(
        "retrieval",
        {
            "plan": str(plan),
            "top_k": int(top_k),
            "candidate_pool": int(candidate_pool) if candidate_pool is not None else None,
            "rerank": bool(rerank),
            "diversity": bool(diversity),
        },
    )


def reranker_contract_id(reranker: Any) -> Optional[str]:
    if reranker is None:
        return None
    enabled = bool(getattr(reranker, "enabled", False))
    payload = {
        "class": f"{type(reranker).__module__}.{type(reranker).__qualname__}",
        "enabled": enabled,
        "provider": getattr(reranker, "provider_id", None),
        "model": getattr(reranker, "model_name", getattr(reranker, "model_id", None)),
        "revision": getattr(reranker, "revision", None),
        "version": getattr(reranker, "version", None),
    }
    return stable_contract_id("reranker", payload)


def model_contract_id(provider: str, model: str, *, prompt_version: Optional[str] = None) -> str:
    return stable_contract_id(
        "model",
        {
            "provider": str(provider),
            "model": str(model),
            "prompt_version": prompt_version,
        },
    )


@dataclass(frozen=True)
class RagRun:
    request_id: str
    tenant_id: str
    user_id: str
    run_kind: str
    status: str
    query_hash: str
    trace_id: Optional[str] = None
    query_text: Optional[str] = None
    route: Optional[str] = None
    retrieval_plan: Optional[str] = None
    retrieval_contract_id: Optional[str] = None
    embedding_contract_id: Optional[str] = None
    index_version_id: Optional[str] = None
    reranker_contract_id: Optional[str] = None
    model_contracts: list[dict[str, Any]] = field(default_factory=list)
    stage_latency_ms: dict[str, Any] = field(default_factory=dict)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    trace_sampled: bool = True
    total_duration_ms: int = 0
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    expires_at: Optional[str] = None


@dataclass(frozen=True)
class EvaluationRun:
    evaluation_run_id: str
    evaluation_contract_id: str
    dataset_name: str
    dataset_version: str
    code_revision: str
    tenant_id: Optional[str] = None
    candidate_ref: Optional[str] = None
    baseline_ref: Optional[str] = None
    runtime_contracts: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    @classmethod
    def build(
        cls,
        *,
        dataset_name: str,
        dataset_version: str,
        code_revision: str,
        runtime_contracts: Mapping[str, Any],
        config: Optional[Mapping[str, Any]] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        latency: Optional[Mapping[str, Any]] = None,
        cost: Optional[Mapping[str, Any]] = None,
        artifacts: Optional[Mapping[str, Any]] = None,
        tenant_id: Optional[str] = None,
        candidate_ref: Optional[str] = None,
        baseline_ref: Optional[str] = None,
        evaluation_run_id: Optional[str] = None,
    ) -> "EvaluationRun":
        contract_id = stable_contract_id(
            "evaluation",
            {
                "dataset_name": dataset_name,
                "dataset_version": dataset_version,
                "runtime_contracts": dict(runtime_contracts),
                "config": dict(config or {}),
            },
        )
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            evaluation_run_id=evaluation_run_id or f"eval-{uuid.uuid4().hex}",
            evaluation_contract_id=contract_id,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            code_revision=code_revision,
            tenant_id=tenant_id,
            candidate_ref=candidate_ref,
            baseline_ref=baseline_ref,
            runtime_contracts=dict(runtime_contracts),
            config=dict(config or {}),
            metrics=dict(metrics or {}),
            latency=dict(latency or {}),
            cost=dict(cost or {}),
            artifacts=dict(artifacts or {}),
            created_at=now,
            completed_at=now,
        )


@dataclass(frozen=True)
class PromotionPolicy:
    max_recall_drop: float = 0.0
    max_mrr_drop: float = 0.0
    max_ndcg_drop: float = 0.0
    max_p95_latency_increase_ratio: float = 0.15
    max_cost_increase_ratio: float = 0.25

    def as_dict(self) -> dict[str, float]:
        return {
            "max_recall_drop": float(self.max_recall_drop),
            "max_mrr_drop": float(self.max_mrr_drop),
            "max_ndcg_drop": float(self.max_ndcg_drop),
            "max_p95_latency_increase_ratio": float(self.max_p95_latency_increase_ratio),
            "max_cost_increase_ratio": float(self.max_cost_increase_ratio),
        }


@dataclass(frozen=True)
class PromotionDecision:
    promotion_decision_id: str
    baseline_evaluation_id: str
    candidate_evaluation_id: str
    decision: str
    policy: dict[str, Any]
    deltas: dict[str, Any]
    reasons: list[str]
    created_at: Optional[str] = None


def new_promotion_decision_id() -> str:
    return f"promotion-{uuid.uuid4().hex}"


def query_hash(query: str) -> str:
    return hashlib.sha256(str(query).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
