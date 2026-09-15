from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..auth.principal import require_admin
from ..quality.contracts import EvaluationRun, PromotionPolicy
from ..quality.promotion import evaluate_promotion
from ..storage.quality_support import ensure_quality_repository


class EvaluationRunRequest(BaseModel):
    evaluation_run_id: Optional[str] = None
    tenant_id: Optional[str] = None
    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    code_revision: str = Field(min_length=1)
    candidate_ref: Optional[str] = None
    baseline_ref: Optional[str] = None
    runtime_contracts: Dict[str, Any] = {}
    config: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    latency: Dict[str, Any] = {}
    cost: Dict[str, Any] = {}
    artifacts: Dict[str, Any] = {}


class PromotionPolicyRequest(BaseModel):
    max_recall_drop: float = Field(default=0.0, ge=0.0)
    max_mrr_drop: float = Field(default=0.0, ge=0.0)
    max_ndcg_drop: float = Field(default=0.0, ge=0.0)
    max_p95_latency_increase_ratio: float = Field(default=0.15, ge=0.0)
    max_cost_increase_ratio: float = Field(default=0.25, ge=0.0)
    min_recall: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_mrr: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_ndcg: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    critical_case_ids: list[str] = Field(default_factory=list)


class PromotionRequest(BaseModel):
    baseline_evaluation_id: str = Field(min_length=1)
    candidate_evaluation_id: str = Field(min_length=1)
    policy: PromotionPolicyRequest = PromotionPolicyRequest()


def create_quality_router(get_services: Callable, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/admin/quality", tags=["quality"])

    @router.get("/runs/{request_id}")
    async def get_run(
        request_id: str,
        _key: Optional[str] = Depends(auth_dep),
    ) -> dict[str, Any]:
        require_admin(_key)
        repo = ensure_quality_repository(get_services().repo)
        run = repo.get_rag_run(request_id)
        if run is None:
            raise HTTPException(status_code=404, detail="RAG run not found")
        return asdict(run)

    @router.get("/runs")
    async def list_runs(
        tenant_id: Optional[str] = None,
        limit: int = Query(default=100, ge=1, le=500),
        _key: Optional[str] = Depends(auth_dep),
    ) -> dict[str, Any]:
        require_admin(_key)
        repo = ensure_quality_repository(get_services().repo)
        runs = repo.list_rag_runs(tenant_id=tenant_id, limit=limit)
        return {"runs": [asdict(item) for item in runs]}

    @router.post("/evaluations", status_code=201)
    async def create_evaluation(
        payload: EvaluationRunRequest,
        _key: Optional[str] = Depends(auth_dep),
    ) -> dict[str, Any]:
        require_admin(_key)
        repo = ensure_quality_repository(get_services().repo)
        run = EvaluationRun.build(
            evaluation_run_id=payload.evaluation_run_id,
            tenant_id=payload.tenant_id,
            dataset_name=payload.dataset_name,
            dataset_version=payload.dataset_version,
            code_revision=payload.code_revision,
            candidate_ref=payload.candidate_ref,
            baseline_ref=payload.baseline_ref,
            runtime_contracts=payload.runtime_contracts,
            config=payload.config,
            metrics=payload.metrics,
            latency=payload.latency,
            cost=payload.cost,
            artifacts=payload.artifacts,
        )
        try:
            repo.add_evaluation_run(run)
        except Exception as exc:
            if repo.get_evaluation_run(run.evaluation_run_id) is not None:
                raise HTTPException(
                    status_code=409,
                    detail="EvaluationRun IDs are immutable and already exist",
                ) from exc
            raise
        return asdict(run)

    @router.get("/evaluations/{evaluation_run_id}")
    async def get_evaluation(
        evaluation_run_id: str,
        _key: Optional[str] = Depends(auth_dep),
    ) -> dict[str, Any]:
        require_admin(_key)
        repo = ensure_quality_repository(get_services().repo)
        run = repo.get_evaluation_run(evaluation_run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="EvaluationRun not found")
        return asdict(run)

    @router.get("/evaluations")
    async def list_evaluations(
        limit: int = Query(default=100, ge=1, le=500),
        _key: Optional[str] = Depends(auth_dep),
    ) -> dict[str, Any]:
        require_admin(_key)
        repo = ensure_quality_repository(get_services().repo)
        runs = repo.list_evaluation_runs(limit=limit)
        return {"evaluations": [asdict(item) for item in runs]}

    @router.post("/promotions/evaluate")
    async def promotion_gate(
        payload: PromotionRequest,
        _key: Optional[str] = Depends(auth_dep),
    ) -> dict[str, Any]:
        require_admin(_key)
        repo = ensure_quality_repository(get_services().repo)
        baseline = repo.get_evaluation_run(payload.baseline_evaluation_id)
        candidate = repo.get_evaluation_run(payload.candidate_evaluation_id)
        if baseline is None or candidate is None:
            missing = []
            if baseline is None:
                missing.append(payload.baseline_evaluation_id)
            if candidate is None:
                missing.append(payload.candidate_evaluation_id)
            raise HTTPException(
                status_code=404,
                detail=f"EvaluationRun not found: {', '.join(missing)}",
            )
        policy = PromotionPolicy(**payload.policy.model_dump())
        decision = evaluate_promotion(baseline, candidate, policy)
        repo.add_promotion_decision(decision)
        return asdict(decision)

    @router.post("/retention/purge")
    async def purge_expired_runs(
        _key: Optional[str] = Depends(auth_dep),
    ) -> dict[str, int]:
        require_admin(_key)
        repo = ensure_quality_repository(get_services().repo)
        return {"deleted": int(repo.purge_expired_rag_runs())}

    return router
