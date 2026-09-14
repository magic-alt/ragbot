"""Golden Dataset promotion runner for #57 dense+sparse retrieval.

This module intentionally compares exactly two retrieval plans over the same
runtime, tenant scope, dense embedding contract, reranker setting and Golden
Dataset:

- baseline: ``hybrid_rrf`` against the active alias;
- candidate: ``qdrant_dense_sparse`` against one validating/ready IndexVersion.

It persists two immutable EvaluationRuns and one PromotionDecision. It never
activates the candidate automatically. ``--mark-ready`` may move an accepted
candidate from validating to ready, but alias activation/rollback remains the
existing explicit IndexVersion operation.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from benchmarks.rag_native_compare import (
    RetrievedHit,
    audit_golden_dataset,
    load_golden_dataset,
    score_case,
    summarize_scores,
)
from services.api.app.factory import build_services_from_env
from services.api.app.quality.contracts import EvaluationRun, PromotionPolicy
from services.api.app.quality.promotion import evaluate_promotion
from services.api.app.retrieval.contracts import RetrievalPlan, RetrievalRequest
from services.api.app.storage.quality_support import ensure_quality_repository


def dataset_version(dataset: Mapping[str, Any]) -> str:
    encoded = json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:24]}"


async def evaluate_dense_sparse_candidate(
    *,
    services: Any,
    dataset: Mapping[str, Any],
    candidate_index_version_id: str,
    tenant_id: str,
    user_id: str = "retrieval-promotion",
    top_k: Optional[int] = None,
    candidate_pool: Optional[int] = None,
    rerank: bool = True,
    repetitions: int = 1,
    deadline_ms: int = 30000,
    code_revision: Optional[str] = None,
    policy: Optional[PromotionPolicy] = None,
    persist: bool = True,
    mark_ready_on_accept: bool = False,
) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    cases = [case for case in list(dataset.get("cases") or []) if case.get("relevance")]
    if not cases:
        raise ValueError("Golden Dataset requires labeled cases")
    effective_top_k = int(top_k or (dataset.get("defaults") or {}).get("top_k") or 10)
    if effective_top_k < 10:
        raise ValueError("top_k must be >= 10 because promotion reports Recall/MRR/nDCG@10")

    repo = ensure_quality_repository(services.repo)
    lifecycle = getattr(services, "index_lifecycle", None)
    if lifecycle is None:
        raise RuntimeError("Dense+sparse promotion requires IndexLifecycleService")
    candidate = lifecycle.repo.get_index_version(candidate_index_version_id)
    if candidate is None:
        raise ValueError(f"Candidate IndexVersion not found: {candidate_index_version_id}")
    active = lifecycle.repo.get_active_index_version(lifecycle.alias_name)
    if active is None:
        raise RuntimeError("No active baseline IndexVersion")
    if candidate.index_version_id == active.index_version_id:
        raise ValueError("Candidate must be evaluated before activation against a distinct baseline")
    if candidate.status not in {"validating", "ready"}:
        raise ValueError(f"Candidate is not evaluable: {candidate.status}")
    sparse_schema = dict((candidate.vector_schema or {}).get("sparse") or {})
    sparse_contract_id = str(sparse_schema.get("contract_id") or "")
    if not sparse_contract_id:
        raise ValueError("Candidate IndexVersion does not declare a sparse contract")
    if candidate.embedding_contract_id != active.embedding_contract_id:
        raise ValueError(
            "Controlled dense+sparse promotion requires the same dense embedding contract "
            f"for baseline and candidate: baseline={active.embedding_contract_id} "
            f"candidate={candidate.embedding_contract_id}"
        )

    # Warm one query on both branches outside measured evidence. This removes
    # lazy model/session initialization from steady-state p95 while preserving
    # real network/encoding/search costs in every measured repetition.
    warm_query = str(cases[0].get("query") or "").strip()
    if warm_query:
        filters = _filters(dataset, cases[0], tenant_id)
        await services.retriever.query(
            _request(
                warm_query,
                filters,
                top_k=effective_top_k,
                candidate_pool=candidate_pool,
                rerank=rerank,
                deadline_ms=deadline_ms,
                plan=RetrievalPlan.HYBRID_RRF,
            )
        )
        await services.retriever.query(
            _request(
                warm_query,
                filters,
                top_k=effective_top_k,
                candidate_pool=candidate_pool,
                rerank=rerank,
                deadline_ms=deadline_ms,
                plan=RetrievalPlan.QDRANT_DENSE_SPARSE,
                index_version_id=candidate.index_version_id,
            )
        )

    baseline_evidence = await _run_plan(
        services,
        dataset,
        cases,
        plan=RetrievalPlan.HYBRID_RRF,
        index_version_id=None,
        tenant_id=tenant_id,
        top_k=effective_top_k,
        candidate_pool=candidate_pool,
        rerank=rerank,
        repetitions=repetitions,
        deadline_ms=deadline_ms,
    )
    candidate_evidence = await _run_plan(
        services,
        dataset,
        cases,
        plan=RetrievalPlan.QDRANT_DENSE_SPARSE,
        index_version_id=candidate.index_version_id,
        tenant_id=tenant_id,
        top_k=effective_top_k,
        candidate_pool=candidate_pool,
        rerank=rerank,
        repetitions=repetitions,
        deadline_ms=deadline_ms,
    )

    revision = code_revision or _git_revision()
    version = dataset_version(dataset)
    common_config = {
        "top_k": effective_top_k,
        "candidate_pool": candidate_pool,
        "rerank": bool(rerank),
        "repetitions": int(repetitions),
        "deadline_ms": int(deadline_ms),
        "tenant_id": tenant_id,
        "controlled_dense_embedding_contract_id": active.embedding_contract_id,
    }
    baseline_run = _evaluation_run(
        dataset=dataset,
        dataset_version_value=version,
        code_revision=revision,
        tenant_id=tenant_id,
        candidate_ref=None,
        baseline_ref=active.index_version_id,
        index_version=active,
        plan=RetrievalPlan.HYBRID_RRF,
        sparse_contract_id=None,
        evidence=baseline_evidence,
        config=common_config,
    )
    candidate_run = _evaluation_run(
        dataset=dataset,
        dataset_version_value=version,
        code_revision=revision,
        tenant_id=tenant_id,
        candidate_ref=candidate.index_version_id,
        baseline_ref=active.index_version_id,
        index_version=candidate,
        plan=RetrievalPlan.QDRANT_DENSE_SPARSE,
        sparse_contract_id=sparse_contract_id,
        evidence=candidate_evidence,
        config=common_config,
    )
    decision = evaluate_promotion(
        baseline_run,
        candidate_run,
        policy or PromotionPolicy(),
    )

    if persist:
        repo.add_evaluation_run(baseline_run)
        repo.add_evaluation_run(candidate_run)
        repo.add_promotion_decision(decision)

    marked_ready = False
    if mark_ready_on_accept and decision.decision == "accept":
        evidence = {
            "promotion_decision_id": decision.promotion_decision_id,
            "baseline_evaluation_id": baseline_run.evaluation_run_id,
            "candidate_evaluation_id": candidate_run.evaluation_run_id,
            "policy": decision.policy,
            "deltas": decision.deltas,
            "dataset_version": version,
            "retrieval_plan": RetrievalPlan.QDRANT_DENSE_SPARSE.value,
            "sparse_contract_id": sparse_contract_id,
        }
        lifecycle.mark_ready(candidate.index_version_id, evidence, approved=True)
        marked_ready = True

    return {
        "dataset": {
            "name": str(dataset.get("name") or "Golden Dataset"),
            "version": version,
            "cases": len(cases),
        },
        "baseline": asdict(baseline_run),
        "candidate": asdict(candidate_run),
        "promotion": asdict(decision),
        "candidate_marked_ready": marked_ready,
        "activation_performed": False,
    }


async def _run_plan(
    services: Any,
    dataset: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    *,
    plan: RetrievalPlan,
    index_version_id: Optional[str],
    tenant_id: str,
    top_k: int,
    candidate_pool: Optional[int],
    rerank: bool,
    repetitions: int,
    deadline_ms: int,
) -> dict[str, Any]:
    scores = []
    latencies_ms: list[float] = []
    per_case: list[dict[str, Any]] = []
    stage_samples: dict[str, list[float]] = {}
    first_trace: dict[str, Any] = {}
    for case in cases:
        query = str(case.get("query") or "").strip()
        if not query:
            continue
        filters = _filters(dataset, case, tenant_id)
        first_hits: Optional[list[RetrievedHit]] = None
        first_response = None
        measured = []
        for _ in range(repetitions):
            started = time.perf_counter()
            response = await services.retriever.query(
                _request(
                    query,
                    filters,
                    top_k=top_k,
                    candidate_pool=candidate_pool,
                    rerank=rerank,
                    deadline_ms=deadline_ms,
                    plan=plan,
                    index_version_id=index_version_id,
                )
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)
            measured.append(elapsed_ms)
            trace = response.trace.as_dict()
            for stage, value in dict(trace.get("stage_ms") or {}).items():
                stage_samples.setdefault(str(stage), []).append(float(value))
            if first_response is None:
                first_response = response
                first_trace = first_trace or trace
                first_hits = [_retrieved_hit(chunk) for chunk in response.chunks]
        score = score_case(case, first_hits or [])
        scores.append(score)
        per_case.append(
            {
                "case_id": score.case_id,
                "category": score.category,
                "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "first_relevant_rank": score.first_relevant_rank,
                "reciprocal_rank_at_10": score.reciprocal_rank_at_10,
                "recall_at_10": score.recall_at_10,
                "ndcg_at_10": score.ndcg_at_10,
                "retrieval_pass": score.retrieval_pass,
                "latency_ms_mean": round(statistics.fmean(measured), 3),
                "top_chunk_ids": [hit.chunk_id for hit in (first_hits or [])],
                "trace": first_response.trace.as_dict() if first_response is not None else {},
            }
        )
    summary = summarize_scores(scores, latencies_ms)
    return {
        "summary": summary,
        "stage_latency": {
            stage: {
                "mean_ms": round(statistics.fmean(values), 3),
                "p95_ms": round(_percentile(values, 0.95), 3),
            }
            for stage, values in stage_samples.items()
            if values
        },
        "cases": per_case,
        "trace_contract": first_trace,
    }


def _evaluation_run(
    *,
    dataset: Mapping[str, Any],
    dataset_version_value: str,
    code_revision: str,
    tenant_id: str,
    candidate_ref: Optional[str],
    baseline_ref: Optional[str],
    index_version: Any,
    plan: RetrievalPlan,
    sparse_contract_id: Optional[str],
    evidence: Mapping[str, Any],
    config: Mapping[str, Any],
) -> EvaluationRun:
    summary = dict(evidence.get("summary") or {})
    runtime_contracts = {
        "retrieval_plan": plan.value,
        "index_version_id": index_version.index_version_id,
        "embedding_contract_id": index_version.embedding_contract_id,
        "sparse_contract_id": sparse_contract_id,
        "vector_schema": dict(index_version.vector_schema or {}),
        "trace_contract": dict(evidence.get("trace_contract") or {}).get(
            "representation_contracts", {}
        ),
    }
    return EvaluationRun.build(
        dataset_name=str(dataset.get("name") or "Golden Dataset"),
        dataset_version=dataset_version_value,
        code_revision=code_revision,
        tenant_id=tenant_id,
        candidate_ref=candidate_ref,
        baseline_ref=baseline_ref,
        runtime_contracts=runtime_contracts,
        config=dict(config),
        metrics={
            "recall": summary.get("recall_at_10"),
            "mrr": summary.get("mrr_at_10"),
            "ndcg": summary.get("ndcg_at_10"),
            "hit_at_5": summary.get("hit_at_5"),
            "hit_at_10": summary.get("hit_at_10"),
            "pass_rate": summary.get("pass_rate"),
        },
        latency={
            "p50_latency_ms": summary.get("query_latency_ms_p50"),
            "p95_latency_ms": summary.get("query_latency_ms_p95"),
            "mean_latency_ms": summary.get("query_latency_ms_mean"),
            "stage_latency": dict(evidence.get("stage_latency") or {}),
        },
        # Retrieval embeddings currently do not expose a normalized provider
        # billing feed. Persist an explicit zero rather than omitting the field,
        # so promotion cost comparison is deterministic. Future provider usage
        # integration can replace this value without changing the gate schema.
        cost={"cost_usd": 0.0, "basis": "retrieval-provider-cost-unavailable"},
        artifacts={
            "cases": list(evidence.get("cases") or []),
            "category_metrics": summary.get("categories") or {},
            "summary": summary,
        },
    )


def _request(
    query: str,
    filters: dict[str, Any],
    *,
    top_k: int,
    candidate_pool: Optional[int],
    rerank: bool,
    deadline_ms: int,
    plan: RetrievalPlan,
    index_version_id: Optional[str] = None,
) -> RetrievalRequest:
    return RetrievalRequest(
        query=query,
        filters=filters,
        top_k=top_k,
        plan=plan,
        candidate_pool=candidate_pool,
        rerank=rerank,
        deadline_ms=deadline_ms,
        diversity=False,
        index_version_id=index_version_id,
    )


def _filters(
    dataset: Mapping[str, Any],
    case: Mapping[str, Any],
    tenant_id: str,
) -> dict[str, Any]:
    defaults = dataset.get("defaults") or {}
    result = {"tenant_id": tenant_id}
    if isinstance(defaults, Mapping):
        result.update(dict(defaults.get("filters") or {}))
    result.update(dict(case.get("filters") or {}))
    return result


def _retrieved_hit(chunk: Any) -> RetrievedHit:
    metadata = dict(getattr(chunk, "metadata", None) or {})
    return RetrievedHit(
        chunk_id=str(getattr(chunk, "chunk_id", "") or ""),
        doc_id=str(getattr(chunk, "doc_id", "") or ""),
        path=str(metadata.get("path") or ""),
        page=metadata.get("page"),
        text=str(getattr(chunk, "text", "") or ""),
        score=float(getattr(chunk, "score", 0.0) or 0.0),
        metadata=metadata,
    )


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()
    except Exception:
        return "unknown"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", required=True)
    parser.add_argument("--candidate-index", required=True)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--user", default="retrieval-promotion")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--candidate-pool", type=int)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--deadline-ms", type=int, default=30000)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--mark-ready", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--max-recall-drop", type=float, default=0.0)
    parser.add_argument("--max-mrr-drop", type=float, default=0.0)
    parser.add_argument("--max-ndcg-drop", type=float, default=0.0)
    parser.add_argument("--max-p95-latency-increase-ratio", type=float, default=0.15)
    parser.add_argument("--max-cost-increase-ratio", type=float, default=0.25)
    return parser.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_golden_dataset(Path(args.golden))
    audit = audit_golden_dataset(dataset, profile="development")
    if not audit["passed"]:
        raise SystemExit(f"Golden Dataset audit failed: {audit}")
    services = build_services_from_env()
    try:
        return await evaluate_dense_sparse_candidate(
            services=services,
            dataset=dataset,
            candidate_index_version_id=args.candidate_index,
            tenant_id=args.tenant,
            user_id=args.user,
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
            rerank=not args.no_rerank,
            repetitions=args.repetitions,
            deadline_ms=args.deadline_ms,
            policy=PromotionPolicy(
                max_recall_drop=args.max_recall_drop,
                max_mrr_drop=args.max_mrr_drop,
                max_ndcg_drop=args.max_ndcg_drop,
                max_p95_latency_increase_ratio=args.max_p95_latency_increase_ratio,
                max_cost_increase_ratio=args.max_cost_increase_ratio,
            ),
            mark_ready_on_accept=args.mark_ready,
        )
    finally:
        close = getattr(services.retriever, "close", None)
        if callable(close):
            close()
        for resource in (services.qdrant, services.repo):
            closer = getattr(resource, "close", None)
            if callable(closer):
                closer()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = asyncio.run(_main_async(args))
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0 if result["promotion"]["decision"] == "accept" else 2


if __name__ == "__main__":
    raise SystemExit(main())
