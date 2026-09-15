"""Evidence-gated Weighted RRF grid for #57 Phase 2.1.

This experiment intentionally reuses one already-built dense+sparse IndexVersion.
It changes only Qdrant RRF fusion weights (and optionally k), so no vectors are
rebuilt and no alias is activated. Each fusion contract receives an immutable
EvaluationRun and PromotionDecision against one shared `hybrid_rrf` baseline.
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

from benchmarks.promotion_evidence import assert_promotion_relevance_scope
from benchmarks.rag_native_compare import (
    RetrievedHit,
    audit_golden_dataset,
    load_golden_dataset,
    score_case,
    summarize_scores,
)
from benchmarks.retrieval_plan_promotion import dataset_version
from services.api.app.factory import build_services_from_env
from services.api.app.quality.contracts import (
    EvaluationRun,
    PromotionPolicy,
    stable_contract_id,
)
from services.api.app.quality.promotion import evaluate_promotion
from services.api.app.retrieval.contracts import (
    QdrantRrfFusionSpec,
    RetrievalPlan,
    RetrievalRequest,
)
from services.api.app.storage.quality_support import ensure_quality_repository


def _scoped_dataset(dataset: Mapping[str, Any], scope_doc_id: Optional[str]) -> dict[str, Any]:
    # JSON round-trip is deliberate: Golden Dataset content is JSON-compatible
    # and the scoped copy becomes the exact content hashed into EvaluationRun.
    scoped = json.loads(json.dumps(dataset, ensure_ascii=False))
    if scope_doc_id:
        defaults = scoped.setdefault("defaults", {})
        filters = defaults.setdefault("filters", {})
        existing = filters.get("doc_ids")
        if existing and list(existing) != [scope_doc_id]:
            raise ValueError(
                "--scope-doc-id conflicts with Golden Dataset defaults.filters.doc_ids"
            )
        filters["doc_ids"] = [scope_doc_id]
    return scoped


def parse_weight_pairs(value: str) -> list[QdrantRrfFusionSpec]:
    specs: list[QdrantRrfFusionSpec] = []
    seen: set[tuple[float, float]] = set()
    for token in str(value or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            dense_raw, sparse_raw = token.split(":", 1)
            pair = (float(dense_raw), float(sparse_raw))
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid RRF weight pair {token!r}; expected dense:sparse, e.g. 3:1"
            ) from exc
        if pair in seen:
            continue
        seen.add(pair)
        specs.append(QdrantRrfFusionSpec(dense_weight=pair[0], sparse_weight=pair[1]))
    if not specs:
        raise ValueError("At least one RRF weight pair is required")
    return specs


def _with_k(spec: QdrantRrfFusionSpec, k: int) -> QdrantRrfFusionSpec:
    return QdrantRrfFusionSpec(
        dense_weight=spec.dense_weight,
        sparse_weight=spec.sparse_weight,
        k=k,
    )


async def evaluate_weighted_rrf_grid(
    *,
    services: Any,
    dataset: Mapping[str, Any],
    candidate_index_version_id: str,
    tenant_id: str,
    fusion_specs: Sequence[QdrantRrfFusionSpec],
    scope_doc_id: Optional[str] = None,
    top_k: Optional[int] = None,
    candidate_pool: Optional[int] = None,
    rerank: bool = False,
    repetitions: int = 3,
    deadline_ms: int = 30000,
    code_revision: Optional[str] = None,
    policy: Optional[PromotionPolicy] = None,
    persist: bool = True,
) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if not fusion_specs:
        raise ValueError("fusion_specs must not be empty")

    scoped = _scoped_dataset(dataset, scope_doc_id)
    cases = [
        case
        for case in list(scoped.get("cases") or [])
        if isinstance(case, Mapping) and case.get("relevance")
    ]
    if not cases:
        raise ValueError("Golden Dataset requires labeled cases")
    scope_audit = assert_promotion_relevance_scope(scoped, cases)
    effective_top_k = int(top_k or (scoped.get("defaults") or {}).get("top_k") or 10)
    if effective_top_k < 10:
        raise ValueError("top_k must be >= 10 because promotion reports Recall/MRR/nDCG@10")

    lifecycle = getattr(services, "index_lifecycle", None)
    if lifecycle is None:
        raise RuntimeError("Weighted RRF promotion requires IndexLifecycleService")
    candidate = lifecycle.repo.get_index_version(candidate_index_version_id)
    active = lifecycle.repo.get_active_index_version(lifecycle.alias_name)
    if candidate is None:
        raise ValueError(f"Candidate IndexVersion not found: {candidate_index_version_id}")
    if active is None:
        raise RuntimeError("No active baseline IndexVersion")
    if candidate.index_version_id == active.index_version_id:
        raise ValueError("Weighted RRF candidate must remain distinct from the active baseline")
    if candidate.status not in {"validating", "ready"}:
        raise ValueError(f"Candidate is not evaluable: {candidate.status}")
    if candidate.embedding_contract_id != active.embedding_contract_id:
        raise ValueError("Weighted RRF experiment requires the same dense embedding contract")
    sparse_schema = dict((candidate.vector_schema or {}).get("sparse") or {})
    sparse_contract_id = str(sparse_schema.get("contract_id") or "")
    if not sparse_contract_id:
        raise ValueError("Candidate IndexVersion does not declare a sparse contract")

    repo = ensure_quality_repository(services.repo)
    revision = code_revision or _git_revision()
    version = dataset_version(scoped)
    policy_value = policy or PromotionPolicy()

    # Warm the exact paths outside measured evidence.
    warm_query = str(cases[0].get("query") or "").strip()
    if warm_query:
        filters = _filters(scoped, cases[0], tenant_id)
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
                fusion_spec=fusion_specs[0],
            )
        )

    baseline_evidence = await _run_plan(
        services,
        scoped,
        cases,
        tenant_id=tenant_id,
        top_k=effective_top_k,
        candidate_pool=candidate_pool,
        rerank=rerank,
        repetitions=repetitions,
        deadline_ms=deadline_ms,
        plan=RetrievalPlan.HYBRID_RRF,
        index_version_id=None,
        fusion_spec=None,
    )
    baseline_fusion_id = stable_contract_id(
        "fusion",
        {"provider_id": "ragbot", "method": "adaptive_rrf"},
    )
    common_config = {
        "top_k": effective_top_k,
        "candidate_pool": candidate_pool,
        "rerank": bool(rerank),
        "repetitions": int(repetitions),
        "deadline_ms": int(deadline_ms),
        "tenant_id": tenant_id,
        "scope_doc_id": scope_doc_id,
        "controlled_dense_embedding_contract_id": active.embedding_contract_id,
        "controlled_sparse_contract_id": sparse_contract_id,
        "controlled_candidate_index_version_id": candidate.index_version_id,
    }
    baseline_run = _evaluation_run(
        dataset=scoped,
        dataset_version_value=version,
        code_revision=revision,
        tenant_id=tenant_id,
        candidate_ref=None,
        baseline_ref=active.index_version_id,
        index_version=active,
        plan=RetrievalPlan.HYBRID_RRF,
        sparse_contract_id=None,
        fusion_contract_id=baseline_fusion_id,
        fusion_spec={"provider_id": "ragbot", "method": "adaptive_rrf"},
        evidence=baseline_evidence,
        config=common_config,
    )
    if persist:
        repo.add_evaluation_run(baseline_run)

    candidates: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for spec in fusion_specs:
        evidence = await _run_plan(
            services,
            scoped,
            cases,
            tenant_id=tenant_id,
            top_k=effective_top_k,
            candidate_pool=candidate_pool,
            rerank=rerank,
            repetitions=repetitions,
            deadline_ms=deadline_ms,
            plan=RetrievalPlan.QDRANT_DENSE_SPARSE,
            index_version_id=candidate.index_version_id,
            fusion_spec=spec,
        )
        candidate_run = _evaluation_run(
            dataset=scoped,
            dataset_version_value=version,
            code_revision=revision,
            tenant_id=tenant_id,
            candidate_ref=candidate.index_version_id,
            baseline_ref=active.index_version_id,
            index_version=candidate,
            plan=RetrievalPlan.QDRANT_DENSE_SPARSE,
            sparse_contract_id=sparse_contract_id,
            fusion_contract_id=spec.contract_id,
            fusion_spec=spec.as_dict(),
            evidence=evidence,
            config={**common_config, "fusion": spec.as_dict()},
        )
        decision = evaluate_promotion(baseline_run, candidate_run, policy_value)
        if persist:
            repo.add_evaluation_run(candidate_run)
            repo.add_promotion_decision(decision)
        item = {
            "fusion": spec.as_dict(),
            "evaluation": asdict(candidate_run),
            "promotion": asdict(decision),
        }
        candidates.append(item)
        if decision.decision == "accept":
            accepted.append(item)

    recommended = _recommend(accepted)
    return {
        "dataset": {
            "name": str(scoped.get("name") or "Golden Dataset"),
            "version": version,
            "cases": len(cases),
            "promotion_relevance_scope": scope_audit,
        },
        "baseline": asdict(baseline_run),
        "candidates": candidates,
        "accepted_count": len(accepted),
        "recommended_fusion_contract_id": (
            recommended["fusion"]["contract_id"] if recommended else None
        ),
        "candidate_marked_ready": False,
        "activation_performed": False,
    }


async def _run_plan(
    services: Any,
    dataset: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    *,
    tenant_id: str,
    top_k: int,
    candidate_pool: Optional[int],
    rerank: bool,
    repetitions: int,
    deadline_ms: int,
    plan: RetrievalPlan,
    index_version_id: Optional[str],
    fusion_spec: Optional[QdrantRrfFusionSpec],
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
        measured: list[float] = []
        case_trace: dict[str, Any] = {}
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
                    fusion_spec=fusion_spec,
                )
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)
            measured.append(elapsed_ms)
            trace = response.trace.as_dict()
            case_trace = case_trace or trace
            first_trace = first_trace or trace
            for stage, value in dict(trace.get("stage_ms") or {}).items():
                stage_samples.setdefault(str(stage), []).append(float(value))
            if first_hits is None:
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
                "trace": case_trace,
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
    fusion_contract_id: str,
    fusion_spec: Mapping[str, Any],
    evidence: Mapping[str, Any],
    config: Mapping[str, Any],
) -> EvaluationRun:
    summary = dict(evidence.get("summary") or {})
    runtime_contracts = {
        "retrieval_plan": plan.value,
        "index_version_id": index_version.index_version_id,
        "embedding_contract_id": index_version.embedding_contract_id,
        "sparse_contract_id": sparse_contract_id,
        "fusion_contract_id": fusion_contract_id,
        "fusion": dict(fusion_spec),
        "vector_schema": dict(index_version.vector_schema or {}),
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
    fusion_spec: Optional[QdrantRrfFusionSpec] = None,
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
        fusion_spec=fusion_spec,
    )


def _filters(dataset: Mapping[str, Any], case: Mapping[str, Any], tenant_id: str) -> dict[str, Any]:
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


def _recommend(items: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not items:
        return None

    def key(item: Mapping[str, Any]) -> tuple[float, float, float, float]:
        evaluation = item.get("evaluation") or {}
        metrics = evaluation.get("metrics") or {}
        latency = evaluation.get("latency") or {}
        return (
            float(metrics.get("ndcg") or 0.0),
            float(metrics.get("mrr") or 0.0),
            float(metrics.get("recall") or 0.0),
            -float(latency.get("p95_latency_ms") or float("inf")),
        )

    return max(items, key=key)


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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=5).strip()
    except Exception:
        return "unknown"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", required=True)
    parser.add_argument("--candidate-index", required=True)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--scope-doc-id")
    parser.add_argument("--weights", default="1:1,2:1,3:1,5:1")
    parser.add_argument("--rrf-k", type=int, default=2)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--candidate-pool", type=int)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--deadline-ms", type=int, default=30000)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--max-recall-drop", type=float, default=0.0)
    parser.add_argument("--max-mrr-drop", type=float, default=0.0)
    parser.add_argument("--max-ndcg-drop", type=float, default=0.0)
    parser.add_argument("--max-p95-latency-increase-ratio", type=float, default=0.15)
    parser.add_argument("--max-cost-increase-ratio", type=float, default=0.25)
    parser.add_argument("--min-recall", type=float)
    parser.add_argument("--min-mrr", type=float)
    parser.add_argument("--min-ndcg", type=float)
    parser.add_argument("--critical-case", action="append", default=[])
    return parser.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_golden_dataset(Path(args.golden))
    audit = audit_golden_dataset(dataset, profile="development")
    if not audit["passed"]:
        raise SystemExit(f"Golden Dataset audit failed: {audit}")
    specs = [_with_k(spec, args.rrf_k) for spec in parse_weight_pairs(args.weights)]
    services = build_services_from_env()
    try:
        return await evaluate_weighted_rrf_grid(
            services=services,
            dataset=dataset,
            candidate_index_version_id=args.candidate_index,
            tenant_id=args.tenant,
            fusion_specs=specs,
            scope_doc_id=args.scope_doc_id,
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
            rerank=bool(args.rerank),
            repetitions=args.repetitions,
            deadline_ms=args.deadline_ms,
            policy=PromotionPolicy(
                max_recall_drop=args.max_recall_drop,
                max_mrr_drop=args.max_mrr_drop,
                max_ndcg_drop=args.max_ndcg_drop,
                max_p95_latency_increase_ratio=args.max_p95_latency_increase_ratio,
                max_cost_increase_ratio=args.max_cost_increase_ratio,
                min_recall=args.min_recall,
                min_mrr=args.min_mrr,
                min_ndcg=args.min_ndcg,
                critical_case_ids=tuple(args.critical_case),
            ),
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
    return 0 if result["accepted_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
