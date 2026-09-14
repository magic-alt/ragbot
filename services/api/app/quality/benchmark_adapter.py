from __future__ import annotations

from typing import Any, Mapping, Optional

from .contracts import EvaluationRun, stable_contract_id


def evaluation_from_native_report(
    report: Mapping[str, Any],
    *,
    backend: str,
    code_revision: str,
    candidate_ref: Optional[str] = None,
    baseline_ref: Optional[str] = None,
    dataset_version: Optional[str] = None,
    runtime_contracts: Optional[Mapping[str, Any]] = None,
    cost_usd: float = 0.0,
) -> EvaluationRun:
    """Convert `benchmarks.rag_native_compare` JSON into an immutable EvaluationRun.

    The adapter does not replace the benchmark. It normalizes its machine-readable
    output into the durable #61 control-plane schema used by promotion gates.
    """

    results = list(report.get("results") or [])
    selected = next(
        (item for item in results if str(item.get("backend") or "") == backend),
        None,
    )
    if selected is None:
        raise ValueError(f"Benchmark report does not contain backend={backend!r}")
    summary = dict(selected.get("summary") or {})
    dataset = dict(report.get("dataset") or {})
    configuration = dict(report.get("configuration") or {})
    corpus = dict(report.get("corpus_manifest") or {})
    runtime = dict(selected.get("runtime") or {})
    contracts = dict(runtime_contracts or {})
    for key in (
        "retrieval_plan",
        "embedding_contract_id",
        "index_version_id",
        "reranker_contract_id",
    ):
        if runtime.get(key) is not None and key not in contracts:
            contracts[key] = runtime.get(key)
    if runtime.get("retrieval_mode") is not None and "retrieval_plan" not in contracts:
        contracts["retrieval_plan"] = runtime.get("retrieval_mode")

    resolved_dataset_version = dataset_version or str(dataset.get("version") or "").strip()
    if not resolved_dataset_version:
        resolved_dataset_version = stable_contract_id(
            "dataset",
            {
                "name": dataset.get("name"),
                "cases": dataset.get("cases"),
                "corpus_sha256": corpus.get("sha256"),
            },
        )

    metrics = {
        "recall": summary.get("recall_at_10"),
        "mrr": summary.get("mrr_at_10"),
        "ndcg": summary.get("ndcg_at_10"),
        "hit_at_1": summary.get("hit_at_1"),
        "hit_at_3": summary.get("hit_at_3"),
        "hit_at_5": summary.get("hit_at_5"),
        "hit_at_10": summary.get("hit_at_10"),
        "pass_rate": summary.get("pass_rate"),
        "categories": summary.get("categories") or {},
    }
    latency = {
        "p50_latency_ms": summary.get("query_latency_ms_p50"),
        "p95_latency_ms": summary.get("query_latency_ms_p95"),
        "mean_latency_ms": summary.get("query_latency_ms_mean"),
        "queries_per_second": summary.get("queries_per_second"),
    }
    artifacts = {
        "benchmark_schema_version": report.get("schema_version"),
        "corpus_manifest": corpus,
        "backend_build": selected.get("build") or {},
        "backend_versions": configuration.get("backend_versions") or {},
        "case_count": summary.get("cases"),
    }
    config = {
        "backend": backend,
        "top_k": configuration.get("top_k"),
        "ragbot_mode": configuration.get("ragbot_mode"),
        "rerank": configuration.get("rerank"),
        "embedding_model": configuration.get("embedding_model"),
        "embedding_dimension": configuration.get("embedding_dimension"),
        "chunk_size": configuration.get("chunk_size"),
        "chunk_overlap": configuration.get("chunk_overlap"),
        "repetitions": configuration.get("repetitions"),
    }
    return EvaluationRun.build(
        dataset_name=str(dataset.get("name") or "Golden Dataset"),
        dataset_version=resolved_dataset_version,
        code_revision=code_revision,
        candidate_ref=candidate_ref or backend,
        baseline_ref=baseline_ref or str(report.get("baseline") or "") or None,
        runtime_contracts=contracts,
        config=config,
        metrics=metrics,
        latency=latency,
        cost={"cost_usd": float(cost_usd)},
        artifacts=artifacts,
    )
