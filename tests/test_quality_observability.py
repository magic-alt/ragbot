from __future__ import annotations

import asyncio
from types import SimpleNamespace

from services.api.app.agent.graph import build_default_services, run_agent
from services.api.app.llm.contracts import ModelCapabilities, ModelUsage, UsageMixin
from services.api.app.llm.router import ModelRouter, model_task_scope
from services.api.app.quality.benchmark_adapter import evaluation_from_native_report
from services.api.app.quality.contracts import (
    EvaluationRun,
    PromotionPolicy,
    stable_contract_id,
)
from services.api.app.quality.promotion import evaluate_promotion
from services.api.app.quality.recorder import QualityRecorder
from services.api.app.storage.quality_support import ensure_quality_repository
from services.api.app.storage.repo import InMemoryRepo


def test_contract_ids_are_order_independent_and_configuration_sensitive() -> None:
    left = stable_contract_id("retrieval", {"plan": "hybrid_rrf", "top_k": 20})
    reordered = stable_contract_id("retrieval", {"top_k": 20, "plan": "hybrid_rrf"})
    changed = stable_contract_id("retrieval", {"plan": "hybrid_rrf", "top_k": 30})
    assert left == reordered
    assert left != changed


def test_search_recorder_persists_lineage_without_query_or_chunk_text(monkeypatch) -> None:
    monkeypatch.setenv("RAGBOT_TRACE_STORE_CONTENT", "false")
    monkeypatch.setenv("RAGBOT_TRACE_DETAIL_SAMPLE_RATE", "0")
    repo = ensure_quality_repository(InMemoryRepo())
    services = SimpleNamespace(
        repo=repo,
        qdrant=SimpleNamespace(alias_name=None),
        embedder=SimpleNamespace(contract_id="embedding:v1:test"),
        reranker=SimpleNamespace(enabled=False),
    )
    request = SimpleNamespace(
        plan=SimpleNamespace(value="hybrid_rrf"),
        top_k=5,
        candidate_pool=20,
        rerank=True,
        diversity=False,
    )
    trace = SimpleNamespace(
        as_dict=lambda: {
            "retrieval_plan": "hybrid_rrf",
            "stage_ms": {"dense.search": 3.1, "lexical.search": 2.2},
        }
    )
    chunk = SimpleNamespace(
        chunk_id="chunk-1",
        doc_id="doc-1",
        text="sensitive document body must not enter durable lineage",
        score=0.91,
        citations=["doc-1:chunk-1"],
        metadata={
            "_retrieval": {
                "final_rank": 1,
                "fusion_score": 0.1,
                "dense": {"rank": 1, "raw_score": 0.91},
            }
        },
    )
    response = SimpleNamespace(trace=trace, chunks=[chunk])

    run = QualityRecorder(repo).record_search(
        services=services,
        request_id="request-1",
        tenant_id="tenant-a",
        user_id="user-a",
        query="private query text",
        request=request,
        response=response,
    )

    persisted = repo.get_rag_run("request-1")
    assert persisted == run
    assert run.query_text is None
    assert run.query_hash
    assert run.embedding_contract_id == "embedding:v1:test"
    assert run.retrieval_contract_id
    assert run.trace_sampled is False
    assert run.trace == {}
    assert run.retrieved[0]["chunk_id"] == "chunk-1"
    assert "text" not in run.retrieved[0]
    assert "candidate_trace" not in run.retrieved[0]


def test_evaluation_runs_are_immutable_and_promotion_is_evidence_based() -> None:
    repo = ensure_quality_repository(InMemoryRepo())
    baseline = EvaluationRun.build(
        evaluation_run_id="eval-baseline",
        dataset_name="golden",
        dataset_version="v1",
        code_revision="base-sha",
        runtime_contracts={"retrieval_plan": "hybrid_rrf"},
        metrics={"recall": 0.90, "mrr": 0.80, "ndcg": 0.82},
        latency={"p95_latency_ms": 100.0},
        cost={"cost_usd": 1.0},
    )
    candidate = EvaluationRun.build(
        evaluation_run_id="eval-candidate",
        dataset_name="golden",
        dataset_version="v1",
        code_revision="candidate-sha",
        runtime_contracts={"retrieval_plan": "qdrant_dense_sparse"},
        metrics={"recall": 0.92, "mrr": 0.81, "ndcg": 0.83},
        latency={"p95_latency_ms": 110.0},
        cost={"cost_usd": 1.10},
    )
    repo.add_evaluation_run(baseline)
    repo.add_evaluation_run(candidate)

    accepted = evaluate_promotion(baseline, candidate, PromotionPolicy())
    assert accepted.decision == "accept"
    assert not accepted.reasons
    repo.add_promotion_decision(accepted)
    assert repo.get_promotion_decision(accepted.promotion_decision_id) == accepted

    regressed = EvaluationRun.build(
        evaluation_run_id="eval-regressed",
        dataset_name="golden",
        dataset_version="v1",
        code_revision="regressed-sha",
        runtime_contracts={"retrieval_plan": "qdrant_dense_sparse"},
        metrics={"recall": 0.80, "mrr": 0.79, "ndcg": 0.80},
        latency={"p95_latency_ms": 130.0},
        cost={"cost_usd": 1.0},
    )
    rejected = evaluate_promotion(baseline, regressed, PromotionPolicy())
    assert rejected.decision == "reject"
    assert any("recall regression" in reason for reason in rejected.reasons)
    assert any("p95_latency_ms" in reason for reason in rejected.reasons)

    try:
        repo.add_evaluation_run(baseline)
    except ValueError as exc:
        assert "immutable" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("EvaluationRun overwrite must be rejected")


def test_native_benchmark_report_maps_to_promotion_metrics() -> None:
    report = {
        "schema_version": 1,
        "baseline": "ragbot",
        "dataset": {"name": "robotics-golden", "cases": 12},
        "corpus_manifest": {"sha256": "abc123"},
        "configuration": {
            "top_k": 10,
            "ragbot_mode": "hybrid",
            "rerank": True,
            "embedding_model": "multilingual-e5",
            "embedding_dimension": 1024,
            "backend_versions": {"ragbot": "0.5.0"},
        },
        "results": [
            {
                "backend": "ragbot",
                "runtime": {
                    "retrieval_plan": "hybrid_rrf",
                    "embedding_contract_id": "embed-1",
                    "index_version_id": "index-1",
                },
                "build": {"kind": "live-index"},
                "summary": {
                    "cases": 12,
                    "pass_rate": 1.0,
                    "hit_at_1": 0.8,
                    "hit_at_3": 0.9,
                    "hit_at_5": 1.0,
                    "hit_at_10": 1.0,
                    "mrr_at_10": 0.88,
                    "recall_at_10": 0.95,
                    "ndcg_at_10": 0.91,
                    "query_latency_ms_p50": 20.0,
                    "query_latency_ms_p95": 45.0,
                    "query_latency_ms_mean": 25.0,
                    "queries_per_second": 40.0,
                    "categories": {"exact": {"cases": 6}},
                },
            }
        ],
    }
    run = evaluation_from_native_report(
        report,
        backend="ragbot",
        code_revision="sha-1",
        candidate_ref="hybrid_rrf",
        cost_usd=0.02,
    )
    assert run.metrics["recall"] == 0.95
    assert run.metrics["mrr"] == 0.88
    assert run.metrics["ndcg"] == 0.91
    assert run.latency["p95_latency_ms"] == 45.0
    assert run.runtime_contracts["index_version_id"] == "index-1"
    assert run.cost["cost_usd"] == 0.02


class _UsageProvider(UsageMixin):
    def __init__(self) -> None:
        super().__init__()
        self.provider_id = "fake"
        self.model_id = "fake-model"
        self.enabled = True
        self.capabilities = ModelCapabilities(structured_output=True, streaming=True)

    async def chat_json(self, **kwargs):
        self._record_usage(ModelUsage(input_tokens=10, output_tokens=4))
        return {"ok": True}

    async def stream_text(self, **kwargs):
        self._record_usage(ModelUsage(input_tokens=2, output_tokens=1))
        yield "ok"

    async def web_search(self, *args, **kwargs):
        return []


def test_model_usage_is_correlated_to_request_id() -> None:
    provider = _UsageProvider()
    router = ModelRouter(provider, provider, routing_enabled=False)

    async def run():
        with model_task_scope(router, "synthesize", request_id="req-42"):
            await router.chat_json("system", "user", {"type": "object"})

    asyncio.run(run())
    records = router.cost_tracker.records_for_request("req-42")
    assert len(records) == 1
    assert records[0].provider == "fake"
    assert records[0].model == "fake-model"
    assert records[0].total_tokens == 14


def test_agent_run_is_persisted_by_request_id_without_raw_query() -> None:
    services = build_default_services()
    state = asyncio.run(run_agent("servo retrieval evidence", "tenant-a", "user-a", services))
    repo = ensure_quality_repository(services.repo)
    run = repo.get_rag_run(state.request_id)
    assert run is not None
    assert run.run_kind == "agent"
    assert run.status == "completed"
    assert run.query_hash
    assert run.query_text is None
