from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest

from services.api.app.quality.contracts import EvaluationRun, PromotionPolicy, RagRun
from services.api.app.quality.promotion import evaluate_promotion
from services.api.app.storage.pg_repo import PostgresRepo
from services.api.app.storage.quality_support import ensure_quality_repository

pytestmark = pytest.mark.skipif(
    not os.getenv("POSTGRES_TEST_DSN"),
    reason="POSTGRES_TEST_DSN not configured",
)


def test_postgres_rag_run_evaluation_feedback_and_promotion_roundtrip() -> None:
    repo = PostgresRepo(os.environ["POSTGRES_TEST_DSN"], pool_min=1, pool_max=2)
    ensure_quality_repository(repo)
    suffix = uuid.uuid4().hex[:12]
    request_id = f"quality-request-{suffix}"
    tenant_id = f"quality-tenant-{suffix}"
    user_id = f"quality-user-{suffix}"
    now = datetime.now(timezone.utc).isoformat()
    try:
        run = RagRun(
            request_id=request_id,
            trace_id=f"trace-{suffix}",
            tenant_id=tenant_id,
            user_id=user_id,
            run_kind="search",
            status="completed",
            query_hash="a" * 64,
            retrieval_plan="hybrid_rrf",
            retrieval_contract_id="retrieval:v1:test",
            embedding_contract_id="embedding:v1:test",
            index_version_id=f"index-{suffix}",
            reranker_contract_id="reranker:v1:test",
            stage_latency_ms={"dense.search": 4.2, "lexical.search": 3.1},
            retrieved=[{"chunk_id": "chunk-1", "final_rank": 1, "score": 0.9}],
            citations=[{"citation_id": "doc-1:chunk-1", "chunk_id": "chunk-1"}],
            usage={"total_tokens": 42, "estimated_cost_usd": 0.001},
            trace={"retrieval_plan": "hybrid_rrf"},
            total_duration_ms=15,
            started_at=now,
            completed_at=now,
        )
        repo.add_rag_run(run)
        loaded = repo.get_rag_run(request_id)
        assert loaded is not None
        assert loaded.retrieval_plan == "hybrid_rrf"
        assert loaded.embedding_contract_id == "embedding:v1:test"
        assert loaded.retrieved[0]["chunk_id"] == "chunk-1"

        repo.add_quality_feedback(
            feedback_id=f"feedback-{suffix}",
            request_id=request_id,
            tenant_id=tenant_id,
            user_id=user_id,
            feedback_type="positive",
            citation_id="doc-1:chunk-1",
            rating=1.0,
            metadata={"source": "integration-test"},
        )

        baseline = EvaluationRun.build(
            evaluation_run_id=f"eval-base-{suffix}",
            dataset_name="golden",
            dataset_version="v1",
            code_revision="base-sha",
            candidate_ref="hybrid_rrf",
            runtime_contracts={"retrieval_plan": "hybrid_rrf"},
            metrics={"recall": 0.9, "mrr": 0.8, "ndcg": 0.82},
            latency={"p95_latency_ms": 100.0},
            cost={"cost_usd": 1.0},
        )
        candidate = EvaluationRun.build(
            evaluation_run_id=f"eval-candidate-{suffix}",
            dataset_name="golden",
            dataset_version="v1",
            code_revision="candidate-sha",
            candidate_ref="dense-sparse",
            baseline_ref="hybrid_rrf",
            runtime_contracts={"retrieval_plan": "qdrant_dense_sparse"},
            metrics={"recall": 0.91, "mrr": 0.81, "ndcg": 0.83},
            latency={"p95_latency_ms": 108.0},
            cost={"cost_usd": 1.1},
        )
        repo.add_evaluation_run(baseline)
        repo.add_evaluation_run(candidate)
        assert repo.get_evaluation_run(candidate.evaluation_run_id).evaluation_contract_id == candidate.evaluation_contract_id

        decision = evaluate_promotion(baseline, candidate, PromotionPolicy())
        repo.add_promotion_decision(decision)
        loaded_decision = repo.get_promotion_decision(decision.promotion_decision_id)
        assert loaded_decision is not None
        assert loaded_decision.decision == "accept"

        recent = repo.list_rag_runs(tenant_id=tenant_id, limit=10)
        assert [item.request_id for item in recent] == [request_id]
    finally:
        repo.close()
