from __future__ import annotations

import asyncio

import pytest

from benchmarks.promotion_evidence import (
    assert_promotion_relevance_scope,
    audit_promotion_relevance_scope,
)
from benchmarks.retrieval_plan_promotion import evaluate_dense_sparse_candidate
from services.api.app.quality.contracts import EvaluationRun, PromotionPolicy
from services.api.app.quality.promotion import evaluate_promotion


def _heuristic_dataset(*, filters=None):
    dataset = {
        "schema_version": 1,
        "name": "heuristic-promotion",
        "defaults": {"top_k": 10},
        "cases": [
            {
                "id": "term-case",
                "category": "cross-lingual",
                "query": "缓存机制",
                "relevance": {"any_terms": ["cache hit", "caching mechanism"]},
            }
        ],
    }
    if filters is not None:
        dataset["defaults"]["filters"] = filters
    return dataset


def test_unscoped_term_relevance_is_not_promotion_eligible() -> None:
    dataset = _heuristic_dataset()
    audit = audit_promotion_relevance_scope(dataset)
    assert audit["promotion_eligible"] is False
    assert audit["ambiguous_cases"] == ["term-case"]
    with pytest.raises(ValueError, match="Promotion-ineligible Golden Dataset"):
        assert_promotion_relevance_scope(dataset)


def test_single_document_scope_makes_term_relevance_bounded() -> None:
    dataset = _heuristic_dataset(filters={"doc_ids": ["deepseek-doc"]})
    audit = assert_promotion_relevance_scope(dataset)
    assert audit["promotion_eligible"] is True
    assert audit["single_document_heuristic_cases"] == 1
    assert audit["ambiguous_cases"] == []


def test_explicit_document_labels_are_promotion_eligible_without_filter() -> None:
    dataset = {
        "cases": [
            {
                "id": "explicit",
                "query": "servo current loop",
                "relevance": {"doc_ids": ["servo-doc"]},
            }
        ]
    }
    audit = assert_promotion_relevance_scope(dataset)
    assert audit["explicit_cases"] == 1


def test_explicit_relevant_total_must_be_positive() -> None:
    dataset = {
        "cases": [
            {
                "id": "bad-total",
                "query": "servo",
                "relevance": {"any_terms": ["servo"], "relevant_total": 0},
            }
        ]
    }
    with pytest.raises(ValueError, match="relevant_total must be > 0"):
        assert_promotion_relevance_scope(dataset)


def _evaluation(run_id: str, *, ndcg: float) -> EvaluationRun:
    return EvaluationRun.build(
        evaluation_run_id=run_id,
        dataset_name="evidence-integrity",
        dataset_version="v1",
        code_revision="test",
        runtime_contracts={"retrieval_plan": run_id},
        metrics={"recall": 1.0, "mrr": 1.0, "ndcg": ndcg},
        latency={"p95_latency_ms": 100.0},
        cost={"cost_usd": 0.0},
    )


def test_promotion_rejects_impossible_ndcg_even_if_candidate_is_faster() -> None:
    baseline = _evaluation("baseline", ndcg=1.0)
    candidate = _evaluation("candidate", ndcg=1.4307)
    decision = evaluate_promotion(
        baseline,
        candidate,
        PromotionPolicy(max_p95_latency_increase_ratio=1.0),
    )
    assert decision.decision == "reject"
    assert any("invalid quality metric range: ndcg" in reason for reason in decision.reasons)
    assert decision.deltas["ndcg"]["valid"] is False


def test_dense_sparse_guard_rejects_ambiguous_dataset_before_services_are_used() -> None:
    with pytest.raises(ValueError, match="Promotion-ineligible Golden Dataset"):
        asyncio.run(
            evaluate_dense_sparse_candidate(
                services=object(),
                dataset=_heuristic_dataset(),
                candidate_index_version_id="idx-never-read",
                tenant_id="tenant-never-read",
                rerank=False,
                repetitions=1,
                persist=False,
            )
        )
