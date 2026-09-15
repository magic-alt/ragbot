from __future__ import annotations

import pytest

from services.api.app.quality.contracts import EvaluationRun, PromotionPolicy
from services.api.app.quality.promotion import evaluate_promotion


def _evaluation(
    run_id: str,
    *,
    recall: float,
    mrr: float,
    ndcg: float,
    cases: list[dict] | None = None,
) -> EvaluationRun:
    return EvaluationRun.build(
        evaluation_run_id=run_id,
        dataset_name="DeepSeek in Action retrieval quality v2",
        dataset_version="sha256:test",
        code_revision="test-sha",
        runtime_contracts={"retrieval_plan": "qdrant_dense_sparse"},
        metrics={"recall": recall, "mrr": mrr, "ndcg": ndcg},
        latency={"p95_latency_ms": 158.4 if run_id == "candidate" else 241.409},
        cost={"cost_usd": 0.0},
        artifacts={"cases": list(cases or [])},
    )


def _real_experiment_pair(*, identifier_pass: bool = False) -> tuple[EvaluationRun, EvaluationRun]:
    baseline = _evaluation(
        "baseline",
        recall=0.95,
        mrr=0.925,
        ndcg=0.9315,
        cases=[
            {
                "case_id": "identifier-deepseek-v3",
                "retrieval_pass": False,
                "first_relevant_rank": None,
            }
        ],
    )
    candidate = _evaluation(
        "candidate",
        recall=0.95,
        mrr=0.95,
        ndcg=0.95,
        cases=[
            {
                "case_id": "identifier-deepseek-v3",
                "retrieval_pass": identifier_pass,
                "first_relevant_rank": 1 if identifier_pass else None,
            }
        ],
    )
    return baseline, candidate


def test_existing_relative_policy_remains_backward_compatible() -> None:
    baseline, candidate = _real_experiment_pair(identifier_pass=False)
    decision = evaluate_promotion(baseline, candidate, PromotionPolicy())
    assert decision.decision == "accept"
    assert decision.reasons == []
    assert "absolute_quality" not in decision.deltas
    assert "critical_cases" not in decision.deltas


def test_absolute_quality_floor_can_reject_relative_improvement() -> None:
    baseline, candidate = _real_experiment_pair(identifier_pass=True)
    decision = evaluate_promotion(
        baseline,
        candidate,
        PromotionPolicy(min_recall=1.0),
    )
    assert decision.decision == "reject"
    assert any("recall absolute floor" in reason for reason in decision.reasons)
    assert decision.deltas["absolute_quality"]["recall"] == {
        "candidate": 0.95,
        "minimum": 1.0,
        "passed": False,
    }


def test_critical_case_failure_rejects_relative_accept() -> None:
    baseline, candidate = _real_experiment_pair(identifier_pass=False)
    decision = evaluate_promotion(
        baseline,
        candidate,
        PromotionPolicy(critical_case_ids=("identifier-deepseek-v3",)),
    )
    assert decision.decision == "reject"
    assert any(
        "critical case failed: identifier-deepseek-v3" in reason
        for reason in decision.reasons
    )
    critical = decision.deltas["critical_cases"]
    assert critical["passed"] == []
    assert critical["failed"] == [
        {"case_id": "identifier-deepseek-v3", "first_relevant_rank": None}
    ]
    assert critical["missing"] == []


def test_missing_critical_case_evidence_fails_closed() -> None:
    baseline, candidate = _real_experiment_pair(identifier_pass=True)
    decision = evaluate_promotion(
        baseline,
        candidate,
        PromotionPolicy(critical_case_ids=("must-exist",)),
    )
    assert decision.decision == "reject"
    assert "critical case evidence missing: must-exist" in decision.reasons
    assert decision.deltas["critical_cases"]["missing"] == ["must-exist"]


def test_release_gates_accept_when_absolute_and_critical_requirements_pass() -> None:
    baseline, candidate = _real_experiment_pair(identifier_pass=True)
    decision = evaluate_promotion(
        baseline,
        candidate,
        PromotionPolicy(
            min_recall=0.95,
            min_mrr=0.95,
            min_ndcg=0.95,
            critical_case_ids=("identifier-deepseek-v3",),
        ),
    )
    assert decision.decision == "accept"
    assert decision.reasons == []
    assert decision.deltas["critical_cases"]["passed"] == ["identifier-deepseek-v3"]
    assert all(
        item["passed"]
        for item in decision.deltas["absolute_quality"].values()
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_recall": -0.01},
        {"min_recall": 1.01},
        {"min_mrr": -1.0},
        {"min_ndcg": 2.0},
    ],
)
def test_absolute_quality_floor_must_be_normalized(kwargs: dict) -> None:
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        PromotionPolicy(**kwargs)


def test_critical_case_ids_are_trimmed_deduplicated_and_serialized() -> None:
    policy = PromotionPolicy(
        min_recall=0.9,
        critical_case_ids=(" identifier-deepseek-v3 ", "identifier-deepseek-v3", "fp8"),
    )
    assert policy.critical_case_ids == ("identifier-deepseek-v3", "fp8")
    payload = policy.as_dict()
    assert payload["min_recall"] == 0.9
    assert payload["min_mrr"] is None
    assert payload["critical_case_ids"] == ["identifier-deepseek-v3", "fp8"]


def test_blank_critical_case_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="critical_case_ids"):
        PromotionPolicy(critical_case_ids=(" ",))
