from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from .contracts import (
    EvaluationRun,
    PromotionDecision,
    PromotionPolicy,
    new_promotion_decision_id,
)


def evaluate_promotion(
    baseline: EvaluationRun,
    candidate: EvaluationRun,
    policy: PromotionPolicy,
    *,
    decision_id: Optional[str] = None,
) -> PromotionDecision:
    """Compare machine-readable evaluation evidence against regression limits.

    Missing metrics are treated as gate failures rather than silently promoted.
    Quality metrics outside their normalized [0, 1] domain are also rejected:
    they indicate invalid evaluation evidence (for example, an unbounded
    relevance universe that produced nDCG > 1), not candidate quality.
    """

    reasons: list[str] = []
    deltas: dict[str, Any] = {}

    _quality_gate(
        "recall",
        baseline.metrics,
        candidate.metrics,
        max_drop=policy.max_recall_drop,
        reasons=reasons,
        deltas=deltas,
    )
    _quality_gate(
        "mrr",
        baseline.metrics,
        candidate.metrics,
        max_drop=policy.max_mrr_drop,
        reasons=reasons,
        deltas=deltas,
    )
    _quality_gate(
        "ndcg",
        baseline.metrics,
        candidate.metrics,
        max_drop=policy.max_ndcg_drop,
        reasons=reasons,
        deltas=deltas,
    )
    _ratio_gate(
        "p95_latency_ms",
        baseline.latency,
        candidate.latency,
        max_increase_ratio=policy.max_p95_latency_increase_ratio,
        reasons=reasons,
        deltas=deltas,
    )
    _ratio_gate(
        "cost_usd",
        baseline.cost,
        candidate.cost,
        max_increase_ratio=policy.max_cost_increase_ratio,
        reasons=reasons,
        deltas=deltas,
        zero_baseline_is_missing=False,
    )

    _absolute_quality_gates(candidate, policy, reasons=reasons, deltas=deltas)
    _critical_case_gate(candidate, policy, reasons=reasons, deltas=deltas)

    return PromotionDecision(
        promotion_decision_id=decision_id or new_promotion_decision_id(),
        baseline_evaluation_id=baseline.evaluation_run_id,
        candidate_evaluation_id=candidate.evaluation_run_id,
        decision="reject" if reasons else "accept",
        policy=policy.as_dict(),
        deltas=deltas,
        reasons=reasons,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _absolute_quality_gates(
    candidate: EvaluationRun,
    policy: PromotionPolicy,
    *,
    reasons: list[str],
    deltas: dict[str, Any],
) -> None:
    configured = {
        "recall": policy.min_recall,
        "mrr": policy.min_mrr,
        "ndcg": policy.min_ndcg,
    }
    if not any(value is not None for value in configured.values()):
        return

    evidence: dict[str, Any] = {}
    for key, minimum in configured.items():
        if minimum is None:
            continue
        candidate_value = _number(candidate.metrics.get(key))
        passed = (
            candidate_value is not None
            and 0.0 <= candidate_value <= 1.0
            and candidate_value >= float(minimum)
        )
        evidence[key] = {
            "candidate": candidate_value,
            "minimum": float(minimum),
            "passed": passed,
        }
        if candidate_value is None:
            reasons.append(f"missing candidate absolute quality metric: {key}")
        elif not 0.0 <= candidate_value <= 1.0:
            reasons.append(
                f"invalid candidate absolute quality metric: {key}={candidate_value}; expected [0,1]"
            )
        elif candidate_value < float(minimum):
            reasons.append(
                f"{key} absolute floor {candidate_value:.6f} is below required {float(minimum):.6f}"
            )
    deltas["absolute_quality"] = evidence


def _critical_case_gate(
    candidate: EvaluationRun,
    policy: PromotionPolicy,
    *,
    reasons: list[str],
    deltas: dict[str, Any],
) -> None:
    required = list(policy.critical_case_ids)
    if not required:
        return

    raw_cases = candidate.artifacts.get("cases")
    case_items = raw_cases if isinstance(raw_cases, list) else []
    by_id: dict[str, Mapping[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for item in case_items:
        if not isinstance(item, Mapping):
            continue
        case_id = str(item.get("case_id") or "").strip()
        if not case_id:
            continue
        if case_id in by_id:
            duplicate_ids.add(case_id)
            continue
        by_id[case_id] = item

    passed: list[str] = []
    failed: list[dict[str, Any]] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    for case_id in required:
        if case_id in duplicate_ids:
            ambiguous.append(case_id)
            reasons.append(f"critical case evidence ambiguous: {case_id}")
            continue
        item = by_id.get(case_id)
        if item is None:
            missing.append(case_id)
            reasons.append(f"critical case evidence missing: {case_id}")
            continue
        if item.get("retrieval_pass") is True:
            passed.append(case_id)
            continue
        failure = {
            "case_id": case_id,
            "first_relevant_rank": item.get("first_relevant_rank"),
        }
        failed.append(failure)
        reasons.append(
            f"critical case failed: {case_id} rank={item.get('first_relevant_rank')}"
        )

    deltas["critical_cases"] = {
        "required": required,
        "passed": passed,
        "failed": failed,
        "missing": missing,
        "ambiguous": ambiguous,
    }


def _quality_gate(
    key: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    max_drop: float,
    reasons: list[str],
    deltas: dict[str, Any],
) -> None:
    base = _number(baseline.get(key))
    cand = _number(candidate.get(key))
    if base is None or cand is None:
        reasons.append(f"missing required quality metric: {key}")
        deltas[key] = {"baseline": base, "candidate": cand, "delta": None}
        return
    if not 0.0 <= base <= 1.0 or not 0.0 <= cand <= 1.0:
        reasons.append(
            f"invalid quality metric range: {key} baseline={base} candidate={cand}; expected [0,1]"
        )
        deltas[key] = {
            "baseline": base,
            "candidate": cand,
            "delta": None,
            "valid": False,
        }
        return
    delta = cand - base
    deltas[key] = {"baseline": base, "candidate": cand, "delta": delta, "valid": True}
    if delta < -abs(float(max_drop)):
        reasons.append(
            f"{key} regression {delta:.6f} exceeds allowed drop {abs(float(max_drop)):.6f}"
        )


def _ratio_gate(
    key: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    max_increase_ratio: float,
    reasons: list[str],
    deltas: dict[str, Any],
    zero_baseline_is_missing: bool = True,
) -> None:
    base = _number(baseline.get(key))
    cand = _number(candidate.get(key))
    if base is None or cand is None or (zero_baseline_is_missing and base <= 0):
        reasons.append(f"missing required comparison metric: {key}")
        deltas[key] = {"baseline": base, "candidate": cand, "increase_ratio": None}
        return
    if base == 0:
        ratio = 0.0 if cand == 0 else float("inf")
    else:
        ratio = (cand - base) / base
    deltas[key] = {"baseline": base, "candidate": cand, "increase_ratio": ratio}
    if ratio > float(max_increase_ratio):
        reasons.append(
            f"{key} increase ratio {ratio:.6f} exceeds allowed {float(max_increase_ratio):.6f}"
        )


def _number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
