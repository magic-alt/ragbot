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
    This keeps new retrieval/index plans opt-in until evidence is complete.
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
    delta = cand - base
    deltas[key] = {"baseline": base, "candidate": cand, "delta": delta}
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
