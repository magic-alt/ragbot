"""Durable quality and promotion-control contracts.

Keep package initialization dependency-light: storage capability adapters import
`quality.contracts`, while the runtime recorder imports storage adapters. Import
`QualityRecorder` explicitly from `quality.recorder` to avoid a package cycle.
"""

from .contracts import (
    EvaluationRun,
    PromotionDecision,
    PromotionPolicy,
    RagRun,
    stable_contract_id,
)
from .promotion import evaluate_promotion

__all__ = [
    "EvaluationRun",
    "PromotionDecision",
    "PromotionPolicy",
    "RagRun",
    "evaluate_promotion",
    "stable_contract_id",
]
