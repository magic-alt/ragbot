"""Fail-closed facade for the #57 dense+sparse promotion runner.

The implementation is preserved in ``retrieval_plan_promotion_legacy`` to keep
its established CLI and report format stable. This facade adds evidence-scope
validation before any baseline/candidate query is executed so heuristic Golden
Dataset labels cannot silently produce invalid Recall/nDCG promotion evidence.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from . import retrieval_plan_promotion_legacy as _legacy
from .promotion_evidence import assert_promotion_relevance_scope

_unsafe_evaluate_dense_sparse_candidate = _legacy.evaluate_dense_sparse_candidate

# Preserve established public helpers/imports.
dataset_version = _legacy.dataset_version
parse_args = _legacy.parse_args


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
    policy: Any = None,
    persist: bool = True,
    mark_ready_on_accept: bool = False,
) -> dict[str, Any]:
    cases = [
        case
        for case in list(dataset.get("cases") or [])
        if isinstance(case, Mapping) and case.get("relevance")
    ]
    scope_audit = assert_promotion_relevance_scope(dataset, cases)
    result = await _unsafe_evaluate_dense_sparse_candidate(
        services=services,
        dataset=dataset,
        candidate_index_version_id=candidate_index_version_id,
        tenant_id=tenant_id,
        user_id=user_id,
        top_k=top_k,
        candidate_pool=candidate_pool,
        rerank=rerank,
        repetitions=repetitions,
        deadline_ms=deadline_ms,
        code_revision=code_revision,
        policy=policy,
        persist=persist,
        mark_ready_on_accept=mark_ready_on_accept,
    )
    result.setdefault("dataset", {})["promotion_relevance_scope"] = scope_audit
    result["evidence_integrity"] = {
        "promotion_eligible": True,
        "scope_audit": scope_audit,
    }
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    # The legacy CLI resolves its module-global function at runtime. Temporarily
    # replace it with the guarded facade so existing command lines keep working
    # without duplicating the mature CLI/service lifecycle implementation.
    original = _legacy.evaluate_dense_sparse_candidate
    _legacy.evaluate_dense_sparse_candidate = evaluate_dense_sparse_candidate
    try:
        return _legacy.main(argv)
    finally:
        _legacy.evaluate_dense_sparse_candidate = original


def __getattr__(name: str) -> Any:
    # Keep private helpers used by tests/extensions source-compatible while the
    # implementation remains in the legacy module.
    return getattr(_legacy, name)


if __name__ == "__main__":
    raise SystemExit(main())
