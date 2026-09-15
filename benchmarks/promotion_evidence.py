"""Fail-closed validation for retrieval promotion evidence.

Term/path/page relevance labels are useful for resilient Golden Datasets, but
without a bounded relevance universe they cannot support Recall or nDCG.  The
promotion path therefore requires either explicit relevance cardinality or an
exact single-document retrieval scope before any candidate query is executed.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _merged_filters(dataset: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    defaults = dataset.get("defaults") or {}
    result: dict[str, Any] = {}
    if isinstance(defaults, Mapping):
        result.update(dict(defaults.get("filters") or {}))
    result.update(dict(case.get("filters") or {}))
    return result


def _single_document_scope(filters: Mapping[str, Any]) -> bool:
    doc_ids = [str(value) for value in _as_list(filters.get("doc_ids")) if str(value)]
    return len(set(doc_ids)) == 1


def audit_promotion_relevance_scope(
    dataset: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    selected = list(cases if cases is not None else dataset.get("cases") or [])
    ambiguous: list[str] = []
    explicit: list[str] = []
    scoped_heuristic: list[str] = []

    for case in selected:
        case_id = str(case.get("id") or "<unknown>")
        rel = case.get("relevance") or {}
        if not isinstance(rel, Mapping):
            ambiguous.append(case_id)
            continue

        explicit_total = rel.get("relevant_total")
        if explicit_total is not None:
            try:
                value = int(explicit_total)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"case={case_id} relevance.relevant_total must be an integer > 0"
                ) from exc
            if value <= 0:
                raise ValueError(
                    f"case={case_id} relevance.relevant_total must be > 0"
                )
            explicit.append(case_id)
            continue

        # Exact chunk/document labels define the relevance universe directly.
        if rel.get("expected_chunk_ids") or rel.get("doc_ids"):
            explicit.append(case_id)
            continue

        heuristic = bool(
            rel.get("pages")
            or rel.get("path_contains")
            or rel.get("all_terms")
            or rel.get("any_terms")
        )
        if heuristic and _single_document_scope(_merged_filters(dataset, case)):
            scoped_heuristic.append(case_id)
            continue

        ambiguous.append(case_id)

    return {
        "promotion_eligible": not ambiguous,
        "cases": len(selected),
        "explicit_cases": len(explicit),
        "single_document_heuristic_cases": len(scoped_heuristic),
        "ambiguous_cases": ambiguous,
    }


def assert_promotion_relevance_scope(
    dataset: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    audit = audit_promotion_relevance_scope(dataset, cases)
    if not audit["promotion_eligible"]:
        raise ValueError(
            "Promotion-ineligible Golden Dataset: relevance universe is ambiguous for cases="
            + ",".join(audit["ambiguous_cases"])
            + ". Term/path/page labels require defaults.filters.doc_ids or "
            "case.filters.doc_ids scoped to exactly one document, or an explicit "
            "relevance.relevant_total / relevance.doc_ids / expected_chunk_ids contract."
        )
    return audit
