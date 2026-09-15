from __future__ import annotations

from benchmarks.identifier_relevance import canonicalize_identifier, identifier_matches
from benchmarks.promotion_evidence import audit_promotion_relevance_scope
from benchmarks.rag_native_compare import RetrievedHit, _match_relevance
from scripts.rag_eval import _chunk_relevant


def test_identifier_canonicalization_unifies_pdf_and_query_separators() -> None:
    expected = "deepseekv3"
    variants = (
        "DeepSeek-V3",
        "DeepSeek\u2011V3",  # U+2011 NON-BREAKING HYPHEN from the real PDF
        "DeepSeek\u2013V3",
        "DeepSeek\u2014V3",
        "DeepSeek_V3",
        "DeepSeek V3",
        "DeepSeek-\nV3",
        "DeepSeek\u2011\nV3",
    )
    assert {canonicalize_identifier(value) for value in variants} == {expected}
    assert all(identifier_matches("The model is DeepSeek\u2011V3.", value) for value in variants)


def test_identifier_relevance_matches_non_breaking_hyphen_in_native_evaluator() -> None:
    case = {
        "id": "identifier-deepseek-v3",
        "category": "identifier",
        "query": "DeepSeek-V3",
        "relevance": {"identifiers": ["DeepSeek-V3"], "max_rank": 5},
    }
    hit = RetrievedHit(
        chunk_id="chunk-1",
        doc_id="deepseek.pdf",
        path="DeepSeek in Action.pdf",
        text="This section introduces DeepSeek\u2011V3 and its architecture.",
    )
    assert _match_relevance(hit, case) is True


def test_identifier_relevance_matches_non_breaking_hyphen_in_live_eval() -> None:
    case = {
        "id": "identifier-deepseek-v3",
        "category": "identifier",
        "query": "DeepSeek-V3",
        "relevance": {"identifiers": ["DeepSeek-V3"], "max_rank": 5},
    }
    chunk = {
        "chunk_id": "chunk-1",
        "doc_id": "deepseek.pdf",
        "text": "DeepSeek\u2011V3 supports the described training setup.",
        "metadata": {"path": "DeepSeek in Action.pdf"},
    }
    assert _chunk_relevant(chunk, case) is True


def test_any_terms_semantics_remain_literal_and_are_not_identifier_canonicalized() -> None:
    case = {
        "id": "literal-term",
        "query": "DeepSeek-V3",
        "relevance": {"any_terms": ["deepseek-v3"]},
    }
    hit = RetrievedHit(
        chunk_id="chunk-1",
        doc_id="deepseek.pdf",
        path="DeepSeek in Action.pdf",
        text="DeepSeek\u2011V3 uses a non-breaking hyphen here.",
    )
    chunk = {
        "chunk_id": "chunk-1",
        "doc_id": "deepseek.pdf",
        "text": hit.text,
        "metadata": {"path": hit.path},
    }
    assert _match_relevance(hit, case) is False
    assert _chunk_relevant(chunk, case) is False


def test_identifier_promotion_evidence_requires_single_document_scope() -> None:
    case = {
        "id": "identifier-deepseek-v3",
        "query": "DeepSeek-V3",
        "relevance": {"identifiers": ["DeepSeek-V3"]},
    }
    unscoped = audit_promotion_relevance_scope({"cases": [case]})
    assert unscoped["promotion_eligible"] is False
    assert unscoped["ambiguous_cases"] == ["identifier-deepseek-v3"]

    scoped = audit_promotion_relevance_scope(
        {
            "defaults": {"filters": {"doc_ids": ["deepseek.pdf"]}},
            "cases": [case],
        }
    )
    assert scoped["promotion_eligible"] is True
    assert scoped["single_document_heuristic_cases"] == 1
