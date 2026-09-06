from __future__ import annotations

import hashlib

from benchmarks.component_compare import RawPdf
from benchmarks.pdf_promotion import (
    PromotionThresholds,
    audit_pdf_structure_suite,
    build_suite_skeleton,
    evaluate_promotion_gate,
)


def _documents() -> list[RawPdf]:
    return [
        RawPdf("book.pdf", "book.pdf", b"book", 10),
        RawPdf("sheet.pdf", "sheet.pdf", b"sheet", 5),
    ]


def _production_dataset() -> dict:
    docs = _documents()
    cases = []
    tags = ["numeric", "table", "page-specific"]
    categories = ["exact", "paraphrase", "cross-lingual"]
    for index in range(50):
        doc = docs[index % len(docs)]
        cases.append(
            {
                "id": f"case-{index:02d}",
                "category": categories[index % len(categories)],
                "tags": [tags[index % len(tags)]],
                "query": f"question {index}",
                "relevance": {"doc_ids": [doc.doc_id], "max_rank": 5},
                "review": {"status": "approved" if index < 45 else "draft"},
            }
        )
    return {
        "schema_version": 1,
        "name": "test suite",
        "pdf_structure_suite": {
            "version": 1,
            "requirements": {
                "min_documents": 2,
                "min_reviewed_rate": 0.8,
                "required_document_types": ["technical_book", "datasheet"],
                "required_case_tags": ["numeric", "table", "page-specific"],
            },
            "documents": [
                {
                    "path": "book.pdf",
                    "sha256": hashlib.sha256(b"book").hexdigest(),
                    "page_count": 10,
                    "document_type": "technical_book",
                    "languages": ["en"],
                },
                {
                    "path": "sheet.pdf",
                    "sha256": hashlib.sha256(b"sheet").hexdigest(),
                    "page_count": 5,
                    "document_type": "datasheet",
                    "languages": ["en"],
                },
            ],
        },
        "cases": cases,
    }


def _pipeline(name: str, *, mrr: float = 1.0, non_boundary: float = 0.8) -> dict:
    return {
        "name": name,
        "chunks": 100,
        "quality": {
            "hit_at_5": 1.0,
            "mrr_at_10": mrr,
            "ndcg_at_10": 1.0,
            "categories": {
                "exact": {"mrr_at_10": mrr},
                "paraphrase": {"mrr_at_10": mrr},
            },
        },
        "fidelity": {"label_text_coverage": 1.0, "page_metadata_coverage": 1.0},
        "timing": {"embedding_seconds": 10.0},
        "chunk_shape": {"non_boundary_end_rate": non_boundary},
        "memory": {"tracemalloc_peak_bytes": 1000},
        "cases": [
            {
                "id": "a",
                "category": "exact",
                "tags": ["numeric"],
                "retrieval_pass": True,
                "first_relevant_rank": 1,
            },
            {
                "id": "b",
                "category": "paraphrase",
                "tags": ["table"],
                "retrieval_pass": True,
                "first_relevant_rank": 1,
            },
        ],
    }


def test_production_suite_audit_requires_reviewed_stable_source_pinned_cases():
    audit = audit_pdf_structure_suite(_production_dataset(), _documents(), profile="production")
    assert audit["passed"] is True
    assert audit["golden_dataset"]["stats"]["cases"] == 50
    assert audit["golden_dataset"]["stats"]["stable_label_rate"] == 1.0
    assert audit["stats"]["approved_review_rate"] == 0.9


def test_production_suite_audit_rejects_corpus_hash_mismatch():
    dataset = _production_dataset()
    dataset["pdf_structure_suite"]["documents"][0]["sha256"] = "0" * 64
    audit = audit_pdf_structure_suite(dataset, _documents(), profile="production")
    assert audit["passed"] is False
    assert any(item["name"] == "corpus_file_identity" and not item["passed"] for item in audit["checks"])


def test_development_profile_allows_legacy_dataset_without_structure_manifest():
    cases = [
        {
            "id": f"d-{index}",
            "category": "exact" if index % 2 else "paraphrase",
            "query": f"q {index}",
            "relevance": {"any_terms": [f"term {index}"]},
        }
        for index in range(10)
    ]
    audit = audit_pdf_structure_suite({"name": "dev", "cases": cases}, _documents(), profile="development")
    assert audit["passed"] is True
    assert audit["warnings"]


def test_suite_skeleton_pins_source_files_without_inventing_questions():
    skeleton = build_suite_skeleton(_documents())
    assert skeleton["cases"] == []
    assert skeleton["pdf_structure_suite"]["documents"][0]["sha256"] == hashlib.sha256(b"book").hexdigest()
    assert skeleton["pdf_structure_suite"]["documents"][0]["document_type"] == "TODO"


def test_promotion_gate_passes_candidate_with_equal_quality_and_cleaner_boundaries():
    control = _pipeline("control", non_boundary=0.8)
    candidate = _pipeline("candidate", non_boundary=0.2)
    candidate["chunks"] = 103
    candidate["timing"]["embedding_seconds"] = 9.5
    candidate["memory"]["tracemalloc_peak_bytes"] = 1050
    gate = evaluate_promotion_gate(control, candidate, {"passed": True})
    assert gate["passed"] is True
    assert not gate["diagnostics"]["new_case_failures"]


def test_promotion_gate_rejects_new_case_failure_and_category_regression():
    control = _pipeline("control")
    candidate = _pipeline("candidate", mrr=0.90, non_boundary=0.2)
    candidate["cases"][1]["retrieval_pass"] = False
    candidate["cases"][1]["first_relevant_rank"] = None
    gate = evaluate_promotion_gate(
        control,
        candidate,
        {"passed": True},
        thresholds=PromotionThresholds(mrr_delta_min=-0.2, category_mrr_delta_min=-0.05),
    )
    assert gate["passed"] is False
    assert gate["diagnostics"]["new_case_failures"] == ["b"]
    assert gate["diagnostics"]["category_regressions"]
