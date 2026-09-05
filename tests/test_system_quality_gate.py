from __future__ import annotations

from eval.system_quality import (
    CASES,
    ProbeResult,
    _mode_metrics,
    build_fixture_pdf,
    evaluate_gate,
)


def test_fixture_pdf_is_self_contained_and_has_all_quality_markers():
    payload = build_fixture_pdf()
    assert payload.startswith(b"%PDF-")
    assert payload.rstrip().endswith(b"%%EOF")
    for case in CASES:
        assert case.sentinel.encode("ascii") in payload


def test_mode_metrics_measure_hit_mrr_stability_and_errors():
    results = []
    for index, case in enumerate(CASES, 1):
        results.append(
            ProbeResult(
                case_id=case.case_id,
                mode="hybrid",
                rank=index,
                reciprocal_rank=1.0 / index,
                latency_ms=10.0 * index,
                diagnostics={"semantic_embedding": True},
                rerank_requested=True,
            )
        )
    metrics = _mode_metrics(results, "hybrid")
    assert metrics["hit_at_5"] == 1.0
    assert metrics["semantic_hit_at_5"] == 1.0
    assert metrics["mrr_at_10"] > 0.45
    assert metrics["errors"] == 0
    assert metrics["p95_ms"] > metrics["p50_ms"]


def test_standard_gate_rejects_hash_embedding_and_accepts_semantic_runtime():
    summary = {
        "runtime": {"semantic_embedding": True},
        "retrieval": {
            "lexical": {"p95_ms": 200.0, "errors": 0},
            "vector": {
                "semantic_hit_at_5": 1.0,
                "p95_ms": 300.0,
                "errors": 0,
            },
            "hybrid": {
                "hit_at_5": 1.0,
                "mrr_at_10": 0.9,
                "p95_ms": 350.0,
                "errors": 0,
            },
        },
        "answer": {"pass_rate": 1.0},
        "ingestion": {"doc_count": 1, "chunk_count": 5},
    }
    assert evaluate_gate(summary, "standard")["passed"] is True

    summary["runtime"] = {"semantic_embedding": False}
    gate = evaluate_gate(summary, "standard")
    assert gate["passed"] is False
    failed = {
        item["name"] for item in gate["checks"] if not item["passed"]
    }
    assert "semantic_embedding" in failed
