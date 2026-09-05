from __future__ import annotations

import importlib.util

import pytest

from benchmarks.rag_native_compare import (
    CorpusUnit,
    RetrievedHit,
    audit_golden_dataset,
    corpus_manifest,
    run_comparison,
    score_case,
    summarize_scores,
    synthetic_native_dataset,
)


def test_dataset_audit_distinguishes_development_and_production_maturity():
    cases = [
        {
            "id": f"case-{index}",
            "category": "exact" if index % 2 else "paraphrase",
            "query": f"query {index}",
            "relevance": {"doc_ids": [f"doc-{index}.md"]},
        }
        for index in range(12)
    ]
    dataset = {"cases": cases}
    development = audit_golden_dataset(dataset, "development")
    production = audit_golden_dataset(dataset, "production")
    assert development["passed"] is True
    assert production["passed"] is False
    failed = {item["name"] for item in production["checks"] if not item["passed"]}
    assert "case_count" in failed
    assert "category_count" in failed


def test_production_dataset_requires_stable_cross_framework_labels():
    cases = []
    for index in range(50):
        relevance = (
            {"doc_ids": [f"doc-{index}.md"]}
            if index < 39
            else {"any_terms": [f"term-{index}"]}
        )
        cases.append(
            {
                "id": f"case-{index}",
                "category": ["exact", "paraphrase", "cross-lingual"][index % 3],
                "query": f"query {index}",
                "relevance": relevance,
            }
        )
    audit = audit_golden_dataset({"cases": cases}, "production")
    assert audit["passed"] is False
    stable = next(item for item in audit["checks"] if item["name"] == "stable_label_rate")
    assert stable["actual"] == 0.78


def test_case_scoring_deduplicates_multiple_chunks_from_same_relevant_document():
    case = {
        "id": "servo",
        "category": "exact",
        "query": "servo loop",
        "relevance": {"doc_ids": ["servo.md", "ethercat.md"], "max_rank": 5},
    }
    hits = [
        RetrievedHit("s1", "servo.md", "servo.md", "servo loop one"),
        RetrievedHit("s2", "servo.md", "servo.md", "servo loop two"),
        RetrievedHit("x", "noise.md", "noise.md", "noise"),
        RetrievedHit("e1", "ethercat.md", "ethercat.md", "ethercat servo"),
    ]
    score = score_case(case, hits)
    assert score.first_relevant_rank == 1
    assert score.reciprocal_rank_at_10 == 1.0
    assert score.precision_at_5 == pytest.approx(2 / 5)
    assert score.recall_at_10 == 1.0
    assert score.ndcg_at_10 < 1.0


def test_summary_reports_quality_latency_and_categories():
    case = {
        "id": "one",
        "category": "cross-lingual",
        "query": "问题",
        "relevance": {"doc_ids": ["answer.md"]},
    }
    score = score_case(case, [RetrievedHit("c", "answer.md", "answer.md", "answer")])
    summary = summarize_scores([score], [10.0, 20.0, 30.0])
    assert summary["hit_at_1"] == 1.0
    assert summary["mrr_at_10"] == 1.0
    assert summary["recall_at_10"] == 1.0
    assert summary["query_latency_ms_p95"] == 29.0
    assert summary["categories"]["cross-lingual"]["hit_at_5"] == 1.0


def test_corpus_manifest_changes_when_content_changes():
    first = corpus_manifest([CorpusUnit("a.md", "a.md", "alpha")])
    second = corpus_manifest([CorpusUnit("a.md", "a.md", "beta")])
    assert first["documents"] == 1
    assert first["sha256"] != second["sha256"]


@pytest.mark.skipif(
    importlib.util.find_spec("langchain_core") is None
    or importlib.util.find_spec("langchain_text_splitters") is None
    or importlib.util.find_spec("llama_index") is None,
    reason="native framework benchmark extras are not installed",
)
def test_langchain_and_llamaindex_native_smoke_with_same_hash_embedder():
    units, dataset = synthetic_native_dataset(documents=8, queries=4)
    report = run_comparison(
        dataset=dataset,
        units=units,
        backends="langchain,llamaindex",
        embedding="hash",
        hash_dimension=64,
        chunk_size=160,
        chunk_overlap=20,
        top_k=10,
        repetitions=1,
    )
    assert [item["backend"] for item in report["results"]] == ["langchain", "llamaindex"]
    assert report["corpus_manifest"]["documents"] == 8
    for result in report["results"]:
        assert result["build"]["chunks"] > 0
        assert result["summary"]["hit_at_5"] >= 0.5
        assert result["summary"]["query_latency_ms_p95"] >= 0.0
