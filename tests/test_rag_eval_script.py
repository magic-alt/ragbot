from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rag_eval.py"
SPEC = importlib.util.spec_from_file_location("rag_eval_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
rag_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rag_eval)


def _chunk(text: str, *, chunk_id: str = "c1", page: int = 1, rank: int = 1):
    return {
        "chunk_id": chunk_id,
        "doc_id": "doc-1",
        "text": text,
        "score": 0.01,
        "metadata": {
            "page": page,
            "path": "/data/book.pdf",
            "_retrieval": {
                "final_rank": rank,
                "vector": {"rank": rank, "score": 0.8},
                "lexical": {"rank": rank, "score": 0.5},
                "rrf_score": 0.01,
            },
        },
    }


def test_relevance_supports_terms_pages_paths_and_exact_chunks():
    terms_case = {"relevance": {"all_terms": ["LoRA", "QLoRA"]}}
    assert rag_eval._chunk_relevant(_chunk("LoRA and QLoRA are PEFT methods"), terms_case)
    assert not rag_eval._chunk_relevant(_chunk("LoRA only"), terms_case)

    page_case = {"relevance": {"pages": [7], "path_contains": ["book.pdf"]}}
    assert rag_eval._chunk_relevant(_chunk("anything", page=7), page_case)
    assert not rag_eval._chunk_relevant(_chunk("anything", page=8), page_case)

    exact_case = {"relevance": {"expected_chunk_ids": ["gold"]}}
    assert rag_eval._chunk_relevant(_chunk("anything", chunk_id="gold"), exact_case)
    assert not rag_eval._chunk_relevant(_chunk("anything", chunk_id="other"), exact_case)


def test_summary_computes_hit_mrr_recall_and_latency():
    diagnostics = {
        "embedding_model": "text-embedding-3-small",
        "semantic_embedding": True,
    }
    results = [
        rag_eval.CaseResult(
            case_id="one",
            category="english",
            query="q1",
            labeled=True,
            retrieval_pass=True,
            first_relevant_rank=1,
            reciprocal_rank_at_10=1.0,
            recall_at_10=1.0,
            search_ms=100.0,
            answer_ms=None,
            answer_pass=None,
            answer=None,
            answer_confidence=None,
            citation_count=None,
            diagnostics=diagnostics,
            chunks=[],
            warnings=[],
        ),
        rag_eval.CaseResult(
            case_id="two",
            category="english",
            query="q2",
            labeled=True,
            retrieval_pass=True,
            first_relevant_rank=5,
            reciprocal_rank_at_10=0.2,
            recall_at_10=0.5,
            search_ms=300.0,
            answer_ms=None,
            answer_pass=None,
            answer=None,
            answer_confidence=None,
            citation_count=None,
            diagnostics=diagnostics,
            chunks=[],
            warnings=[],
        ),
    ]

    summary = rag_eval._summary(results)
    assert summary["hit_at_1"] == 0.5
    assert summary["hit_at_5"] == 1.0
    assert summary["mrr_at_10"] == 0.6
    assert summary["recall_at_10"] == 0.75
    assert summary["p50_search_ms"] == 200.0
    assert summary["pass_rate"] == 1.0


def test_threshold_gate_supports_quality_latency_and_semantic_requirements():
    summary = {
        "pass_rate": 0.9,
        "hit_at_1": 0.7,
        "hit_at_3": 0.8,
        "hit_at_5": 0.9,
        "hit_at_10": 1.0,
        "mrr_at_10": 0.75,
        "recall_at_10": 0.8,
        "p95_search_ms": 450.0,
        "p95_answer_ms": 1200.0,
        "runtime": {"semantic_embedding": True},
    }
    gate = rag_eval._evaluate_thresholds(
        summary,
        {
            "hit_at_5_min": 0.8,
            "mrr_at_10_min": 0.7,
            "p95_search_ms_max": 500,
            "semantic_embedding_required": True,
        },
    )
    assert gate["passed"] is True

    summary["runtime"] = {"semantic_embedding": False}
    failed = rag_eval._evaluate_thresholds(summary, {"semantic_embedding_required": True})
    assert failed["passed"] is False


def test_run_case_uses_live_search_trace_and_optional_chat(monkeypatch):
    calls = []

    def fake_http(server, path, payload, *, api_key, timeout):
        calls.append((path, payload))
        if path == "/search":
            return {
                "diagnostics": {
                    "embedding_model": "text-embedding-3-small",
                    "embedding_backend": "APIEmbedder",
                    "embedding_dimension": 1536,
                    "semantic_embedding": True,
                    "warnings": [],
                },
                "chunks": [
                    _chunk("QLoRA combines LoRA with quantization", chunk_id="gold", page=12)
                ],
            }
        if path == "/chat":
            return {
                "answer": "QLoRA combines LoRA with quantization.",
                "citations": [{"chunk_id": "gold"}],
                "confidence": "high",
            }
        raise AssertionError(path)

    monkeypatch.setattr(rag_eval, "_http_json", fake_http)
    dataset = {"defaults": {"top_k": 10, "filters": {"source_types": ["pdf"]}}}
    case = {
        "id": "lora",
        "query": "LoRA vs QLoRA",
        "category": "english",
        "relevance": {"expected_chunk_ids": ["gold"], "max_rank": 5},
        "answer": {"contains_all": ["LoRA", "quantization"], "min_citations": 1},
    }
    result = rag_eval.run_case(
        dataset,
        case,
        server="http://localhost:8000",
        tenant="engineering",
        user="test",
        api_key=None,
        timeout=5,
        cli_top_k=None,
        with_answers=True,
    )
    assert result.retrieval_pass is True
    assert result.first_relevant_rank == 1
    assert result.recall_at_10 == 1.0
    assert result.answer_pass is True
    assert [path for path, _ in calls] == ["/search", "/chat"]


def test_reports_render_json_friendly_markdown_and_html(tmp_path):
    report = {
        "name": "demo",
        "generated_at": "2026-09-03T00:00:00+00:00",
        "server": "http://127.0.0.1:8000",
        "tenant": "engineering",
        "summary": {
            "total_cases": 1,
            "pass_rate": 1.0,
            "hit_at_1": 1.0,
            "hit_at_3": 1.0,
            "hit_at_5": 1.0,
            "hit_at_10": 1.0,
            "mrr_at_10": 1.0,
            "recall_at_10": 1.0,
            "p50_search_ms": 10.0,
            "p95_search_ms": 10.0,
            "runtime": {"embedding_model": "semantic", "semantic_embedding": True},
            "warnings": [],
        },
        "gate": {"passed": True, "checks": {}},
        "cases": [
            {
                "case_id": "case-1",
                "category": "test",
                "query": "hello",
                "case_pass": True,
                "first_relevant_rank": 1,
                "reciprocal_rank_at_10": 1.0,
                "recall_at_10": 1.0,
                "search_ms": 10.0,
                "answer": None,
                "answer_confidence": None,
                "citation_count": None,
                "error": None,
                "chunks": [],
            }
        ],
    }
    markdown = rag_eval._markdown(report)
    rendered = rag_eval._html_report(report)
    assert "Hit@5" in markdown
    assert "Ragbot RAG Evaluation" in rendered
    paths = rag_eval._write_reports(report, tmp_path)
    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert paths["html"].exists()
    assert paths["latest"].exists()
