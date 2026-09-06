from __future__ import annotations

from benchmarks.component_compare import (
    _label_text_coverage,
    parser_config,
    run_splitter_backend,
)
from benchmarks.rag_native_compare import CorpusUnit
from services.api.app.retrieval.embedder import HashEmbedder


def _cases():
    return [
        {
            "id": "fp8",
            "category": "exact",
            "query": "mixed precision FP8 memory",
            "relevance": {"any_terms": ["mixed precision", "fp8"], "max_rank": 5},
        },
        {
            "id": "cache",
            "category": "paraphrase",
            "query": "cache repeated computation",
            "relevance": {"any_terms": ["cache hit", "cache miss"], "max_rank": 5},
        },
    ]


def test_splitter_benchmark_reuses_concept_label_scorer():
    units = [
        CorpusUnit(
            doc_id="book.pdf",
            path="book.pdf",
            page=1,
            text="Mixed precision FP8 lowers memory pressure. Cache hit and cache miss behavior avoids repeated work.",
        )
    ]
    result = run_splitter_backend(
        "ragbot",
        units,
        _cases(),
        embedder=HashEmbedder(256),
        chunk_size=200,
        chunk_overlap=20,
        top_k=5,
        repetitions=2,
    )
    assert result["component"] == "splitter"
    assert result["backend"] == "ragbot"
    assert result["chunks"] >= 1
    assert result["quality"]["hit_at_5"] == 1.0
    assert result["quality"]["recall_at_10"] == 1.0
    assert result["chunk_shape"]["character_inflation_ratio"] >= 0.9


def test_label_text_coverage_checks_answer_bearing_terms_before_retrieval():
    assert _label_text_coverage(_cases(), "FP8 mixed precision plus cache hit and cache miss") == 1.0
    assert _label_text_coverage(_cases(), "FP8 mixed precision only") == 0.5


def test_parser_configs_are_explicit_and_framework_neutral():
    assert parser_config("pypdf2") == {"provider": "ragbot", "strategy": "pypdf2"}
    assert parser_config("pymupdf") == {"provider": "pymupdf", "strategy": "blocks"}
    assert parser_config("docling") == {"provider": "docling", "strategy": "document"}
    assert parser_config("unstructured") == {"provider": "unstructured", "strategy": "elements"}
