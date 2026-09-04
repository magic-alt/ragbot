from __future__ import annotations

import json
from pathlib import Path

from benchmarks.rag_framework_compare import (
    RagbotFixedWindowChunker,
    load_golden,
    run_backend,
    synthetic_dataset,
)
from services.api.app.retrieval.embedder import HashEmbedder
from services.worker.jobs.ingest_pdf import _split_text


def test_ragbot_benchmark_chunker_matches_production_splitter() -> None:
    text = (
        "First paragraph has several words and punctuation.\n\n"
        "第二段用于验证 Unicode 字符切分。\n\n"
        "Third paragraph makes the input long enough to overlap."
    ) * 8
    chunk_size = 93
    overlap = 17

    benchmark = RagbotFixedWindowChunker(chunk_size, overlap).split(text)
    production = _split_text(text, chunk_size, overlap)

    assert benchmark == production


def test_controlled_ragbot_backend_smoke_produces_quality_and_timing_metrics() -> None:
    documents, cases = synthetic_dataset(documents=12, queries=8)

    result = run_backend(
        "ragbot",
        documents,
        cases,
        chunk_size=240,
        chunk_overlap=40,
        top_k=10,
        embedder=HashEmbedder(256),
    )

    assert result["backend"] == "ragbot"
    assert result["documents"] == 12
    assert result["queries"] == len(cases)
    assert result["chunks"] > 0
    assert 0.0 <= result["quality"]["hit_at_5"] <= 1.0
    assert 0.0 <= result["quality"]["mrr_at_10"] <= 1.0
    assert result["timing"]["split_seconds"] >= 0.0
    assert result["memory"]["tracemalloc_peak_bytes"] > 0


def test_load_golden_accepts_existing_dataset_shape(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            {
                "name": "framework-test",
                "cases": [
                    {
                        "id": "doc-id-case",
                        "query": "Where is the retry budget documented?",
                        "relevance": {"doc_ids": ["ops.md"]},
                    },
                    {
                        "id": "path-case",
                        "query": "Where is the vector index described?",
                        "relevance": {"path_contains": "architecture/"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    cases = load_golden(path)

    assert cases[0].expected_doc_ids == ("ops.md",)
    assert cases[1].path_contains == ("architecture/",)
