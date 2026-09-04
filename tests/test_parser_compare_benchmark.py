from __future__ import annotations

from argparse import Namespace

import pytest

from benchmarks import parser_compare
from services.api.app.retrieval.embedder import HashEmbedder


def test_synthetic_parser_dataset_and_controlled_backends() -> None:
    pytest.importorskip("pymupdf")
    documents, cases = parser_compare.synthetic_pdf_dataset(4, 3)

    assert len(documents) == 4
    assert len(cases) == 3
    assert all(document.path.endswith(".pdf") for document in documents)

    embedder = HashEmbedder(64)
    legacy = parser_compare.run_backend(
        "pypdf2",
        documents,
        cases,
        chunk_size=400,
        chunk_overlap=40,
        top_k=10,
        embedder=embedder,
    )
    pymupdf = parser_compare.run_backend(
        "pymupdf",
        documents,
        cases,
        chunk_size=400,
        chunk_overlap=40,
        top_k=10,
        embedder=embedder,
    )

    assert legacy["parser"]["provider"] == "ragbot"
    assert pymupdf["parser"]["provider"] == "pymupdf"
    assert legacy["documents"] == pymupdf["documents"] == 4
    assert legacy["queries"] == pymupdf["queries"] == 3
    assert pymupdf["structure"]["bbox_block_rate"] > 0


def test_parser_benchmark_run_writes_common_methodology(tmp_path) -> None:
    pytest.importorskip("pymupdf")
    output = tmp_path / "parser.json"
    args = Namespace(
        backends="pypdf2,pymupdf",
        chunk_size=400,
        chunk_overlap=40,
        top_k=10,
        embedding="hash",
        hash_dimension=64,
        corpus_dir="",
        golden="",
        synthetic_documents=3,
        synthetic_queries=2,
        output=str(output),
    )

    payload = parser_compare.run(args)

    assert output.exists()
    assert payload["methodology"]["changed_variable"] == "parser implementation"
    assert payload["configuration"]["backends"] == ["pypdf2", "pymupdf"]
    assert len(payload["results"]) == 2
