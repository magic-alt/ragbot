from __future__ import annotations

import pytest

from services.api.app.storage.models import Chunk
from services.worker.chunking import chunking_metadata, resolve_chunking_spec, split_text
from services.worker.connectors.incremental import previous_by_external_id, reusable_chunks


def test_fixed_chunker_preserves_legacy_character_windows() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    segments, metadata = split_text(text, None, chunk_size=10, chunk_overlap=2)

    assert segments == ["abcdefghij", "ijklmnopqr", "qrstuvwxyz"]
    assert metadata["chunker_provider"] == "ragbot"
    assert metadata["chunker_strategy"] == "fixed"
    assert metadata["chunk_size"] == 10
    assert metadata["chunk_overlap"] == 2
    assert metadata["chunker_config_hash"]


def test_chunker_identity_changes_when_strategy_or_budget_changes() -> None:
    fixed = chunking_metadata(None, chunk_size=800, chunk_overlap=100)
    recursive = chunking_metadata(
        {"provider": "langchain", "strategy": "recursive"},
        chunk_size=800,
        chunk_overlap=100,
    )
    resized = chunking_metadata(None, chunk_size=1200, chunk_overlap=100)

    assert fixed["chunker_config_hash"] != recursive["chunker_config_hash"]
    assert fixed["chunker_config_hash"] != resized["chunker_config_hash"]


def test_repo_legacy_strategy_has_explicit_structural_identity() -> None:
    spec = resolve_chunking_spec(
        None,
        chunk_size=50,
        chunk_overlap=100,
        language="python",
        default_strategy="structural",
    )
    assert spec.provider == "ragbot"
    assert spec.strategy == "structural"


def test_metadata_first_reuse_requires_same_chunker_contract() -> None:
    required = chunking_metadata(None, chunk_size=800, chunk_overlap=100)
    previous = Chunk(
        chunk_id="old",
        doc_id="doc",
        tenant_id="tenant",
        chunk_index=0,
        text="unchanged",
        metadata={"external_id": "remote-1", "remote_version": "v1", **required},
    )
    grouped = previous_by_external_id([previous])

    assert reusable_chunks(
        grouped,
        external_id="remote-1",
        remote_version="v1",
        required_metadata=required,
    ) is not None

    changed = chunking_metadata(
        {"provider": "langchain", "strategy": "recursive"},
        chunk_size=800,
        chunk_overlap=100,
    )
    assert reusable_chunks(
        grouped,
        external_id="remote-1",
        remote_version="v1",
        required_metadata=changed,
    ) is None


def test_pdf_ingestion_preserves_page_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from services.worker.jobs import ingest_pdf as module

    monkeypatch.setattr(
        module,
        "fetch_pdf_pages",
        lambda _path: [(1, "alpha beta gamma"), (3, "delta epsilon zeta")],
    )
    chunks = list(
        module.ingest_pdf(
            "fixture.pdf",
            doc_id="doc",
            tenant_id="tenant",
            chunk_size=100,
            chunk_overlap=0,
        )
    )

    assert [chunk.page for chunk in chunks] == [1, 3]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert all(chunk.metadata["parser_provider"] == "pypdf2" for chunk in chunks)
    assert all(chunk.metadata["chunker_provider"] == "ragbot" for chunk in chunks)


def test_langchain_recursive_adapter_when_extra_is_installed() -> None:
    pytest.importorskip("langchain_text_splitters")
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    segments, metadata = split_text(
        text,
        {"provider": "langchain", "strategy": "recursive"},
        chunk_size=30,
        chunk_overlap=5,
    )
    assert segments
    assert metadata["chunker_provider"] == "langchain"


def test_llamaindex_sentence_adapter_when_extra_is_installed() -> None:
    pytest.importorskip("llama_index.core")
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    segments, metadata = split_text(
        text,
        {"provider": "llamaindex", "strategy": "sentence"},
        chunk_size=35,
        chunk_overlap=5,
    )
    assert segments
    assert metadata["chunker_provider"] == "llamaindex"
