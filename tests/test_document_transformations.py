from __future__ import annotations

import pytest

from services.api.app.routes.sources import _validate_source_config
from services.api.app.storage.models import Chunk
from services.worker.chunking import chunking_metadata, resolve_chunking_spec, split_text
from services.worker.connectors.incremental import previous_by_external_id, reusable_chunks
from services.worker.parsing import (
    DocumentBlock,
    NormalizedDocument,
    iter_document_segments,
    parse_document,
    resolve_parser_spec,
)
from services.worker.pipeline import _reuse_key


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


def test_parser_identity_is_stable_and_changes_with_provider_or_options() -> None:
    legacy = resolve_parser_spec(None, name="manual.pdf", media_type="application/pdf")
    same = resolve_parser_spec(None, name="manual.pdf", media_type="application/pdf")
    pymupdf = resolve_parser_spec(
        {"provider": "pymupdf", "strategy": "blocks"},
        name="manual.pdf",
        media_type="application/pdf",
    )
    sorted_false = resolve_parser_spec(
        {"provider": "pymupdf", "strategy": "blocks", "options": {"sort": False}},
        name="manual.pdf",
        media_type="application/pdf",
    )

    assert legacy.provider == "ragbot"
    assert legacy.strategy == "pypdf2"
    assert legacy.config_hash == same.config_hash
    assert legacy.config_hash != pymupdf.config_hash
    assert pymupdf.config_hash != sorted_false.config_hash


def test_office_resources_resolve_to_docling_without_importing_optional_dependency() -> None:
    spec = resolve_parser_spec(
        None,
        name="architecture.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert spec.provider == "docling"
    assert spec.strategy == "document"


def test_ragbot_html_parser_emits_structured_blocks() -> None:
    document, metadata = parse_document(
        b"<html><body><nav>drop me</nav><h1>Servo</h1><p>Current loop.</p><pre>FOC()</pre></body></html>",
        None,
        name="runbook.html",
        media_type="text/html",
    )

    assert metadata["parser_provider"] == "ragbot"
    assert metadata["parser_strategy"] == "html"
    assert [block.kind for block in document.blocks] == ["heading", "paragraph", "code"]
    assert document.blocks[1].section == "Servo"
    assert "drop me" not in document.text


def test_block_to_chunk_bridge_preserves_page_bbox_and_kind() -> None:
    document = NormalizedDocument(
        name="manual.pdf",
        media_type="application/pdf",
        blocks=[
            DocumentBlock(
                block_index=7,
                text="abcdefghijklmnopqrstuvwxyz",
                kind="text_block",
                page=3,
                section="Commissioning",
                bbox=(10.0, 20.0, 30.0, 40.0),
            )
        ],
    )
    segments = list(
        iter_document_segments(
            document,
            None,
            chunk_size=10,
            chunk_overlap=2,
        )
    )

    assert [segment.text for segment in segments] == ["abcdefghij", "ijklmnopqr", "qrstuvwxyz"]
    assert all(segment.page == 3 for segment in segments)
    assert all(segment.section == "Commissioning" for segment in segments)
    assert all(segment.metadata["block_index"] == 7 for segment in segments)
    assert all(segment.metadata["block_kind"] == "text_block" for segment in segments)
    assert all(segment.metadata["bbox"] == [10.0, 20.0, 30.0, 40.0] for segment in segments)


def test_pdf_ingestion_preserves_parser_block_page_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from services.worker.jobs import ingest_pdf as module

    fake_document = NormalizedDocument(
        name="fixture.pdf",
        media_type="application/pdf",
        blocks=[
            DocumentBlock(block_index=0, text="alpha beta gamma", kind="page", page=1),
            DocumentBlock(block_index=1, text="delta epsilon zeta", kind="page", page=3),
        ],
    )
    monkeypatch.setattr(module, "fetch_pdf_bytes", lambda _path: b"pdf-bytes")
    monkeypatch.setattr(
        module,
        "parse_document",
        lambda *_args, **_kwargs: (
            fake_document,
            {
                "parser_provider": "ragbot",
                "parser_strategy": "pypdf2",
                "parser_version": 1,
                "parser_config_hash": "parser-hash",
            },
        ),
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
    assert all(chunk.metadata["parser_provider"] == "ragbot" for chunk in chunks)
    assert all(chunk.metadata["parser_strategy"] == "pypdf2" for chunk in chunks)
    assert all(chunk.metadata["parser_config_hash"] == "parser-hash" for chunk in chunks)
    assert all(chunk.metadata["chunker_provider"] == "ragbot" for chunk in chunks)


def test_reuse_identity_changes_when_parser_contract_changes() -> None:
    base = dict(
        chunk_id="chunk",
        doc_id="doc",
        tenant_id="tenant",
        chunk_index=0,
        text="same",
        checksum="checksum",
        metadata={
            "source_type": "pdf",
            "version": "1.0",
            "parser_provider": "ragbot",
            "parser_strategy": "pypdf2",
            "parser_version": 1,
            "parser_config_hash": "old",
            **chunking_metadata(None, chunk_size=800, chunk_overlap=100),
        },
    )
    old = Chunk(**base)
    changed = Chunk(**{**base, "metadata": {**base["metadata"], "parser_config_hash": "new"}})
    assert _reuse_key(old) != _reuse_key(changed)


def test_source_validation_rejects_invalid_or_unsupported_parser_config() -> None:
    with pytest.raises(Exception) as invalid:
        _validate_source_config(
            "pdf",
            {"path": "/data/a.pdf", "parsing": {"provider": "unknown"}},
        )
    assert getattr(invalid.value, "status_code", None) == 422

    with pytest.raises(Exception) as unsupported:
        _validate_source_config(
            "notion",
            {
                "page_id": "p",
                "credential_ref": "env:NOTION_TOKEN",
                "parsing": {"provider": "docling"},
            },
        )
    assert getattr(unsupported.value, "status_code", None) == 422


def test_pymupdf_block_adapter_when_extra_is_installed() -> None:
    pymupdf = pytest.importorskip("pymupdf")
    source = pymupdf.open()
    page = source.new_page()
    page.insert_text((72, 72), "Servo parser benchmark block")
    data = source.tobytes()
    source.close()

    document, metadata = parse_document(
        data,
        {"provider": "pymupdf", "strategy": "blocks"},
        name="fixture.pdf",
        media_type="application/pdf",
    )

    assert metadata["parser_provider"] == "pymupdf"
    assert document.blocks
    assert document.blocks[0].page == 1
    assert document.blocks[0].bbox is not None
    assert "Servo parser benchmark block" in document.text


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
