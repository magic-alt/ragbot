from __future__ import annotations

from benchmarks.factorial_compare import compact_factorial_cells, splitter_config
from services.worker.chunking import chunking_metadata, resolve_chunking_spec
from services.worker.parsing import (
    DocumentBlock,
    NormalizedDocument,
    coalesce_document_blocks,
    iter_document_segments,
)


def _document() -> NormalizedDocument:
    return NormalizedDocument(
        name="manual.pdf",
        media_type="application/pdf",
        blocks=[
            DocumentBlock(0, "Heading one", kind="text_block", page=1, bbox=(10, 10, 100, 30)),
            DocumentBlock(1, "First paragraph with useful context.", kind="text_block", page=1, bbox=(10, 40, 200, 80)),
            DocumentBlock(2, "Second paragraph continues the same idea.", kind="text_block", page=1, bbox=(10, 90, 200, 130)),
            DocumentBlock(3, "Page two starts here.", kind="text_block", page=2, bbox=(10, 10, 180, 40)),
        ],
    )


def test_coalescer_merges_adjacent_same_page_blocks_and_preserves_provenance():
    blocks = coalesce_document_blocks(_document(), target_chars=400)
    assert len(blocks) == 2
    first = blocks[0]
    assert first.page == 1
    assert first.kind == "coalesced_text"
    assert first.metadata["source_block_indices"] == [0, 1, 2]
    assert first.metadata["source_block_count"] == 3
    assert first.bbox == (10.0, 10.0, 200.0, 130.0)
    assert "First paragraph" in first.text
    assert blocks[1].page == 2


def test_coalescer_never_crosses_page_boundary_by_default():
    blocks = coalesce_document_blocks(_document(), target_chars=10_000)
    assert len(blocks) == 2
    assert [block.page for block in blocks] == [1, 2]


def test_bridge_coalescing_is_opt_in_and_reduces_parser_fragmentation():
    document = _document()
    raw = list(
        iter_document_segments(
            document,
            {"provider": "ragbot", "strategy": "fixed"},
            chunk_size=200,
            chunk_overlap=0,
        )
    )
    coalesced = list(
        iter_document_segments(
            document,
            {
                "provider": "ragbot",
                "strategy": "fixed",
                "block_coalescing": {"enabled": True, "target_chars": 800},
            },
            chunk_size=200,
            chunk_overlap=0,
        )
    )
    assert len(raw) == 4
    assert len(coalesced) == 2
    assert raw[0].metadata["block_coalescing"]["enabled"] is False
    assert coalesced[0].metadata["block_coalescing"]["enabled"] is True
    assert coalesced[0].metadata["block_metadata"]["source_block_count"] == 3


def test_coalescing_changes_durable_chunk_contract_but_disabled_path_keeps_legacy_hash():
    legacy = chunking_metadata(None, chunk_size=800, chunk_overlap=100)
    explicit_disabled = chunking_metadata(
        {"block_coalescing": {"enabled": False}},
        chunk_size=800,
        chunk_overlap=100,
    )
    enabled = chunking_metadata(
        {"block_coalescing": {"enabled": True, "target_chars": 3200}},
        chunk_size=800,
        chunk_overlap=100,
    )
    enabled_other_target = chunking_metadata(
        {"block_coalescing": {"enabled": True, "target_chars": 2400}},
        chunk_size=800,
        chunk_overlap=100,
    )

    assert explicit_disabled["chunker_config_hash"] == legacy["chunker_config_hash"]
    assert enabled["chunker_config_hash"] != legacy["chunker_config_hash"]
    assert enabled_other_target["chunker_config_hash"] != enabled["chunker_config_hash"]
    assert enabled["block_coalescing_enabled"] is True
    assert enabled["block_coalescing_target_chars"] == 3200

    spec = resolve_chunking_spec(
        {"block_coalescing": True},
        chunk_size=800,
        chunk_overlap=100,
    )
    assert spec.block_coalescing_enabled is True
    assert spec.block_coalescing_target_chars == 3200


def test_table_blocks_remain_structural_boundaries():
    document = NormalizedDocument(
        name="table.pdf",
        media_type="application/pdf",
        blocks=[
            DocumentBlock(0, "before", kind="text_block", page=1),
            DocumentBlock(1, "A | B\n1 | 2", kind="table", page=1),
            DocumentBlock(2, "after", kind="text_block", page=1),
        ],
    )
    blocks = coalesce_document_blocks(document, target_chars=1000)
    assert len(blocks) == 3
    assert blocks[1].kind == "table"
    assert blocks[1].metadata["source_block_count"] == 1


def test_compact_factorial_design_is_three_pipeline_levels_by_two_splitters():
    cells = compact_factorial_cells(
        ["pypdf2", "pymupdf"],
        ["ragbot", "llamaindex"],
    )
    assert cells == [
        ("pypdf2", "ragbot", "raw"),
        ("pypdf2", "llamaindex", "raw"),
        ("pymupdf", "ragbot", "raw"),
        ("pymupdf", "llamaindex", "raw"),
        ("pymupdf", "ragbot", "coalesced"),
        ("pymupdf", "llamaindex", "coalesced"),
    ]


def test_factorial_splitter_config_uses_production_bridge_contract():
    config = splitter_config(
        "llamaindex",
        coalesced=True,
        chunk_size=800,
        coalesce_target_multiplier=4.0,
    )
    assert config["provider"] == "llamaindex"
    assert config["strategy"] == "sentence"
    assert config["block_coalescing"] == {
        "enabled": True,
        "target_chars": 3200,
        "respect_page": True,
        "respect_section": True,
    }
