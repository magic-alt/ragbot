from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from services.worker.chunking import resolve_chunking_spec, split_text

from .coalescing import coalesce_document_blocks
from .models import DocumentBlock, NormalizedDocument


@dataclass(frozen=True)
class ParsedSegment:
    """Chunk-ready segment while retaining parser provenance."""

    text: str
    block_index: int
    block_kind: str
    page: int | None = None
    section: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _bridge_blocks(
    document: NormalizedDocument,
    chunking: Mapping[str, Any] | None,
    *,
    chunk_size: int,
    chunk_overlap: int,
    language: str | None,
) -> tuple[list[DocumentBlock], dict[str, Any]]:
    spec = resolve_chunking_spec(
        chunking,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        language=language,
    )
    if not spec.block_coalescing_enabled:
        return list(document.blocks), {
            "enabled": False,
            "input_blocks": len(document.blocks),
            "output_blocks": len(document.blocks),
        }

    blocks = coalesce_document_blocks(
        document,
        target_chars=int(spec.block_coalescing_target_chars or chunk_size * 4),
        respect_page=spec.block_coalescing_respect_page,
        respect_section=spec.block_coalescing_respect_section,
    )
    return blocks, {
        "enabled": True,
        "target_chars": int(spec.block_coalescing_target_chars or chunk_size * 4),
        "respect_page": spec.block_coalescing_respect_page,
        "respect_section": spec.block_coalescing_respect_section,
        "input_blocks": len(document.blocks),
        "output_blocks": len(blocks),
    }


def iter_document_segments(
    document: NormalizedDocument,
    chunking: Mapping[str, Any] | None,
    *,
    chunk_size: int,
    chunk_overlap: int,
    language: str | None = None,
) -> Iterable[ParsedSegment]:
    """Apply the configured Chunker to parser blocks or coalesced block windows.

    ``block_coalescing`` is intentionally opt-in so existing Source contracts
    keep byte-for-byte chunking behavior until a benchmarked configuration is
    explicitly promoted. Example configuration::

        {
            "provider": "llamaindex",
            "strategy": "sentence",
            "block_coalescing": {
                "enabled": True,
                "target_chars": 3200,
                "respect_page": True,
            },
        }

    Coalescing participates in the ChunkingSpec config hash, so durable
    metadata-first refresh cannot reuse chunks produced under a different bridge
    preprocessing contract.
    """
    blocks, coalescing_metadata = _bridge_blocks(
        document,
        chunking,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        language=language,
    )
    for block in blocks:
        segments, chunker_metadata = split_text(
            block.text,
            chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            language=language,
        )
        provenance: dict[str, Any] = {
            **chunker_metadata,
            "block_index": block.block_index,
            "block_kind": block.kind,
            "block_coalescing": dict(coalescing_metadata),
        }
        if block.bbox is not None:
            provenance["bbox"] = list(block.bbox)
        if block.metadata:
            provenance["block_metadata"] = dict(block.metadata)
        for text in segments:
            if not text.strip():
                continue
            yield ParsedSegment(
                text=text,
                block_index=block.block_index,
                block_kind=block.kind,
                page=block.page,
                section=block.section,
                bbox=block.bbox,
                metadata=dict(provenance),
            )
