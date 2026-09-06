from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from services.worker.chunking import split_text

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


def _coalescing_options(
    chunking: Mapping[str, Any] | None,
    *,
    chunk_size: int,
) -> tuple[bool, int, bool, bool]:
    raw: Any = (chunking or {}).get("block_coalescing") if isinstance(chunking, Mapping) else None
    if raw in (None, False):
        return False, chunk_size * 4, True, True
    if raw is True:
        return True, chunk_size * 4, True, True
    if not isinstance(raw, Mapping):
        raise ValueError("chunking.block_coalescing must be a boolean or object")

    enabled = bool(raw.get("enabled", True))
    target_chars = int(raw.get("target_chars") or chunk_size * 4)
    if target_chars < 1:
        raise ValueError("chunking.block_coalescing.target_chars must be >= 1")
    return (
        enabled,
        target_chars,
        bool(raw.get("respect_page", True)),
        bool(raw.get("respect_section", True)),
    )


def _bridge_blocks(
    document: NormalizedDocument,
    chunking: Mapping[str, Any] | None,
    *,
    chunk_size: int,
) -> tuple[list[DocumentBlock], dict[str, Any]]:
    enabled, target_chars, respect_page, respect_section = _coalescing_options(
        chunking,
        chunk_size=chunk_size,
    )
    if not enabled:
        return list(document.blocks), {
            "enabled": False,
            "input_blocks": len(document.blocks),
            "output_blocks": len(document.blocks),
        }

    blocks = coalesce_document_blocks(
        document,
        target_chars=target_chars,
        respect_page=respect_page,
        respect_section=respect_section,
    )
    return blocks, {
        "enabled": True,
        "target_chars": target_chars,
        "respect_page": respect_page,
        "respect_section": respect_section,
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
    explicitly promoted. Example::

        {
            "provider": "llamaindex",
            "strategy": "sentence",
            "block_coalescing": {
                "enabled": true,
                "target_chars": 3200,
                "respect_page": true
            }
        }
    """
    blocks, coalescing_metadata = _bridge_blocks(
        document,
        chunking,
        chunk_size=chunk_size,
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
