from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from services.worker.chunking import split_text

from .models import NormalizedDocument


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


def iter_document_segments(
    document: NormalizedDocument,
    chunking: Mapping[str, Any] | None,
    *,
    chunk_size: int,
    chunk_overlap: int,
    language: str | None = None,
) -> Iterable[ParsedSegment]:
    """Apply the configured Chunker independently to parser-owned blocks."""
    for block in document.blocks:
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
