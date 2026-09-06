from __future__ import annotations

from collections.abc import Iterable

from .models import DocumentBlock, NormalizedDocument

_STANDALONE_KINDS = {"table", "code", "image", "figure"}


def _bbox_union(blocks: Iterable[DocumentBlock]) -> tuple[float, float, float, float] | None:
    boxes = [block.bbox for block in blocks if block.bbox is not None]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _section_compatible(left: str | None, right: str | None) -> bool:
    # Missing section metadata should not prevent useful coalescing. When both
    # parsers provide a section, however, preserve that semantic boundary.
    return left is None or right is None or left == right


def _flush(blocks: list[DocumentBlock], separator: str) -> DocumentBlock | None:
    if not blocks:
        return None
    if len(blocks) == 1:
        block = blocks[0]
        metadata = dict(block.metadata)
        metadata.setdefault("source_block_indices", [block.block_index])
        metadata.setdefault("source_block_kinds", [block.kind])
        metadata["source_block_count"] = 1
        metadata["coalesced"] = False
        return DocumentBlock(
            block_index=block.block_index,
            text=block.text,
            kind=block.kind,
            page=block.page,
            section=block.section,
            bbox=block.bbox,
            metadata=metadata,
        )

    pages = {block.page for block in blocks if block.page is not None}
    sections = {block.section for block in blocks if block.section}
    source_boxes = [list(block.bbox) for block in blocks if block.bbox is not None]
    metadata = {
        "coalesced": True,
        "source_block_count": len(blocks),
        "source_block_indices": [block.block_index for block in blocks],
        "source_block_kinds": [block.kind for block in blocks],
    }
    if source_boxes:
        metadata["source_bboxes"] = source_boxes
    return DocumentBlock(
        block_index=blocks[0].block_index,
        text=separator.join(block.text.strip() for block in blocks if block.text.strip()),
        kind="coalesced_text",
        page=next(iter(pages)) if len(pages) == 1 else None,
        section=next(iter(sections)) if len(sections) == 1 else blocks[0].section,
        bbox=_bbox_union(blocks),
        metadata=metadata,
    )


def coalesce_document_blocks(
    document: NormalizedDocument,
    *,
    target_chars: int,
    respect_page: bool = True,
    respect_section: bool = True,
    separator: str = "\n\n",
) -> list[DocumentBlock]:
    """Coalesce parser-owned blocks into chunking windows without losing provenance.

    Structured parsers such as PyMuPDF and Docling may emit many small blocks.
    Sending each block independently to a chunker prevents the configured chunk
    budget from spanning adjacent paragraphs and can explode the embedding/index
    cardinality. This function groups compatible blocks before semantic/fixed
    splitting while retaining page, bbox and source-block provenance.

    The function deliberately does not split oversized source blocks; the normal
    chunker remains responsible for enforcing the final chunk budget.
    """
    if target_chars < 1:
        raise ValueError("target_chars must be >= 1")
    if not separator:
        raise ValueError("separator must not be empty")

    output: list[DocumentBlock] = []
    pending: list[DocumentBlock] = []
    pending_chars = 0

    def emit_pending() -> None:
        nonlocal pending, pending_chars
        merged = _flush(pending, separator)
        if merged is not None:
            output.append(merged)
        pending = []
        pending_chars = 0

    for block in document.blocks:
        kind = block.kind.casefold()
        standalone = kind in _STANDALONE_KINDS or "table" in kind or "code" in kind
        if standalone:
            emit_pending()
            merged = _flush([block], separator)
            if merged is not None:
                output.append(merged)
            continue

        if pending:
            previous = pending[-1]
            page_break = respect_page and previous.page != block.page
            section_break = respect_section and not _section_compatible(previous.section, block.section)
            projected = pending_chars + len(separator) + len(block.text)
            size_break = projected > target_chars
            if page_break or section_break or size_break:
                emit_pending()

        pending.append(block)
        pending_chars = len(block.text) if len(pending) == 1 else pending_chars + len(separator) + len(block.text)

    emit_pending()
    return output
