from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Iterable, Optional

from ..chunking import split_text
from ..connectors.pdf import fetch_pdf_bytes, fetch_pdf_pages  # fetch_pdf_pages kept for compatibility
from ..parsing import iter_document_segments, parse_document
from services.api.app.storage.models import Chunk
from services.worker.dedup.hashing import content_hash

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


def ingest_pdf(
    path: str,
    doc_id: str,
    tenant_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    version: str = "1.0",
    tags: Optional[list] = None,
    acl_hash: Optional[str] = None,
    chunking: Optional[dict] = None,
    parsing: Optional[dict] = None,
) -> Iterable[Chunk]:
    """Fetch a PDF resource, normalize it through Parser Port, then chunk blocks."""
    logger.info("Ingesting PDF: %s (doc_id=%s)", path, doc_id)
    try:
        body = fetch_pdf_bytes(path)
    except ValueError as exc:
        if "Local source is outside RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS" not in str(exc):
            raise
        requested = str(path)
        resolved = str(Path(requested).expanduser().resolve())
        allowed = os.getenv("RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS", "")
        raise ValueError(
            "PDF local path rejected: "
            f"requested={requested!r}, resolved={resolved!r}, allowed_roots={allowed!r}; {exc}"
        ) from exc

    document, parser_metadata = parse_document(
        body,
        parsing,
        name=Path(path).name or "document.pdf",
        media_type="application/pdf",
        uri=path,
    )
    if not document.blocks:
        logger.warning("No text extracted from PDF: %s", path)
        return

    chunk_index = 0
    for segment in iter_document_segments(
        document,
        chunking,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ):
        yield Chunk(
            chunk_id=uuid.uuid4().hex,
            doc_id=doc_id,
            tenant_id=tenant_id,
            chunk_index=chunk_index,
            text=segment.text,
            path=path,
            page=segment.page,
            section=segment.section,
            checksum=content_hash(segment.text),
            metadata={
                "source_type": "pdf",
                "version": version,
                "tags": tags or [],
                "acl_hash": acl_hash or "public",
                **parser_metadata,
                **segment.metadata,
            },
        )
        chunk_index += 1
    logger.info("PDF ingestion complete: %s -> %d chunks", path, chunk_index)


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Compatibility shim for old tests/callers; implementation lives in chunking/."""
    segments, _metadata = split_text(
        text,
        None,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return segments
