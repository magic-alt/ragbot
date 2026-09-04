from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Iterable, Optional

from ..chunking import split_text
from ..connectors.pdf import fetch_pdf_pages
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
) -> Iterable[Chunk]:
    """Extract page-aware PDF chunks through Ragbot's document-transform port."""
    logger.info("Ingesting PDF: %s (doc_id=%s)", path, doc_id)
    try:
        pages = fetch_pdf_pages(path)
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
    if not pages:
        logger.warning("No text extracted from PDF: %s", path)
        return

    chunk_index = 0
    for page_number, page_text in pages:
        segments, chunker_metadata = split_text(
            page_text,
            chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for segment in segments:
            chunk = Chunk(
                chunk_id=uuid.uuid4().hex,
                doc_id=doc_id,
                tenant_id=tenant_id,
                chunk_index=chunk_index,
                text=segment,
                path=path,
                page=page_number,
                checksum=content_hash(segment),
                metadata={
                    "source_type": "pdf",
                    "version": version,
                    "tags": tags or [],
                    "acl_hash": acl_hash or "public",
                    "parser_provider": "pypdf2",
                    "parser_version": 1,
                    **chunker_metadata,
                },
            )
            chunk_index += 1
            yield chunk
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
