from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Iterable, Optional

from ..connectors.pdf import fetch_pdf
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
) -> Iterable[Chunk]:
    """Extract text from a PDF and yield Chunk objects."""
    logger.info("Ingesting PDF: %s (doc_id=%s)", path, doc_id)
    try:
        text = fetch_pdf(path)
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
    if not text or text == path:
        logger.warning("No text extracted from PDF: %s", path)
        return

    segments = _split_text(text, chunk_size, chunk_overlap)
    for idx, segment in enumerate(segments):
        chunk = Chunk(
            chunk_id=uuid.uuid4().hex,
            doc_id=doc_id,
            tenant_id=tenant_id,
            chunk_index=idx,
            text=segment,
            path=path,
            checksum=content_hash(segment),
            metadata={
                "source_type": "pdf",
                "version": version,
                "tags": tags or [],
                "acl_hash": acl_hash or "public",
            },
        )
        yield chunk
    logger.info("PDF ingestion complete: %s -> %d chunks", path, len(segments))


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    segments: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        segment = text[start:end].strip()
        if segment:
            segments.append(segment)
        start = end - overlap if end < len(text) else end
    return segments
