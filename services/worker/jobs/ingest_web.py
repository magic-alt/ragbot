from __future__ import annotations

import logging
import uuid
from typing import Iterable, Optional

from ..chunking import split_text
from ..connectors.web import fetch_web
from services.api.app.storage.models import Chunk
from services.worker.dedup.hashing import content_hash

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


def ingest_web(
    url: str,
    doc_id: str,
    tenant_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    version: str = "1.0",
    tags: Optional[list] = None,
    acl_hash: Optional[str] = None,
    chunking: Optional[dict] = None,
) -> Iterable[Chunk]:
    """Fetch a web page and chunk it through the configured transformation adapter."""
    logger.info("Ingesting web page: %s (doc_id=%s)", url, doc_id)
    text = fetch_web(url)
    if not text or text == url:
        logger.warning("No text extracted from URL: %s", url)
        return

    segments, chunker_metadata = split_text(
        text,
        chunking,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    for idx, segment in enumerate(segments):
        yield Chunk(
            chunk_id=uuid.uuid4().hex,
            doc_id=doc_id,
            tenant_id=tenant_id,
            chunk_index=idx,
            text=segment,
            url=url,
            checksum=content_hash(segment),
            metadata={
                "source_type": "web",
                "version": version,
                "tags": tags or [],
                "acl_hash": acl_hash or "public",
                **chunker_metadata,
            },
        )
    logger.info("Web ingestion complete: %s -> %d chunks", url, len(segments))
