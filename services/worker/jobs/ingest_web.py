from __future__ import annotations

import logging
import uuid
from pathlib import PurePosixPath
from typing import Iterable, Optional
from urllib.parse import urlparse

from ..connectors.web import fetch_web_resource
from ..parsing import iter_document_segments, parse_document
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
    parsing: Optional[dict] = None,
) -> Iterable[Chunk]:
    """Fetch raw web content, normalize blocks, then apply the configured Chunker."""
    logger.info("Ingesting web page: %s (doc_id=%s)", url, doc_id)
    resource = fetch_web_resource(url)
    parser_config = _with_encoding(parsing, resource.encoding)
    parsed_path = PurePosixPath(urlparse(resource.url).path)
    name = parsed_path.name or ("index.html" if "html" in resource.content_type else "resource.txt")
    document, parser_metadata = parse_document(
        resource.body,
        parser_config,
        name=name,
        media_type=resource.content_type,
        uri=resource.url,
    )
    if not document.blocks:
        logger.warning("No text extracted from URL: %s", url)
        return

    count = 0
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
            chunk_index=count,
            text=segment.text,
            url=resource.url,
            page=segment.page,
            section=segment.section,
            checksum=content_hash(segment.text),
            metadata={
                "source_type": "web",
                "version": version,
                "tags": tags or [],
                "acl_hash": acl_hash or "public",
                "content_type": resource.content_type,
                **parser_metadata,
                **segment.metadata,
            },
        )
        count += 1
    logger.info("Web ingestion complete: %s -> %d chunks", url, count)


def _with_encoding(parsing: Optional[dict], encoding: str) -> dict:
    config = dict(parsing or {})
    options = dict(config.get("options") or {})
    options.setdefault("encoding", encoding or "utf-8")
    config["options"] = options
    return config
