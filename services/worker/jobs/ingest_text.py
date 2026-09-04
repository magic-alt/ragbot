"""Ingest local files through Parser Port followed by Ragbot's Chunker Port."""
from __future__ import annotations

import logging
import mimetypes
import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Set

from ..chunking import split_text
from ..chunking.languages import language_for_path
from ..connectors.local_fs import list_files, read_file_bytes
from ..parsing import iter_document_segments, parse_document
from services.api.app.storage.models import Chunk
from services.worker.dedup.hashing import content_hash

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


def ingest_text_file(
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
    logger.info("Ingesting local file: %s (doc_id=%s)", path, doc_id)
    body = read_file_bytes(path)
    if not body:
        logger.warning("Empty file: %s", path)
        return

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    source_type = "markdown" if suffix in {".md", ".markdown"} else "text"
    language = language_for_path(path)
    media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    document, parser_metadata = parse_document(
        body,
        parsing,
        name=file_path.name,
        media_type=media_type,
        uri=str(file_path),
    )
    if not document.blocks:
        logger.warning("No content parsed from file: %s", path)
        return

    count = 0
    for segment in iter_document_segments(
        document,
        chunking,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        language=language,
    ):
        section = segment.section
        if section is None and source_type == "markdown":
            section = _extract_section(segment.text)
        yield Chunk(
            chunk_id=uuid.uuid4().hex,
            doc_id=doc_id,
            tenant_id=tenant_id,
            chunk_index=count,
            text=segment.text,
            path=path,
            page=segment.page,
            section=section,
            checksum=content_hash(segment.text),
            metadata={
                "source_type": source_type,
                "filename": file_path.name,
                "media_type": media_type,
                "version": version,
                "tags": tags or [],
                "acl_hash": acl_hash or "public",
                **parser_metadata,
                **segment.metadata,
            },
        )
        count += 1
    logger.info("Local file ingestion complete: %s -> %d chunks", path, count)


def ingest_local_fs(
    directory: str,
    doc_id: str,
    tenant_id: str,
    extensions: Optional[List[str]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    version: str = "1.0",
    tags: Optional[list] = None,
    acl_hash: Optional[str] = None,
    chunking: Optional[dict] = None,
    parsing: Optional[dict] = None,
) -> Iterable[Chunk]:
    """Scan a directory and ingest matching resources as file-level documents."""
    logger.info("Ingesting local_fs directory: %s (doc_id=%s)", directory, doc_id)
    ext_set: Optional[Set[str]] = None
    if extensions:
        ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}

    files = list_files(directory, extensions=ext_set)
    logger.info("Found %d files to ingest in %s", len(files), directory)

    total_chunks = 0
    for file_path in files:
        for chunk in ingest_text_file(
            path=file_path,
            doc_id=f"{doc_id}:{Path(file_path).name}",
            tenant_id=tenant_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            version=version,
            tags=tags,
            acl_hash=acl_hash,
            chunking=chunking,
            parsing=parsing,
        ):
            total_chunks += 1
            yield chunk

    logger.info("Local FS ingestion complete: %s -> %d total chunks", directory, total_chunks)


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Compatibility shim; fixed splitting is centralized under chunking/."""
    segments, _metadata = split_text(
        text,
        None,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return segments


def _extract_section(text: str) -> Optional[str]:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return None
