"""Ingest local text/markdown files through Ragbot's document transformation port."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Set

from ..chunking import split_text
from ..chunking.languages import language_for_path
from ..connectors.local_fs import list_files, read_file
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
) -> Iterable[Chunk]:
    logger.info("Ingesting text file: %s (doc_id=%s)", path, doc_id)
    text = read_file(path)
    if not text.strip():
        logger.warning("Empty file: %s", path)
        return

    file_path = Path(path)
    source_type = "markdown" if file_path.suffix.lower() in {".md", ".markdown"} else "text"
    language = language_for_path(path)
    segments, chunker_metadata = split_text(
        text,
        chunking,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        language=language,
    )
    for idx, segment in enumerate(segments):
        yield Chunk(
            chunk_id=uuid.uuid4().hex,
            doc_id=doc_id,
            tenant_id=tenant_id,
            chunk_index=idx,
            text=segment,
            path=path,
            section=_extract_section(segment) if source_type == "markdown" else None,
            checksum=content_hash(segment),
            metadata={
                "source_type": source_type,
                "filename": file_path.name,
                "version": version,
                "tags": tags or [],
                "acl_hash": acl_hash or "public",
                **chunker_metadata,
            },
        )
    logger.info("Text file ingestion complete: %s -> %d chunks", path, len(segments))


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
) -> Iterable[Chunk]:
    """Scan a directory and ingest matching files as file-level documents."""
    logger.info("Ingesting local_fs directory: %s (doc_id=%s)", directory, doc_id)
    ext_set: Optional[Set[str]] = None
    if extensions:
        ext_set = {e if e.startswith(".") else f".{e}" for e in extensions}

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
