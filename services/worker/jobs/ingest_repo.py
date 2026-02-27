from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Iterable, Optional, Set

from ..connectors.git import fetch_git
from services.api.app.storage.models import Chunk
from services.worker.dedup.hashing import content_hash

logger = logging.getLogger(__name__)

_CODE_EXTENSIONS: Set[str] = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh",
    ".sql", ".html", ".css", ".yaml", ".yml", ".toml", ".json", ".xml",
    ".md", ".txt",
}

_EXCLUDED_DIRS: Set[str] = {
    ".git", "node_modules", "__pycache__", ".mypy_cache", ".venv", "venv",
    "dist", "build",
}

DEFAULT_CHUNK_SIZE = 600


def ingest_repo(
    url_or_path: str,
    doc_id: str,
    tenant_id: str,
    ref: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    version: str = "1.0",
    tags: Optional[list] = None,
    acl_hash: Optional[str] = None,
) -> Iterable[Chunk]:
    """Clone/open a repo and yield Chunk objects for each source file."""
    logger.info("Ingesting repo: %s (doc_id=%s)", url_or_path, doc_id)
    repo_path = fetch_git(url_or_path, ref=ref)
    root = Path(repo_path)
    idx = 0
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if any(part in _EXCLUDED_DIRS for part in file_path.parts):
            continue
        if file_path.suffix not in _CODE_EXTENSIONS:
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            continue
        rel_path = str(file_path.relative_to(root))
        segments = _split_file(content, chunk_size)
        for segment in segments:
            chunk = Chunk(
                chunk_id=uuid.uuid4().hex,
                doc_id=doc_id,
                tenant_id=tenant_id,
                chunk_index=idx,
                text=segment,
                path=rel_path,
                checksum=content_hash(segment),
                metadata={
                    "source_type": "repo",
                    "ref": ref or "main",
                    "version": version,
                    "tags": tags or [],
                    "acl_hash": acl_hash or "public",
                },
            )
            yield chunk
            idx += 1
    logger.info("Repo ingestion complete: %s -> %d chunks", url_or_path, idx)


def _split_file(content: str, chunk_size: int) -> list[str]:
    lines = content.splitlines(keepends=True)
    segments: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        current.append(line)
        current_len += len(line)
        if current_len >= chunk_size:
            segments.append("".join(current).strip())
            current = []
            current_len = 0
    if current:
        text = "".join(current).strip()
        if text:
            segments.append(text)
    return segments
