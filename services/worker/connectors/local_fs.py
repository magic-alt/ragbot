"""Local filesystem connector.

Lists and reads files from a local directory, filtering by extension.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".csv", ".log"}

_EXCLUDED_DIRS: Set[str] = {
    ".git", "node_modules", "__pycache__", ".mypy_cache", ".venv", "venv",
    "dist", "build", ".tox", ".eggs",
}


def list_files(
    directory: str,
    extensions: Optional[Set[str]] = None,
    max_files: int = 10000,
) -> List[str]:
    """List files in a directory matching the given extensions."""
    exts = extensions or DEFAULT_EXTENSIONS
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    result: List[str] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if any(part in _EXCLUDED_DIRS for part in file_path.parts):
            continue
        if file_path.suffix.lower() in exts:
            result.append(str(file_path))
        if len(result) >= max_files:
            logger.warning("File limit reached (%d), stopping scan", max_files)
            break
    return result


def read_file(path: str, max_size: int = 10 * 1024 * 1024) -> str:
    """Read a text file, returning its content. Limits to max_size bytes."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    size = file_path.stat().st_size
    if size > max_size:
        logger.warning("File too large (%d bytes), truncating to %d", size, max_size)

    return file_path.read_text(encoding="utf-8", errors="replace")[:max_size]
