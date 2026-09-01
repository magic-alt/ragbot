"""Local filesystem connector with production path boundaries."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set

from .security import validate_local_source_path

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
    """List allowed text files without escaping configured source roots."""
    exts = extensions or DEFAULT_EXTENSIONS
    root = Path(validate_local_source_path(directory))
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    result: List[str] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if any(part in _EXCLUDED_DIRS for part in file_path.parts):
            continue
        if file_path.suffix.lower() not in exts:
            continue
        # Resolve every file separately so a symlink cannot escape an allowed
        # directory when production roots are configured.
        safe_path = validate_local_source_path(str(file_path))
        result.append(safe_path)
        if len(result) >= max_files:
            logger.warning("File limit reached (%d), stopping scan", max_files)
            break
    return result


def read_file(path: str, max_size: int = 10 * 1024 * 1024) -> str:
    """Read at most ``max_size`` bytes from an allowed local text file."""
    file_path = Path(validate_local_source_path(path))
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    size = file_path.stat().st_size
    if size > max_size:
        logger.warning("File too large (%d bytes), reading first %d bytes", size, max_size)
    with file_path.open("rb") as handle:
        raw = handle.read(max_size)
    return raw.decode("utf-8", errors="replace")
