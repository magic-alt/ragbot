from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

from ..chunking import resolve_chunking_spec, split_text
from ..chunking.languages import language_for_path
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
    "dist", "build", ".tox", ".eggs",
}

DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100

_LANG_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".jsx": "javascript", ".java": "java",
    ".go": "go", ".rs": "rust", ".c": "c", ".cpp": "cpp",
    ".h": "c", ".hpp": "cpp", ".cs": "csharp", ".rb": "ruby",
    ".php": "php", ".swift": "swift", ".kt": "kotlin",
    ".scala": "scala", ".sh": "shell", ".sql": "sql",
    ".html": "html", ".css": "css", ".yaml": "yaml",
    ".yml": "yaml", ".toml": "toml", ".json": "json",
    ".xml": "xml", ".md": "markdown", ".txt": "text",
}


def ingest_repo(
    url_or_path: str,
    doc_id: str,
    tenant_id: str,
    ref: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    version: str = "1.0",
    tags: Optional[list] = None,
    acl_hash: Optional[str] = None,
    chunking: Optional[dict] = None,
) -> Iterable[Chunk]:
    """Index repository files with native structural or framework code chunking.

    `ragbot/structural` remains the compatibility default. Sources can opt into
    `langchain/code`, `langchain/recursive`, or `llamaindex/sentence` without
    allowing those frameworks to own repository lifecycle, ACL, or persistence.
    """
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
        language = language_for_path(rel_path) or _LANG_MAP.get(file_path.suffix, "unknown")
        spec = resolve_chunking_spec(
            chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            language=language,
            default_strategy="structural",
        )

        if spec.provider == "ragbot" and spec.strategy == "structural":
            if file_path.suffix == ".py":
                segments = _split_python_symbols(content, spec.chunk_size)
            elif file_path.suffix in {".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp", ".cs"}:
                segments = _split_by_functions(content, spec.chunk_size)
            else:
                segments = _split_file(content, spec.chunk_size)
            chunker_metadata = spec.metadata()
        else:
            parts, chunker_metadata = split_text(
                content,
                chunking,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                language=language,
                default_strategy="structural",
            )
            segments = _locate_framework_segments(content, parts)

        for symbol_name, start_line, end_line, segment in segments:
            yield Chunk(
                chunk_id=uuid.uuid4().hex,
                doc_id=doc_id,
                tenant_id=tenant_id,
                chunk_index=idx,
                text=segment,
                path=rel_path,
                section=symbol_name,
                checksum=content_hash(segment),
                metadata={
                    "source_type": "repo",
                    "ref": ref or "main",
                    "version": version,
                    "tags": tags or [],
                    "acl_hash": acl_hash or "public",
                    "line_start": start_line,
                    "line_end": end_line,
                    "language": language,
                    **chunker_metadata,
                },
            )
            yield chunk if False else None
            idx += 1
    logger.info("Repo ingestion complete: %s -> %d chunks", url_or_path, idx)


def _locate_framework_segments(
    content: str,
    parts: list[str],
) -> List[Tuple[Optional[str], int, int, str]]:
    """Recover best-effort line spans without coupling framework adapters to Chunk."""
    result: List[Tuple[Optional[str], int, int, str]] = []
    cursor = 0
    for part in parts:
        position = content.find(part, cursor)
        if position < 0:
            position = content.find(part)
        if position < 0:
            position = cursor
        end = position + len(part)
        start_line = content.count("\n", 0, position) + 1
        end_line = content.count("\n", 0, end) + 1
        result.append((None, start_line, end_line, part))
        cursor = max(cursor, end)
    return result


# ── Python AST-based symbol chunking ──────────────────────────────────


def _split_python_symbols(content: str, chunk_size: int) -> List[Tuple[Optional[str], int, int, str]]:
    try:
        import ast
        tree = ast.parse(content)
    except SyntaxError:
        return _split_file(content, chunk_size)

    lines = content.splitlines(keepends=True)
    symbols: List[Tuple[str, int, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 1
            symbols.append((name, start, end))

    if not symbols:
        return _split_file(content, chunk_size)

    symbols.sort(key=lambda s: s[1])
    result: List[Tuple[Optional[str], int, int, str]] = []

    if symbols[0][1] > 0:
        preamble = "".join(lines[:symbols[0][1]]).strip()
        if preamble:
            result.append(("<module>", 1, symbols[0][1], preamble))

    for name, start, end in symbols:
        text = "".join(lines[start:end]).strip()
        if not text:
            continue
        if len(text) > chunk_size * 2:
            result.extend(_split_large_symbol(name, text, start, chunk_size))
        else:
            result.append((name, start + 1, end, text))

    return result


def _split_by_functions(content: str, chunk_size: int) -> List[Tuple[Optional[str], int, int, str]]:
    pattern = re.compile(
        r"^(?:(?:export\s+)?(?:async\s+)?(?:function|class|def|func|fn|pub\s+fn|public|private|protected|static)\s+\w+)",
        re.MULTILINE,
    )

    lines = content.splitlines(keepends=True)
    boundaries: List[int] = []

    for match in pattern.finditer(content):
        line_num = content[:match.start()].count("\n")
        boundaries.append(line_num)

    if not boundaries:
        return _split_file(content, chunk_size)

    result: List[Tuple[Optional[str], int, int, str]] = []

    if boundaries[0] > 0:
        preamble = "".join(lines[:boundaries[0]]).strip()
        if preamble:
            result.append(("<imports>", 1, boundaries[0], preamble))

    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
        text = "".join(lines[start:end]).strip()
        if not text:
            continue
        first_line = lines[start].strip() if start < len(lines) else ""
        name = _extract_symbol_name(first_line)

        if len(text) > chunk_size * 2:
            result.extend(_split_large_symbol(name or "<anonymous>", text, start, chunk_size))
        else:
            result.append((name, start + 1, end, text))

    return result


def _extract_symbol_name(line: str) -> Optional[str]:
    match = re.search(r"(?:function|class|def|func|fn)\s+(\w+)", line)
    if match:
        return match.group(1)
    match = re.search(r"(?:public|private|protected|static)\s+\w+\s+(\w+)\s*\(", line)
    if match:
        return match.group(1)
    return None


def _split_large_symbol(name: str, text: str, base_line: int, chunk_size: int) -> List[Tuple[Optional[str], int, int, str]]:
    lines = text.splitlines(keepends=True)
    result = []
    current: List[str] = []
    current_len = 0
    start_offset = 0
    part = 1

    for i, line in enumerate(lines):
        current.append(line)
        current_len += len(line)
        if current_len >= chunk_size:
            chunk_text = "".join(current).strip()
            if chunk_text:
                result.append((f"{name} (part {part})", base_line + start_offset + 1, base_line + i + 1, chunk_text))
                part += 1
            current = []
            current_len = 0
            start_offset = i + 1

    if current:
        chunk_text = "".join(current).strip()
        if chunk_text:
            label = f"{name} (part {part})" if part > 1 else name
            result.append((label, base_line + start_offset + 1, base_line + len(lines), chunk_text))

    return result


def _split_file(content: str, chunk_size: int) -> List[Tuple[Optional[str], int, int, str]]:
    lines = content.splitlines(keepends=True)
    result: List[Tuple[Optional[str], int, int, str]] = []
    current: List[str] = []
    current_len = 0
    start_line = 0
    for i, line in enumerate(lines):
        current.append(line)
        current_len += len(line)
        if current_len >= chunk_size:
            text = "".join(current).strip()
            if text:
                result.append((None, start_line + 1, i + 1, text))
            current = []
            current_len = 0
            start_line = i + 1
    if current:
        text = "".join(current).strip()
        if text:
            result.append((None, start_line + 1, len(lines), text))
    return result
