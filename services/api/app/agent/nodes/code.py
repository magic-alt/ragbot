from __future__ import annotations

import difflib
import fnmatch
import logging
import re
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..state import AgentState, Citation, EvidenceItem, ToolCallRecord, now_ms
from ..reliability import safe_tool_call
from contracts.types import CodeSnippet, PatchResult

logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh",
    ".bash", ".sql", ".html", ".css", ".scss", ".yaml", ".yml", ".toml",
    ".json", ".xml", ".md", ".txt", ".cfg", ".ini", ".conf", ".r", ".R",
    ".lua", ".pl", ".pm", ".ex", ".exs", ".erl", ".hs", ".ml", ".vue",
    ".svelte",
})

_EXCLUDED_DIRS = frozenset({
    ".git", ".svn", ".hg", "node_modules", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".tox", ".venv", "venv", ".env", "dist", "build",
    ".idea", ".vscode",
})

_EXCLUDED_FILES = frozenset({
    ".env", ".env.local", ".env.production", ".env.staging",
    "credentials.json", "secrets.json", "id_rsa", "id_ed25519",
    ".htpasswd", ".netrc",
})


class CodeSearch:
    def __init__(self, repo_roots: Dict[str, str], in_memory_files: Optional[Dict[str, Dict[str, str]]] = None) -> None:
        self._repo_roots = repo_roots
        self._in_memory_files = in_memory_files or {}

    def search(self, query: str, repo: str, ref: str = "main", path_glob: Optional[str] = None, max_hits: int = 5) -> List[CodeSnippet]:
        try:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
        except re.error:
            logger.warning("Invalid regex pattern after escaping: %s", query)
            return []
        files = self._collect_files(repo, path_glob)
        hits: List[CodeSnippet] = []
        for path, content in files:
            for match in pattern.finditer(content):
                start_line, end_line, snippet = _extract_snippet(content, match.start(), match.end())
                hits.append(
                    CodeSnippet(
                        path=path,
                        ref=ref,
                        line_start=start_line,
                        line_end=end_line,
                        content=snippet,
                    )
                )
                if len(hits) >= max_hits:
                    return hits
        return hits

    def open_file(
        self, path: str, repo: str, ref: str = "main",
        start_line: Optional[int] = None, end_line: Optional[int] = None,
    ) -> str:
        """Read a file or line range with line numbers."""
        content = self._read_file(path, repo)
        if content is None:
            raise FileNotFoundError(f"File not found: {path} in repo {repo}")
        lines = content.splitlines()
        start = max(1, start_line or 1)
        end = min(len(lines), end_line or len(lines))
        numbered = []
        for i in range(start - 1, end):
            numbered.append(f"{i + 1:>5} | {lines[i]}")
        return "\n".join(numbered)

    def generate_patch(
        self, path: str, original: str, replacement: str, repo: str = "default",
    ) -> PatchResult:
        """Generate a unified diff for a code change."""
        orig_lines = original.splitlines(keepends=True)
        new_lines = replacement.splitlines(keepends=True)
        diff = difflib.unified_diff(
            orig_lines, new_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
        diff_text = "".join(diff)
        return PatchResult(
            path=path,
            diff=diff_text,
            original_lines=len(orig_lines),
            modified_lines=len(new_lines),
        )

    def explain_error(
        self, error_text: str, repo: str, ref: str = "main", max_hits: int = 5,
    ) -> List[CodeSnippet]:
        """Parse an error/stack trace and find related code locations."""
        locations = _parse_stack_trace(error_text)
        snippets: List[CodeSnippet] = []
        for file_path, line_num in locations:
            try:
                content = self._read_file(file_path, repo)
                if content is None:
                    continue
                lines = content.splitlines()
                ctx = 3
                start = max(0, line_num - 1 - ctx)
                end = min(len(lines), line_num + ctx)
                snippet_text = "\n".join(lines[start:end])
                snippets.append(CodeSnippet(
                    path=file_path, ref=ref,
                    line_start=start + 1, line_end=end,
                    content=snippet_text,
                ))
                if len(snippets) >= max_hits:
                    break
            except Exception:
                continue
        if not snippets:
            keywords = _extract_error_keywords(error_text)
            for kw in keywords[:3]:
                snippets.extend(self.search(kw, repo, ref, max_hits=2))
                if len(snippets) >= max_hits:
                    break
        return snippets[:max_hits]

    def _read_file(self, path: str, repo: str) -> Optional[str]:
        """Read a single file from a repo by path."""
        if repo in self._in_memory_files:
            return self._in_memory_files[repo].get(path)
        root = self._repo_roots.get(repo)
        if not root:
            return None
        root_path = Path(root).resolve()
        file_path = (root_path / path).resolve()
        if not str(file_path).startswith(str(root_path)):
            return None
        if not file_path.is_file():
            return None
        if file_path.name in _EXCLUDED_FILES:
            return None
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception:
            return None

    def _collect_files(self, repo: str, path_glob: Optional[str]) -> Iterable[tuple[str, str]]:
        if repo in self._in_memory_files:
            for path, content in self._in_memory_files[repo].items():
                if path_glob and not fnmatch.fnmatch(path, path_glob):
                    continue
                yield path, content
            return
        root = self._repo_roots.get(repo)
        if not root:
            return []
        root_path = Path(root).resolve()
        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue
            resolved = file_path.resolve()
            if not str(resolved).startswith(str(root_path)):
                continue
            if any(part in _EXCLUDED_DIRS for part in resolved.parts):
                continue
            if resolved.name in _EXCLUDED_FILES:
                continue
            if resolved.suffix not in _ALLOWED_EXTENSIONS:
                continue
            rel_path = str(file_path.relative_to(root_path))
            if path_glob and not fnmatch.fnmatch(rel_path, path_glob):
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            yield rel_path, content


async def code_node(state: AgentState, services: Any) -> AgentState:
    repo = state.constraints.repo or "default"
    params = {"query": state.query, "repo": repo, "ref": state.constraints.ref or "main"}
    start_ms = now_ms()
    try:
        snippets = await safe_tool_call("code_search", services.code_search.search, state.query, repo=repo, max_hits=8)
        citations = [
            Citation(
                kind="code",
                path=snippet.path,
                ref=snippet.ref,
                line_start=snippet.line_start,
                line_end=snippet.line_end,
            )
            for snippet in snippets
        ]
        text = _format_snippets(snippets, limit=8)
        state.evidence.append(
            EvidenceItem(
                kind="code_snippets",
                score=1.0,
                text=text,
                citations=citations,
                metadata={"count": len(snippets)},
            )
        )
        record = ToolCallRecord(
            name="code_search",
            args=params,
            ok=True,
            started_at_ms=start_ms,
            ended_at_ms=now_ms(),
            result_preview={"count": len(snippets)},
        )
    except Exception as exc:
        record = ToolCallRecord(
            name="code_search",
            args=params,
            ok=False,
            started_at_ms=start_ms,
            ended_at_ms=now_ms(),
            error=str(exc),
        )
    state.tool_calls.append(record)
    return state


def _format_snippets(snippets: List[CodeSnippet], limit: int = 8) -> str:
    parts: List[str] = []
    for snippet in snippets[:limit]:
        header = f"{snippet.path}:{snippet.line_start}-{snippet.line_end}"
        body = snippet.content.strip().replace("\n", " ")
        parts.append(f"{header} {body}")
    return " ".join(parts)


def _extract_snippet(content: str, start: int, end: int, context: int = 2) -> tuple[int, int, str]:
    lines = content.splitlines()
    char_count = 0
    line_index = 0
    for idx, line in enumerate(lines):
        line_len = len(line) + 1
        if char_count + line_len > start:
            line_index = idx
            break
        char_count += line_len
    start_line = max(0, line_index - context)
    end_line = min(len(lines), line_index + context + 1)
    snippet = "\n".join(lines[start_line:end_line])
    return start_line + 1, end_line, snippet


# ── Stack trace parsing helpers ──────────────────────────────────────


def _parse_stack_trace(text: str) -> List[Tuple[str, int]]:
    """Extract (file_path, line_number) pairs from stack traces."""
    locations: List[Tuple[str, int]] = []
    # Python: File "path", line N
    for m in re.finditer(r'File ["\']([^"\']+)["\'],\s*line\s+(\d+)', text):
        locations.append((m.group(1), int(m.group(2))))
    # JavaScript/Node: at ... (path:line:col) or path:line:col
    for m in re.finditer(r'(?:at\s+.*?\(|at\s+)([^\s():]+):(\d+):\d+', text):
        locations.append((m.group(1), int(m.group(2))))
    # Go: path.go:line
    for m in re.finditer(r'([^\s]+\.go):(\d+)', text):
        locations.append((m.group(1), int(m.group(2))))
    # Java: at package.Class.method(File.java:line)
    for m in re.finditer(r'\(([^)]+\.java):(\d+)\)', text):
        locations.append((m.group(1), int(m.group(2))))
    # Generic: filename.ext:line
    if not locations:
        for m in re.finditer(r'([^\s:]+\.\w+):(\d+)', text):
            locations.append((m.group(1), int(m.group(2))))
    return locations


def _extract_error_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from error text for fallback search."""
    keywords: List[str] = []
    # Extract exception/error class names
    for m in re.finditer(r'(\w+(?:Error|Exception|Fault|Failure))', text):
        kw = m.group(1)
        if kw not in keywords:
            keywords.append(kw)
    # Extract function/method names from stack frames
    for m in re.finditer(r'in\s+(\w+)', text):
        kw = m.group(1)
        if kw not in keywords and len(kw) > 2:
            keywords.append(kw)
    return keywords


# ── Agent node functions ─────────────────────────────────────────────


async def open_file_node(state: AgentState, services: Any) -> AgentState:
    """Agent node: read a file or file range."""
    repo = state.constraints.repo or "default"
    # Extract path from query (format: "open path/to/file.py:10-20" or just path)
    path, start_line, end_line = _parse_file_reference(state.query)
    params = {"path": path, "repo": repo, "start_line": start_line, "end_line": end_line}
    start_ms = now_ms()
    try:
        content = await safe_tool_call(
            "open_file", services.code_search.open_file,
            path=path, repo=repo,
            start_line=start_line, end_line=end_line,
        )
        citations = [Citation(kind="code", path=path, ref=state.constraints.ref or "main",
                              line_start=start_line, line_end=end_line)]
        state.evidence.append(EvidenceItem(
            kind="file_content", score=1.0, text=content,
            citations=citations,
            metadata={"path": path, "lines": f"{start_line or 1}-{end_line or '?'}"},
        ))
        record = ToolCallRecord(
            name="open_file", args=params, ok=True,
            started_at_ms=start_ms, ended_at_ms=now_ms(),
            result_preview={"path": path, "length": len(content)},
        )
    except Exception as exc:
        record = ToolCallRecord(
            name="open_file", args=params, ok=False,
            started_at_ms=start_ms, ended_at_ms=now_ms(), error=str(exc),
        )
    state.tool_calls.append(record)
    return state


async def apply_patch_node(state: AgentState, services: Any) -> AgentState:
    """Agent node: generate a unified diff patch."""
    repo = state.constraints.repo or "default"
    # The query should contain the patch instruction; evidence should have the code context
    params = {"query": state.query, "repo": repo}
    start_ms = now_ms()
    try:
        # Look for file_content evidence to use as the original
        original_text = ""
        patch_path = ""
        for ev in reversed(state.evidence):
            if ev.kind == "file_content" and ev.text:
                # Strip line numbers from open_file output
                raw_lines = []
                for line in ev.text.splitlines():
                    if " | " in line:
                        raw_lines.append(line.split(" | ", 1)[1])
                    else:
                        raw_lines.append(line)
                original_text = "\n".join(raw_lines) + "\n"
                patch_path = ev.metadata.get("path", "unknown")
                break
            if ev.kind == "code_snippets" and ev.citations:
                original_text = ev.text + "\n"
                patch_path = ev.citations[0].path or "unknown"
                break

        if not original_text:
            raise ValueError("No source code in evidence to generate patch from")

        # For now, create a placeholder patch showing the intent
        # The actual replacement would come from LLM in a full implementation
        result = services.code_search.generate_patch(
            path=patch_path,
            original=original_text,
            replacement=original_text,  # Identity patch as placeholder
            repo=repo,
        )
        state.evidence.append(EvidenceItem(
            kind="patch", score=1.0,
            text=f"Patch for {patch_path}:\n{result.diff}" if result.diff else f"No changes needed for {patch_path}",
            citations=[Citation(kind="code", path=patch_path)],
            metadata={"path": patch_path, "original_lines": result.original_lines,
                       "modified_lines": result.modified_lines},
        ))
        record = ToolCallRecord(
            name="apply_patch", args=params, ok=True,
            started_at_ms=start_ms, ended_at_ms=now_ms(),
            result_preview={"path": patch_path, "diff_len": len(result.diff)},
        )
    except Exception as exc:
        record = ToolCallRecord(
            name="apply_patch", args=params, ok=False,
            started_at_ms=start_ms, ended_at_ms=now_ms(), error=str(exc),
        )
    state.tool_calls.append(record)
    return state


async def explain_error_node(state: AgentState, services: Any) -> AgentState:
    """Agent node: parse stack trace and find related code."""
    repo = state.constraints.repo or "default"
    params = {"error_text": state.query, "repo": repo}
    start_ms = now_ms()
    try:
        snippets = await safe_tool_call(
            "explain_error", services.code_search.explain_error,
            error_text=state.query, repo=repo, max_hits=5,
        )
        citations = [
            Citation(kind="code", path=s.path, ref=s.ref,
                     line_start=s.line_start, line_end=s.line_end)
            for s in snippets
        ]
        text = _format_snippets(snippets, limit=5)
        state.evidence.append(EvidenceItem(
            kind="error_analysis", score=1.0, text=text,
            citations=citations,
            metadata={"count": len(snippets), "locations_found": len(snippets)},
        ))
        record = ToolCallRecord(
            name="explain_error", args=params, ok=True,
            started_at_ms=start_ms, ended_at_ms=now_ms(),
            result_preview={"count": len(snippets)},
        )
    except Exception as exc:
        record = ToolCallRecord(
            name="explain_error", args=params, ok=False,
            started_at_ms=start_ms, ended_at_ms=now_ms(), error=str(exc),
        )
    state.tool_calls.append(record)
    return state


def _parse_file_reference(query: str) -> Tuple[str, Optional[int], Optional[int]]:
    """Parse a file path with optional line range from query."""
    # Try format: "path/to/file.py:10-20"
    m = re.search(r'([^\s]+\.\w+):(\d+)-(\d+)', query)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    # Try format: "path/to/file.py:10"
    m = re.search(r'([^\s]+\.\w+):(\d+)', query)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(2)) + 20
    # Just a path
    m = re.search(r'([^\s]+\.\w+)', query)
    if m:
        return m.group(1), None, None
    return query.strip(), None, None

