from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..state import AgentState, Citation, EvidenceItem, ToolCallRecord, now_ms
from contracts.types import CodeSnippet


class CodeSearch:
    def __init__(self, repo_roots: Dict[str, str], in_memory_files: Optional[Dict[str, Dict[str, str]]] = None) -> None:
        self._repo_roots = repo_roots
        self._in_memory_files = in_memory_files or {}

    def search(self, query: str, repo: str, ref: str = "main", path_glob: Optional[str] = None, max_hits: int = 5) -> List[CodeSnippet]:
        pattern = re.compile(query)
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
        root_path = Path(root)
        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue
            rel_path = str(file_path.relative_to(root_path))
            if path_glob and not fnmatch.fnmatch(rel_path, path_glob):
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            yield rel_path, content


def code_node(state: AgentState, services: Any) -> AgentState:
    repo = state.constraints.repo or "default"
    params = {"query": state.query, "repo": repo, "ref": state.constraints.ref or "main"}
    start_ms = now_ms()
    try:
        snippets = services.code_search.search(state.query, repo=repo, max_hits=8)
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

