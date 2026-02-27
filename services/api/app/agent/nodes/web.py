from __future__ import annotations

import logging
from typing import Any, List, Optional
from urllib.parse import urlparse

from ..state import AgentState, Citation, EvidenceItem, ToolCallRecord, now_ms

logger = logging.getLogger(__name__)


def web_node(state: AgentState, services: Any) -> AgentState:
    args = {"query": state.query, "recency_days": 30}
    start_ms = now_ms()
    try:
        llm = getattr(services, "llm", None)
        allowed_domains = _domains_from_prefix(state.constraints.url_prefix)
        snippets = []
        if llm and getattr(llm, "enabled", False):
            snippets = llm.web_search(state.query, allowed_domains=allowed_domains, recency_days=30)
            executed = True
        else:
            executed = False
        citations = [Citation(kind="web", url=snip.get("url"), title=snip.get("title")) for snip in snippets]
        text = _format_snippets(snippets, limit=6)
        if executed:
            state.evidence.append(
                EvidenceItem(
                    kind="web_snippets",
                    score=1.0 if snippets else 0.0,
                    text=text,
                    citations=citations,
                    metadata={"count": len(snippets), "items": snippets},
                )
            )
        record = ToolCallRecord(
            name="web_search",
            args=args,
            ok=executed,
            started_at_ms=start_ms,
            ended_at_ms=now_ms(),
            result_preview={"count": len(snippets)},
            error=None if executed else "LLM not available for web search",
        )
    except Exception as exc:
        record = ToolCallRecord(
            name="web_search",
            args=args,
            ok=False,
            started_at_ms=start_ms,
            ended_at_ms=now_ms(),
            error=str(exc),
        )
    state.tool_calls.append(record)
    return state


def _format_snippets(snippets: List[dict], limit: int = 6) -> str:
    parts: List[str] = []
    for snip in snippets[:limit]:
        title = snip.get("title", "")
        url = snip.get("url", "")
        snippet = snip.get("snippet", "")
        score = snip.get("score")
        published_at = snip.get("published_at")
        meta = []
        if published_at:
            meta.append(f"published_at={published_at}")
        if score is not None:
            meta.append(f"score={score}")
        meta_text = f" ({', '.join(meta)})" if meta else ""
        parts.append(f"{title} {url} {snippet}{meta_text}".strip())
    return " ".join(parts)


def _domains_from_prefix(prefix: Optional[str]) -> Optional[List[str]]:
    if not prefix:
        return None
    parsed = urlparse(prefix if "://" in prefix else f"https://{prefix}")
    if parsed.netloc:
        return [parsed.netloc]
    return None

