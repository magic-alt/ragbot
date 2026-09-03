from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .lexical import lexicalize
from ..storage.models import Chunk
from ..storage.protocol import Repo

# The in-memory development backend intentionally stays lightweight, but common
# question words should not outrank domain terms such as LoRA, QLoRA, GPU or
# quantization. PostgreSQL keeps using its native FTS implementation.
_ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "between",
    "by", "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "i", "if", "in", "into", "is", "it", "its", "of", "on",
    "or", "our", "should", "that", "the", "their", "them", "there", "these",
    "this", "those", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your",
}


def fts_search(repo: Repo, query: str, filters: Dict[str, Any], top_k: int) -> List[Tuple[Chunk, float]]:
    """Run lexical retrieval using the repository's native backend when available.

    Production PostgresRepo exposes ``search_chunks_fts`` and uses the GIN index
    created by migration 001. In-memory repositories use a deterministic scan
    implementation. The fallback removes high-frequency English question words
    and reuses Ragbot's CJK bigram lexicalizer so local testing is substantially
    less noisy without introducing an external search dependency.
    """
    native = getattr(repo, "search_chunks_fts", None)
    if callable(native):
        return native(query, filters, top_k)

    tokens = _tokenize(query)
    if not tokens:
        return []

    results: List[Tuple[Chunk, float]] = []
    for chunk in repo.iter_chunks():
        if not _match_filters(chunk, filters):
            continue
        score = _tf_score(tokens, chunk.text)
        if score > 0:
            results.append((chunk, score))
    results.sort(key=lambda item: item[1], reverse=True)
    return results[:top_k]


def _tokenize(text: str) -> List[str]:
    terms = lexicalize(text).split()
    filtered = [term for term in terms if term not in _ENGLISH_STOPWORDS]
    # If a query consists entirely of stopwords, retaining the original terms is
    # more useful than returning no lexical candidates at all.
    return filtered or terms


def _tf_score(tokens: List[str], text: str) -> float:
    doc_tokens = _tokenize(text)
    if not doc_tokens:
        return 0.0
    counts: Dict[str, int] = {}
    for tok in doc_tokens:
        counts[tok] = counts.get(tok, 0) + 1
    score = 0.0
    for tok in tokens:
        score += counts.get(tok, 0)
    return score / math.sqrt(len(doc_tokens))


def _match_filters(chunk: Chunk, filters: Dict[str, Any]) -> bool:
    if not filters:
        return True
    tenant_id = filters.get("tenant_id")
    if tenant_id and chunk.tenant_id != tenant_id:
        return False
    source_types = filters.get("source_types")
    if source_types:
        chunk_source_type = (chunk.metadata or {}).get("source_type")
        if chunk_source_type not in source_types:
            return False
    doc_ids = filters.get("doc_ids")
    if doc_ids and chunk.doc_id not in doc_ids:
        return False
    tags = filters.get("tags")
    if tags:
        chunk_tags = (chunk.metadata or {}).get("tags") or []
        if not any(tag in chunk_tags for tag in tags):
            return False
    path_prefix = filters.get("path_prefix")
    if path_prefix:
        path = chunk.path or ""
        if not path.startswith(path_prefix):
            return False
    url_prefix = filters.get("url_prefix")
    if url_prefix:
        url = chunk.url or ""
        if not url.startswith(url_prefix):
            return False
    time_range = filters.get("time_range")
    if time_range:
        start = _to_epoch(time_range.get("start"))
        end = _to_epoch(time_range.get("end"))
        raw_ts = None
        if chunk.metadata:
            raw_ts = chunk.metadata.get("ingested_at") or chunk.metadata.get("doc_updated_at")
        raw_ts = raw_ts or chunk.created_at
        timestamp = _to_epoch(raw_ts)
        if timestamp is not None:
            if start is not None and timestamp < start:
                return False
            if end is not None and timestamp > end:
                return False
    security_scope = filters.get("security_scope")
    if security_scope:
        allowed = set(security_scope)
        acl_hash = chunk.metadata.get("acl_hash") if chunk.metadata else None
        if acl_hash is None:
            return "public" in allowed
        if acl_hash not in allowed:
            return False
    return True


def _to_epoch(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None
