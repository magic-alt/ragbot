from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple

from ..storage.models import Chunk
from ..storage.repo import InMemoryRepo


def fts_search(repo: InMemoryRepo, query: str, filters: Dict[str, Any], top_k: int) -> List[Tuple[Chunk, float]]:
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
    return re.findall(r"[A-Za-z0-9_\-]+", text.lower())


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
        start = time_range.get("start")
        end = time_range.get("end")
        timestamp = chunk.metadata.get("ingested_at") if chunk.metadata else None
        if timestamp:
            if start and timestamp < start:
                return False
            if end and timestamp > end:
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

