from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..storage.models import Chunk
from ..storage.repo import InMemoryRepo


class InvertedIndex:
    """Simple in-memory inverted index for FTS acceleration."""

    def __init__(self) -> None:
        self._index: Dict[str, set] = defaultdict(set)
        self._indexed_ids: set = set()

    def add(self, chunk_id: str, text: str) -> None:
        if chunk_id in self._indexed_ids:
            return
        self._indexed_ids.add(chunk_id)
        for tok in _tokenize(text):
            self._index[tok].add(chunk_id)

    def candidates(self, tokens: List[str]) -> set:
        if not tokens:
            return set()
        result: set = set()
        for tok in tokens:
            result |= self._index.get(tok, set())
        return result

    @property
    def size(self) -> int:
        return len(self._indexed_ids)


_global_index = InvertedIndex()


def fts_search(repo: InMemoryRepo, query: str, filters: Dict[str, Any], top_k: int) -> List[Tuple[Chunk, float]]:
    tokens = _tokenize(query)
    if not tokens:
        return []

    _ensure_indexed(repo)

    candidate_ids = _global_index.candidates(tokens)
    results: List[Tuple[Chunk, float]] = []
    for chunk_id in candidate_ids:
        chunk = repo.get_chunk(chunk_id)
        if not chunk:
            continue
        if not _match_filters(chunk, filters):
            continue
        score = _tf_score(tokens, chunk.text)
        if score > 0:
            results.append((chunk, score))
    results.sort(key=lambda item: item[1], reverse=True)
    return results[:top_k]


def _ensure_indexed(repo: InMemoryRepo) -> None:
    for chunk in repo.iter_chunks():
        _global_index.add(chunk.chunk_id, chunk.text)


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
        raw_ts = chunk.metadata.get("ingested_at") if chunk.metadata else None
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
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None
