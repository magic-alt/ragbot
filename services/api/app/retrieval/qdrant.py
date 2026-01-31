from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Tuple


class InMemoryQdrant:
    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self._points: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def upsert(self, points: Iterable[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        for point_id, vector, payload in points:
            self._points[point_id] = (vector, payload)

    def search(self, query_vector: List[float], filters: Dict[str, Any], top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for point_id, (vector, payload) in self._points.items():
            if not _match_filters(payload, filters):
                continue
            score = _cosine_similarity(query_vector, vector)
            results.append((point_id, score, payload))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]


def embed_text(text: str, dim: int = 64) -> List[float]:
    tokens = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
    vec = [0.0] * dim
    for tok in tokens:
        idx = (hash(tok) % dim + dim) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _match_filters(payload: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    if not filters:
        return True
    tenant_id = filters.get("tenant_id")
    if tenant_id and payload.get("tenant_id") != tenant_id:
        return False
    source_types = filters.get("source_types")
    if source_types and payload.get("source_type") not in source_types:
        return False
    doc_ids = filters.get("doc_ids")
    if doc_ids and payload.get("doc_id") not in doc_ids:
        return False
    path_prefix = filters.get("path_prefix")
    if path_prefix:
        path = payload.get("path") or ""
        if not path.startswith(path_prefix):
            return False
    url_prefix = filters.get("url_prefix")
    if url_prefix:
        url = payload.get("url") or ""
        if not url.startswith(url_prefix):
            return False
    time_range = filters.get("time_range")
    if time_range:
        start = time_range.get("start")
        end = time_range.get("end")
        timestamp = payload.get("ingested_at") or payload.get("doc_updated_at")
        if timestamp:
            if start and timestamp < start:
                return False
            if end and timestamp > end:
                return False
    security_scope = filters.get("security_scope")
    if security_scope:
        allowed = set(security_scope)
        acl_hash = payload.get("acl_hash")
        if acl_hash is None:
            return "public" in allowed
        if acl_hash not in allowed:
            return False
    return True

