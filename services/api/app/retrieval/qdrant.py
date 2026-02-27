from __future__ import annotations

import logging
import math
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str, int], List[float]]

_embed_fn: Optional[EmbedFn] = None


def get_embed_fn() -> EmbedFn:
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = (os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/")
    model = os.getenv("EMBEDDING_MODEL", "")
    if api_key and model:
        logger.info("Using real embedding API: model=%s", model)
        _embed_fn = _make_api_embed_fn(api_key, base_url, model)
    else:
        logger.info("Using hash-based embedding (set EMBEDDING_MODEL + EMBEDDING_API_KEY for real embeddings)")
        _embed_fn = _hash_embed
    return _embed_fn


def _make_api_embed_fn(api_key: str, base_url: str, model: str) -> EmbedFn:
    def _api_embed(text: str, dim: int = 1536) -> List[float]:
        url = f"{base_url}/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": model, "input": text}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            vector = data["data"][0]["embedding"]
            if len(vector) != dim:
                vector = vector[:dim] if len(vector) > dim else vector + [0.0] * (dim - len(vector))
            return vector
        except Exception as exc:
            logger.warning("Embedding API failed, falling back to hash: %s", exc)
            return _hash_embed(text, dim)
    return _api_embed


def _hash_embed(text: str, dim: int = 64) -> List[float]:
    tokens = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
    vec = [0.0] * dim
    for tok in tokens:
        idx = (hash(tok) % dim + dim) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed_text(text: str, dim: int = 64) -> List[float]:
    fn = get_embed_fn()
    return fn(text, dim)


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


class QdrantClientAdapter:
    def __init__(
        self,
        url: str,
        api_key: Optional[str],
        collection_name: str = "rag_chunks",
        dim: int = 1536,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as rest
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("qdrant-client is required for QdrantClientAdapter") from exc

        self._rest = rest
        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection = collection_name
        self._dim = dim
        self._ensure_collection()

    @property
    def dim(self) -> int:
        return self._dim

    def upsert(self, points: Iterable[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        rest = self._rest
        payload_points = [
            rest.PointStruct(id=point_id, vector=vector, payload=payload)
            for point_id, vector, payload in points
        ]
        if not payload_points:
            return
        self._client.upsert(collection_name=self._collection, points=payload_points)

    def search(self, query_vector: List[float], filters: Dict[str, Any], top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        qfilter = _build_qdrant_filter(filters, self._rest)
        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=qfilter,
        )
        return [(str(hit.id), hit.score, hit.payload or {}) for hit in results]

    def _ensure_collection(self) -> None:
        rest = self._rest
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=rest.VectorParams(size=self._dim, distance=rest.Distance.COSINE),
        )


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
    tags = filters.get("tags")
    if tags:
        payload_tags = payload.get("tags") or []
        if not any(tag in payload_tags for tag in tags):
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
        start = to_epoch(time_range.get("start"))
        end = to_epoch(time_range.get("end"))
        timestamp = to_epoch(
            payload.get("ingested_at_ts")
            or payload.get("doc_updated_at_ts")
            or payload.get("ingested_at")
            or payload.get("doc_updated_at")
        )
        if timestamp is not None:
            if start is not None and timestamp < start:
                return False
            if end is not None and timestamp > end:
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


def _build_qdrant_filter(filters: Dict[str, Any], rest: Any) -> Optional[Any]:
    if not filters:
        return None
    must = []
    should = []
    tenant_id = filters.get("tenant_id")
    if tenant_id:
        must.append(rest.FieldCondition(key="tenant_id", match=rest.MatchValue(value=tenant_id)))
    source_types = filters.get("source_types")
    if source_types:
        must.append(rest.FieldCondition(key="source_type", match=rest.MatchAny(any=source_types)))
    doc_ids = filters.get("doc_ids")
    if doc_ids:
        must.append(rest.FieldCondition(key="doc_id", match=rest.MatchAny(any=doc_ids)))
    tags = filters.get("tags")
    if tags:
        must.append(rest.FieldCondition(key="tags", match=rest.MatchAny(any=tags)))
    path_prefix = filters.get("path_prefix")
    if path_prefix:
        must.append(rest.FieldCondition(key="path", match=rest.MatchText(text=path_prefix)))
    url_prefix = filters.get("url_prefix")
    if url_prefix:
        must.append(rest.FieldCondition(key="url", match=rest.MatchText(text=url_prefix)))
    time_range = filters.get("time_range")
    if time_range:
        start = to_epoch(time_range.get("start"))
        end = to_epoch(time_range.get("end"))
        if start is not None or end is not None:
            range_clause = rest.Range(gte=start, lte=end)
            should.append(rest.FieldCondition(key="ingested_at_ts", range=range_clause))
            should.append(rest.FieldCondition(key="doc_updated_at_ts", range=range_clause))
    security_scope = filters.get("security_scope")
    if security_scope:
        must.append(rest.FieldCondition(key="acl_hash", match=rest.MatchAny(any=security_scope)))
    if not must and not should:
        return None
    return rest.Filter(must=must or None, should=should or None)


def to_epoch(value: Any) -> Optional[float]:
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

