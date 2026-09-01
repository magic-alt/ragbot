from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


class InMemoryQdrant:
    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self._points: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def upsert(self, points: Iterable[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        for point_id, vector, payload in points:
            if len(vector) != self._dim:
                raise ValueError(
                    f"Vector dimension mismatch: got {len(vector)}, expected {self._dim}"
                )
            self._points[point_id] = (vector, payload)

    def delete_points(self, point_ids: Iterable[str]) -> int:
        deleted = 0
        for point_id in set(point_ids):
            if self._points.pop(point_id, None) is not None:
                deleted += 1
        return deleted

    def delete_by_doc_ids(self, doc_ids: Iterable[str]) -> int:
        ids = set(doc_ids)
        if not ids:
            return 0
        to_delete = [
            point_id
            for point_id, (_vector, payload) in self._points.items()
            if payload.get("doc_id") in ids
        ]
        return self.delete_points(to_delete)

    def healthcheck(self) -> bool:
        return True

    def close(self) -> None:
        return None

    def search(self, query_vector: List[float], filters: Dict[str, Any], top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        if len(query_vector) != self._dim:
            raise ValueError(
                f"Query vector dimension mismatch: got {len(query_vector)}, expected {self._dim}"
            )
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
        payload_points = []
        for point_id, vector, payload in points:
            if len(vector) != self._dim:
                raise ValueError(
                    f"Vector dimension mismatch: got {len(vector)}, expected {self._dim}"
                )
            payload_points.append(rest.PointStruct(id=point_id, vector=vector, payload=payload))
        if not payload_points:
            return
        self._client.upsert(
            collection_name=self._collection,
            points=payload_points,
            wait=True,
        )

    def delete_points(self, point_ids: Iterable[str]) -> int:
        ids = list(dict.fromkeys(point_ids))
        if not ids:
            return 0
        self._client.delete(
            collection_name=self._collection,
            points_selector=self._rest.PointIdsList(points=ids),
            wait=True,
        )
        return len(ids)

    def delete_by_doc_ids(self, doc_ids: Iterable[str]) -> int:
        ids = list(dict.fromkeys(doc_ids))
        if not ids:
            return 0
        selector = self._rest.FilterSelector(
            filter=self._rest.Filter(
                must=[
                    self._rest.FieldCondition(
                        key="doc_id",
                        match=self._rest.MatchAny(any=ids),
                    )
                ]
            )
        )
        self._client.delete(
            collection_name=self._collection,
            points_selector=selector,
            wait=True,
        )
        return len(ids)

    def healthcheck(self) -> bool:
        return bool(self._client.collection_exists(self._collection))

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def search(self, query_vector: List[float], filters: Dict[str, Any], top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        if len(query_vector) != self._dim:
            raise ValueError(
                f"Query vector dimension mismatch: got {len(query_vector)}, expected {self._dim}"
            )
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
            info = self._client.get_collection(self._collection)
            vectors = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
            actual_dim = getattr(vectors, "size", None)
            if actual_dim is not None and int(actual_dim) != self._dim:
                raise RuntimeError(
                    "Existing Qdrant collection dimension does not match configuration: "
                    f"collection={self._collection}, actual={actual_dim}, configured={self._dim}. "
                    "Use a compatible collection or reindex after changing embedding dimensions."
                )
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=rest.VectorParams(size=self._dim, distance=rest.Distance.COSINE),
        )


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} != {len(b)}")
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
