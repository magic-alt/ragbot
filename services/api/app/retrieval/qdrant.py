from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Stable forever: changing this namespace would orphan every existing vector.
_QDRANT_POINT_NAMESPACE = uuid.UUID("b7a573f1-d7ec-4d1d-9797-f067b5d42c7d")


def point_id_for_chunk(chunk_id: str) -> str:
    """Return a deterministic Qdrant-compatible UUID for a logical chunk id."""
    return str(uuid.uuid5(_QDRANT_POINT_NAMESPACE, str(chunk_id)))


def normalize_qdrant_point_id(point_id: Optional[str], chunk_id: str) -> str:
    """Normalize a persisted point id or derive the canonical deterministic UUID.

    Historical Ragbot versions persisted arbitrary logical chunk IDs as
    ``qdrant_point_id``. Real Qdrant accepts only unsigned integers or UUIDs, so
    arbitrary legacy strings must not be reused for deletes/re-ingestion.
    """
    if point_id is not None:
        raw = str(point_id).strip()
        if raw.isdigit():
            return raw
        try:
            return str(uuid.UUID(raw))
        except (ValueError, AttributeError):
            pass
    return point_id_for_chunk(chunk_id)


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
                raise ValueError(f"Vector dimension mismatch: got {len(vector)}, expected {self._dim}")
            self._points[str(point_id)] = (vector, payload)

    def delete_points(self, point_ids: Iterable[str]) -> int:
        deleted = 0
        for point_id in set(str(item) for item in point_ids):
            if self._points.pop(point_id, None) is not None:
                deleted += 1
        return deleted

    def delete_by_doc_ids(self, doc_ids: Iterable[str]) -> int:
        ids = set(doc_ids)
        if not ids:
            return 0
        return self.delete_points(
            point_id
            for point_id, (_vector, payload) in self._points.items()
            if payload.get("doc_id") in ids
        )

    def count(self) -> int:
        return len(self._points)

    def healthcheck(self) -> bool:
        return True

    def close(self) -> None:
        return None

    def search(self, query_vector: List[float], filters: Dict[str, Any], top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        if len(query_vector) != self._dim:
            raise ValueError(f"Query vector dimension mismatch: got {len(query_vector)}, expected {self._dim}")
        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for point_id, (vector, payload) in self._points.items():
            if _match_filters(payload, filters):
                results.append((point_id, _cosine_similarity(query_vector, vector), payload))
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
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("qdrant-client is required for QdrantClientAdapter") from exc
        self._rest = rest
        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection = collection_name
        self._dim = dim
        self._ensure_collection()
        self._ensure_payload_indexes()

    @property
    def dim(self) -> int:
        return self._dim

    def upsert(self, points: Iterable[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        payload_points = []
        for point_id, vector, payload in points:
            if len(vector) != self._dim:
                raise ValueError(f"Vector dimension mismatch: got {len(vector)}, expected {self._dim}")
            payload_points.append(self._rest.PointStruct(id=point_id, vector=vector, payload=payload))
        if payload_points:
            self._client.upsert(collection_name=self._collection, points=payload_points, wait=True)

    def delete_points(self, point_ids: Iterable[str]) -> int:
        ids = list(dict.fromkeys(str(item) for item in point_ids))
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
                must=[self._rest.FieldCondition(key="doc_id", match=self._rest.MatchAny(any=ids))]
            )
        )
        self._client.delete(collection_name=self._collection, points_selector=selector, wait=True)
        return len(ids)

    def count(self) -> int:
        return int(self._client.count(collection_name=self._collection, exact=True).count)

    def healthcheck(self) -> bool:
        return bool(self._client.collection_exists(self._collection))

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def search(self, query_vector: List[float], filters: Dict[str, Any], top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        if len(query_vector) != self._dim:
            raise ValueError(f"Query vector dimension mismatch: got {len(query_vector)}, expected {self._dim}")
        qfilter = _build_qdrant_filter(filters, self._rest)
        search = getattr(self._client, "search", None)
        if callable(search):
            results = search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                query_filter=qfilter,
            )
        else:
            response = self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                query_filter=qfilter,
            )
            results = response.points
        return [(str(hit.id), float(hit.score), hit.payload or {}) for hit in results]

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

    def _ensure_payload_indexes(self) -> None:
        rest = self._rest
        schemas = {
            "tenant_id": rest.PayloadSchemaType.KEYWORD,
            "source_type": rest.PayloadSchemaType.KEYWORD,
            "doc_id": rest.PayloadSchemaType.KEYWORD,
            "chunk_id": rest.PayloadSchemaType.KEYWORD,
            "acl_hash": rest.PayloadSchemaType.KEYWORD,
            "tags": rest.PayloadSchemaType.KEYWORD,
            "ingested_at_ts": rest.PayloadSchemaType.FLOAT,
            "doc_updated_at_ts": rest.PayloadSchemaType.FLOAT,
        }
        info = self._client.get_collection(self._collection)
        existing = set((getattr(info, "payload_schema", None) or {}).keys())
        for field_name, schema in schemas.items():
            if field_name in existing:
                continue
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field_name,
                field_schema=schema,
                wait=True,
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
    if tags and not any(tag in (payload.get("tags") or []) for tag in tags):
        return False
    path_prefix = filters.get("path_prefix")
    if path_prefix and not (payload.get("path") or "").startswith(path_prefix):
        return False
    url_prefix = filters.get("url_prefix")
    if url_prefix and not (payload.get("url") or "").startswith(url_prefix):
        return False
    time_range = filters.get("time_range")
    if time_range:
        start = to_epoch(time_range.get("start"))
        end = to_epoch(time_range.get("end"))
        timestamp = to_epoch(
            payload.get("ingested_at_ts") or payload.get("doc_updated_at_ts")
            or payload.get("ingested_at") or payload.get("doc_updated_at")
        )
        if timestamp is not None:
            if start is not None and timestamp < start:
                return False
            if end is not None and timestamp > end:
                return False
    security_scope = filters.get("security_scope")
    if security_scope:
        acl_hash = payload.get("acl_hash")
        if acl_hash is None:
            return "public" in set(security_scope)
        if acl_hash not in set(security_scope):
            return False
    return True


def _build_qdrant_filter(filters: Dict[str, Any], rest: Any) -> Optional[Any]:
    if not filters:
        return None
    must = []
    should = []
    if filters.get("tenant_id"):
        must.append(rest.FieldCondition(key="tenant_id", match=rest.MatchValue(value=filters["tenant_id"])))
    if filters.get("source_types"):
        must.append(rest.FieldCondition(key="source_type", match=rest.MatchAny(any=filters["source_types"])))
    if filters.get("doc_ids"):
        must.append(rest.FieldCondition(key="doc_id", match=rest.MatchAny(any=filters["doc_ids"])))
    if filters.get("tags"):
        must.append(rest.FieldCondition(key="tags", match=rest.MatchAny(any=filters["tags"])))
    if filters.get("path_prefix"):
        must.append(rest.FieldCondition(key="path", match=rest.MatchText(text=filters["path_prefix"])))
    if filters.get("url_prefix"):
        must.append(rest.FieldCondition(key="url", match=rest.MatchText(text=filters["url_prefix"])))
    time_range = filters.get("time_range")
    if time_range:
        start = to_epoch(time_range.get("start"))
        end = to_epoch(time_range.get("end"))
        if start is not None or end is not None:
            range_clause = rest.Range(gte=start, lte=end)
            should.append(rest.FieldCondition(key="ingested_at_ts", range=range_clause))
            should.append(rest.FieldCondition(key="doc_updated_at_ts", range=range_clause))
    if filters.get("security_scope"):
        must.append(rest.FieldCondition(key="acl_hash", match=rest.MatchAny(any=filters["security_scope"])))
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
