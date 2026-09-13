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
    """Qdrant adapter with optional stable alias -> physical index indirection.

    `collection_name` remains the legacy physical collection for compatibility.
    When `alias_name` is provided, startup creates the alias if necessary and
    all steady-state reads/writes use the alias. New IndexVersion builds write
    directly to independent physical collections and activation atomically
    switches the alias.
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str],
        collection_name: str = "rag_chunks",
        dim: int = 1536,
        alias_name: Optional[str] = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as rest
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("qdrant-client is required for QdrantClientAdapter") from exc
        self._rest = rest
        self._client = QdrantClient(url=url, api_key=api_key)
        self._legacy_collection = collection_name
        self._alias = str(alias_name or "").strip() or None
        self._configured_dim = int(dim)

        if self._alias:
            target = self.alias_target(self._alias)
            if target:
                self._validate_collection_dimension(target, self._configured_dim)
                self._physical_collection = target
                self._ensure_payload_indexes_for(target)
            else:
                self._ensure_collection_named(collection_name, self._configured_dim)
                self._ensure_payload_indexes_for(collection_name)
                self._switch_alias_to(collection_name)
                self._physical_collection = collection_name
            self._collection = self._alias
        else:
            self._ensure_collection_named(collection_name, self._configured_dim)
            self._ensure_payload_indexes_for(collection_name)
            self._physical_collection = collection_name
            self._collection = collection_name

    @property
    def alias_name(self) -> Optional[str]:
        return self._alias

    @property
    def collection_name(self) -> str:
        """Logical query/write name (alias when lifecycle is enabled)."""
        return self._collection

    @property
    def dim(self) -> int:
        # An alias may be switched by another API/CLI process. Resolve the
        # query-visible physical collection so long-lived replicas immediately
        # observe the new vector schema instead of validating against stale dim.
        if self._alias:
            return self.collection_dimension(self.active_collection_name())
        return self._configured_dim

    def active_collection_name(self) -> str:
        if not self._alias:
            return self._physical_collection
        target = self.alias_target(self._alias)
        if not target:
            raise RuntimeError(f"Qdrant active alias is missing: {self._alias}")
        self._physical_collection = target
        return target

    def alias_target(self, alias_name: Optional[str] = None) -> Optional[str]:
        alias = str(alias_name or self._alias or "").strip()
        if not alias:
            return None
        response = self._client.get_aliases()
        for item in getattr(response, "aliases", None) or []:
            if str(getattr(item, "alias_name", "")) == alias:
                return str(getattr(item, "collection_name", "")) or None
        return None

    def collection_dimension(self, collection_name: str) -> int:
        info = self._client.get_collection(collection_name)
        vectors = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
        actual = getattr(vectors, "size", None)
        if actual is None and isinstance(vectors, dict):
            unnamed = vectors.get("") or next(iter(vectors.values()), None)
            actual = getattr(unnamed, "size", None)
        if actual is None:
            raise RuntimeError(f"Unable to resolve Qdrant vector dimension: {collection_name}")
        return int(actual)

    def create_physical_collection(
        self,
        collection_name: str,
        *,
        dim: int,
        distance: str = "cosine",
    ) -> None:
        if self._client.collection_exists(collection_name):
            self._validate_collection_dimension(collection_name, int(dim))
        else:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=self._rest.VectorParams(
                    size=int(dim), distance=_distance(distance, self._rest)
                ),
            )
        self._ensure_payload_indexes_for(collection_name)

    def switch_alias(self, collection_name: str) -> Optional[str]:
        if not self._alias:
            raise RuntimeError("Qdrant index activation requires alias_name configuration")
        if not self._client.collection_exists(collection_name):
            raise ValueError(f"Qdrant collection does not exist: {collection_name}")
        previous = self.alias_target(self._alias)
        if previous == collection_name:
            self._physical_collection = collection_name
            return previous
        self._switch_alias_to(collection_name)
        self._physical_collection = collection_name
        return previous

    def delete_collection(self, collection_name: str) -> bool:
        if self._alias and self.alias_target(self._alias) == collection_name:
            raise ValueError(f"Cannot delete active Qdrant collection: {collection_name}")
        if not self._client.collection_exists(collection_name):
            return False
        self._client.delete_collection(collection_name=collection_name)
        return True

    def upsert_to_collection(
        self,
        collection_name: str,
        points: Iterable[Tuple[str, List[float], Dict[str, Any]]],
    ) -> None:
        expected_dim = self.collection_dimension(collection_name)
        self._upsert_named(collection_name, points, expected_dim)

    def search_collection(
        self,
        collection_name: str,
        query_vector: List[float],
        filters: Dict[str, Any],
        top_k: int,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        expected_dim = self.collection_dimension(collection_name)
        if len(query_vector) != expected_dim:
            raise ValueError(
                f"Query vector dimension mismatch: got {len(query_vector)}, expected {expected_dim}"
            )
        return self._search_named(collection_name, query_vector, filters, top_k)

    def upsert(self, points: Iterable[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        self._upsert_named(self._collection, points, self.dim)

    def _upsert_named(
        self,
        collection_name: str,
        points: Iterable[Tuple[str, List[float], Dict[str, Any]]],
        expected_dim: int,
    ) -> None:
        payload_points = []
        for point_id, vector, payload in points:
            if len(vector) != expected_dim:
                raise ValueError(
                    f"Vector dimension mismatch: got {len(vector)}, expected {expected_dim}"
                )
            payload_points.append(self._rest.PointStruct(id=point_id, vector=vector, payload=payload))
        if payload_points:
            self._client.upsert(collection_name=collection_name, points=payload_points, wait=True)

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

    def count_collection(self, collection_name: str) -> int:
        return int(self._client.count(collection_name=collection_name, exact=True).count)

    def healthcheck(self) -> bool:
        try:
            return bool(self._client.collection_exists(self.active_collection_name()))
        except Exception:
            logger.exception("Qdrant healthcheck failed")
            return False

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def search(self, query_vector: List[float], filters: Dict[str, Any], top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        expected_dim = self.dim
        if len(query_vector) != expected_dim:
            raise ValueError(
                f"Query vector dimension mismatch: got {len(query_vector)}, expected {expected_dim}"
            )
        return self._search_named(self._collection, query_vector, filters, top_k)

    def _search_named(
        self,
        collection_name: str,
        query_vector: List[float],
        filters: Dict[str, Any],
        top_k: int,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        qfilter = _build_qdrant_filter(filters, self._rest)
        search = getattr(self._client, "search", None)
        if callable(search):
            results = search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                query_filter=qfilter,
            )
        else:
            response = self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                query_filter=qfilter,
            )
            results = response.points
        return [(str(hit.id), float(hit.score), hit.payload or {}) for hit in results]

    def _ensure_collection_named(self, collection_name: str, dim: int) -> None:
        if self._client.collection_exists(collection_name):
            self._validate_collection_dimension(collection_name, dim)
            return
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=self._rest.VectorParams(size=dim, distance=self._rest.Distance.COSINE),
        )

    def _validate_collection_dimension(self, collection_name: str, dim: int) -> None:
        actual_dim = self.collection_dimension(collection_name)
        if actual_dim != int(dim):
            raise RuntimeError(
                "Existing Qdrant collection dimension does not match configuration: "
                f"collection={collection_name}, actual={actual_dim}, configured={dim}. "
                "Build and activate a compatible IndexVersion after changing embedding contracts."
            )

    def _switch_alias_to(self, collection_name: str) -> None:
        rest = self._rest
        actions = []
        current = self.alias_target(self._alias)
        if current:
            actions.append(
                rest.DeleteAliasOperation(delete_alias=rest.DeleteAlias(alias_name=self._alias))
            )
        actions.append(
            rest.CreateAliasOperation(
                create_alias=rest.CreateAlias(
                    collection_name=collection_name,
                    alias_name=self._alias,
                )
            )
        )
        # Qdrant applies one update_collection_aliases request atomically.
        self._client.update_collection_aliases(change_aliases_operations=actions)

    def _ensure_payload_indexes_for(self, collection_name: str) -> None:
        rest = self._rest
        schemas = {
            "tenant_id": rest.PayloadSchemaType.KEYWORD,
            "source_type": rest.PayloadSchemaType.KEYWORD,
            "doc_id": rest.PayloadSchemaType.KEYWORD,
            "chunk_id": rest.PayloadSchemaType.KEYWORD,
            "source_id": rest.PayloadSchemaType.KEYWORD,
            "generation_id": rest.PayloadSchemaType.KEYWORD,
            "index_version_id": rest.PayloadSchemaType.KEYWORD,
            "embedding_contract_id": rest.PayloadSchemaType.KEYWORD,
            "acl_hash": rest.PayloadSchemaType.KEYWORD,
            "tags": rest.PayloadSchemaType.KEYWORD,
            "ingested_at_ts": rest.PayloadSchemaType.FLOAT,
            "doc_updated_at_ts": rest.PayloadSchemaType.FLOAT,
        }
        info = self._client.get_collection(collection_name)
        existing = set((getattr(info, "payload_schema", None) or {}).keys())
        for field_name, schema in schemas.items():
            if field_name in existing:
                continue
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )


def _distance(value: str, rest: Any) -> Any:
    normalized = str(value or "cosine").strip().lower()
    mapping = {
        "cosine": rest.Distance.COSINE,
        "dot": rest.Distance.DOT,
        "euclid": rest.Distance.EUCLID,
        "manhattan": rest.Distance.MANHATTAN,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported Qdrant distance: {value}") from exc


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
