from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from .qdrant import _build_qdrant_filter, _distance
from .sparse import SparseEncoder, SparseVector


class QdrantHybridAdapter:
    """Add named dense+sparse capabilities to the stable Qdrant adapter.

    The wrapped adapter keeps all alias, deletion, health and lifecycle behavior.
    This façade only owns representation-aware collection creation/upsert/search
    so sparse support cannot accidentally alter legacy unnamed dense collections.
    """

    def __init__(self, delegate: Any, sparse_encoder: Optional[SparseEncoder] = None) -> None:
        self._delegate = delegate
        self._sparse_encoder = sparse_encoder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    @property
    def sparse_encoder(self) -> Optional[SparseEncoder]:
        return self._sparse_encoder

    @property
    def sparse_contract_id(self) -> Optional[str]:
        if self._sparse_encoder is None:
            return None
        return self._sparse_encoder.contract_id

    @property
    def dim(self) -> int:
        return self.collection_dimension(self.active_collection_name())

    def collection_vector_schema(self, collection_name: str) -> dict[str, Any]:
        info = self._delegate._client.get_collection(collection_name)
        params = getattr(getattr(info, "config", None), "params", None)
        vectors = getattr(params, "vectors", None)
        sparse_vectors = getattr(params, "sparse_vectors", None)
        dense: dict[str, dict[str, Any]] = {}
        if isinstance(vectors, dict):
            for name, config in vectors.items():
                dense[str(name)] = {
                    "dimension": int(getattr(config, "size", 0) or 0),
                    "distance": _enum_value(getattr(config, "distance", None)),
                }
        else:
            dimension = int(getattr(vectors, "size", 0) or 0)
            if dimension:
                dense[""] = {
                    "dimension": dimension,
                    "distance": _enum_value(getattr(vectors, "distance", None)),
                }
        sparse: dict[str, dict[str, Any]] = {}
        if isinstance(sparse_vectors, dict):
            for name, config in sparse_vectors.items():
                sparse[str(name)] = {
                    "modifier": _enum_value(getattr(config, "modifier", None)),
                }
        return {"dense": dense, "sparse": sparse}

    def collection_dimension(self, collection_name: str) -> int:
        schema = self.collection_vector_schema(collection_name)
        dense = schema["dense"]
        if "dense" in dense:
            return int(dense["dense"]["dimension"])
        if "" in dense:
            return int(dense[""]["dimension"])
        if len(dense) == 1:
            return int(next(iter(dense.values()))["dimension"])
        raise RuntimeError(
            f"Unable to resolve primary dense vector for Qdrant collection: {collection_name}"
        )

    def create_hybrid_collection(
        self,
        collection_name: str,
        *,
        dim: int,
        distance: str,
        dense_name: str = "dense",
        sparse_name: str = "sparse",
        sparse_modifier: str = "idf",
    ) -> None:
        client = self._delegate._client
        rest = self._delegate._rest
        if client.collection_exists(collection_name):
            schema = self.collection_vector_schema(collection_name)
            expected_dense = schema["dense"].get(dense_name)
            if not expected_dense or int(expected_dense.get("dimension") or 0) != int(dim):
                raise RuntimeError(
                    f"Existing Qdrant collection does not match named dense schema: {collection_name}"
                )
            if sparse_name not in schema["sparse"]:
                raise RuntimeError(
                    f"Existing Qdrant collection is missing sparse vector {sparse_name!r}: {collection_name}"
                )
        else:
            modifier = _modifier(sparse_modifier, rest)
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    dense_name: rest.VectorParams(
                        size=int(dim), distance=_distance(distance, rest)
                    )
                },
                sparse_vectors_config={
                    sparse_name: rest.SparseVectorParams(modifier=modifier)
                },
            )
        self._delegate._ensure_payload_indexes_for(collection_name)

    def upsert_hybrid_to_collection(
        self,
        collection_name: str,
        points: Iterable[Tuple[str, List[float], SparseVector, Dict[str, Any]]],
        *,
        dense_name: str = "dense",
        sparse_name: str = "sparse",
    ) -> None:
        rest = self._delegate._rest
        expected_dim = self.collection_dimension(collection_name)
        payload_points = []
        for point_id, dense, sparse, payload in points:
            if len(dense) != expected_dim:
                raise ValueError(
                    f"Vector dimension mismatch: got {len(dense)}, expected {expected_dim}"
                )
            payload_points.append(
                rest.PointStruct(
                    id=point_id,
                    vector={
                        dense_name: list(dense),
                        sparse_name: rest.SparseVector(
                            indices=list(sparse.indices), values=list(sparse.values)
                        ),
                    },
                    payload=payload,
                )
            )
        if payload_points:
            self._delegate._client.upsert(
                collection_name=collection_name,
                points=payload_points,
                wait=True,
            )

    def upsert_to_collection(
        self,
        collection_name: str,
        points: Iterable[Tuple[str, List[float], Dict[str, Any]]],
    ) -> None:
        items = list(points)
        schema = self.collection_vector_schema(collection_name)
        if not schema["sparse"]:
            return self._delegate.upsert_to_collection(collection_name, items)
        self._upsert_dense_payload_points(collection_name, items, schema)

    def upsert(self, points: Iterable[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        items = list(points)
        physical = self.active_collection_name()
        schema = self.collection_vector_schema(physical)
        if not schema["sparse"]:
            return self._delegate.upsert(items)
        # Write through the alias so publication semantics remain unchanged.
        self._upsert_dense_payload_points(self.collection_name, items, schema)

    def _upsert_dense_payload_points(
        self,
        collection_name: str,
        items: list[Tuple[str, List[float], Dict[str, Any]]],
        schema: dict[str, Any],
    ) -> None:
        encoder = self._require_sparse_encoder()
        dense_name = _primary_dense_name(schema)
        sparse_name = _primary_sparse_name(schema)
        expected_dim = int(schema["dense"][dense_name]["dimension"])
        texts = [str((payload or {}).get("text") or "") for _id, _dense, payload in items]
        sparse_vectors = encoder.embed_documents(texts)
        if len(sparse_vectors) != len(items):
            raise RuntimeError("Sparse encoder result count does not match Qdrant upsert batch")
        hybrid = []
        for (point_id, dense, payload), sparse in zip(items, sparse_vectors):
            if len(dense) != expected_dim:
                raise ValueError(
                    f"Vector dimension mismatch: got {len(dense)}, expected {expected_dim}"
                )
            hybrid.append((point_id, dense, sparse, payload))
        self.upsert_hybrid_to_collection(
            collection_name,
            hybrid,
            dense_name=dense_name,
            sparse_name=sparse_name,
        )

    def search_collection(
        self,
        collection_name: str,
        query_vector: List[float],
        filters: Dict[str, Any],
        top_k: int,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        schema = self.collection_vector_schema(collection_name)
        if "" in schema["dense"]:
            return self._delegate.search_collection(
                collection_name, query_vector, filters, top_k
            )
        return self._named_dense_search(
            collection_name, query_vector, filters, top_k, _primary_dense_name(schema)
        )

    def search(
        self,
        query_vector: List[float],
        filters: Dict[str, Any],
        top_k: int,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        physical = self.active_collection_name()
        schema = self.collection_vector_schema(physical)
        if "" in schema["dense"]:
            return self._delegate.search(query_vector, filters, top_k)
        return self._named_dense_search(
            self.collection_name, query_vector, filters, top_k, _primary_dense_name(schema)
        )

    def _named_dense_search(
        self,
        collection_name: str,
        query_vector: List[float],
        filters: Dict[str, Any],
        top_k: int,
        dense_name: str,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        qfilter = _build_qdrant_filter(filters, self._delegate._rest)
        response = self._delegate._client.query_points(
            collection_name=collection_name,
            query=list(query_vector),
            using=dense_name,
            limit=int(top_k),
            with_payload=True,
            with_vectors=False,
            query_filter=qfilter,
        )
        return [
            (str(hit.id), float(hit.score), hit.payload or {})
            for hit in response.points
        ]

    def native_hybrid_search(
        self,
        dense_vector: List[float],
        sparse_vector: SparseVector,
        filters: Dict[str, Any],
        top_k: int,
        *,
        collection_name: Optional[str] = None,
        dense_name: str = "dense",
        sparse_name: str = "sparse",
        prefetch_limit: Optional[int] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        rest = self._delegate._rest
        target = collection_name or self.collection_name
        qfilter = _build_qdrant_filter(filters, rest)
        prefetch = max(int(top_k), int(prefetch_limit or max(top_k, top_k * 4)))
        response = self._delegate._client.query_points(
            collection_name=target,
            prefetch=[
                rest.Prefetch(
                    query=list(dense_vector),
                    using=dense_name,
                    limit=prefetch,
                ),
                rest.Prefetch(
                    query=rest.SparseVector(
                        indices=list(sparse_vector.indices),
                        values=list(sparse_vector.values),
                    ),
                    using=sparse_name,
                    limit=prefetch,
                ),
            ],
            query=rest.FusionQuery(fusion=rest.Fusion.RRF),
            limit=int(top_k),
            with_payload=True,
            with_vectors=False,
            query_filter=qfilter,
        )
        return [
            (str(hit.id), float(hit.score), hit.payload or {})
            for hit in response.points
        ]

    def _require_sparse_encoder(self) -> SparseEncoder:
        if self._sparse_encoder is None:
            raise RuntimeError(
                "Active Qdrant index requires sparse vectors but no sparse encoder is configured"
            )
        return self._sparse_encoder


def wrap_qdrant_hybrid(store: Any, sparse_encoder: Optional[SparseEncoder]) -> Any:
    if store.__class__.__name__ == "InMemoryQdrant":
        return store
    if isinstance(store, QdrantHybridAdapter):
        return store
    if not hasattr(store, "_client") or not hasattr(store, "_rest"):
        return store
    return QdrantHybridAdapter(store, sparse_encoder=sparse_encoder)


def _primary_dense_name(schema: dict[str, Any]) -> str:
    dense = schema.get("dense") or {}
    if "dense" in dense:
        return "dense"
    named = [name for name in dense if name]
    if len(named) == 1:
        return named[0]
    raise RuntimeError(f"Ambiguous named dense vector schema: {sorted(dense)}")


def _primary_sparse_name(schema: dict[str, Any]) -> str:
    sparse = schema.get("sparse") or {}
    if "sparse" in sparse:
        return "sparse"
    names = list(sparse)
    if len(names) == 1:
        return names[0]
    raise RuntimeError(f"Ambiguous sparse vector schema: {sorted(sparse)}")


def _modifier(value: str, rest: Any) -> Any:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "none"}:
        return None
    if normalized == "idf":
        return rest.Modifier.IDF
    raise ValueError(f"Unsupported Qdrant sparse modifier: {value}")


def _enum_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw).lower()
