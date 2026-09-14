from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from services.api.app.storage.models import IndexVersion
from services.worker.jobs.embed_and_upsert import _build_payload

from .index_lifecycle import (
    IndexLifecycleService,
    _capture_contracts,
    _catalog_fingerprint,
    _count_chunks,
    _embedding_spec,
    _now,
    _physical_name,
)
from .qdrant import normalize_qdrant_point_id
from .sparse import SparseEncoder


class SparseIndexLifecycleService(IndexLifecycleService):
    """IndexLifecycle extension for the controlled #57 dense+sparse candidate."""

    def __init__(
        self,
        repo: Any,
        vector_store: Any,
        embedding_router: Any,
        *,
        alias_name: Optional[str] = None,
        sparse_encoder: Optional[SparseEncoder] = None,
    ) -> None:
        super().__init__(repo, vector_store, embedding_router, alias_name=alias_name)
        self.sparse_encoder = sparse_encoder

    def sparse_contract_metadata(self) -> list[dict[str, Any]]:
        if self.sparse_encoder is None:
            return []
        return [
            {
                "contract_id": self.sparse_encoder.contract_id,
                **self.sparse_encoder.spec.as_public_dict(),
            }
        ]

    def create_candidate(
        self,
        embedding_contract_id_value: str,
        *,
        index_version_id: Optional[str] = None,
        physical_collection: Optional[str] = None,
        sparse_contract_id: Optional[str] = None,
    ) -> IndexVersion:
        if not sparse_contract_id:
            return super().create_candidate(
                embedding_contract_id_value,
                index_version_id=index_version_id,
                physical_collection=physical_collection,
            )
        encoder = self._require_sparse(sparse_contract_id)
        active = self.repo.get_active_index_version(self.alias_name)
        if active is None:
            raise ValueError("Dense+sparse candidate requires an active dense baseline IndexVersion")
        if str(active.embedding_contract_id) != str(embedding_contract_id_value):
            raise ValueError(
                "Phase-2 dense+sparse candidate must preserve the active dense embedding contract: "
                f"active={active.embedding_contract_id} requested={embedding_contract_id_value}"
            )
        embedder = self.embedding_router.get(embedding_contract_id_value)
        spec = _embedding_spec(embedder)
        dim = int(getattr(embedder, "dimension", 0) or 0)
        if dim <= 0:
            raise ValueError("Candidate embedding dimension must be positive")
        distance = str(spec.get("distance") or "cosine").lower()
        version_id = index_version_id or f"idx-{uuid.uuid4().hex[:16]}"
        physical = physical_collection or _physical_name(
            self.alias_name, embedding_contract_id_value, version_id
        )
        sparse_spec = encoder.spec.as_public_dict()
        dense_name = "dense"
        sparse_name = encoder.spec.vector_name
        version = IndexVersion(
            index_version_id=version_id,
            alias_name=self.alias_name,
            physical_collection=physical,
            embedding_contract_id=embedding_contract_id_value,
            embedding_spec=spec,
            vector_schema={
                "dense": {
                    "name": dense_name,
                    "dimension": dim,
                    "distance": distance,
                },
                "sparse": {
                    "name": sparse_name,
                    "contract_id": encoder.contract_id,
                    **sparse_spec,
                },
                "fusion": {"provider": "qdrant", "method": "rrf"},
            },
            status="building",
            created_at=_now(),
            build_stats={
                "experiment": "qdrant_dense_sparse",
                "control_dense_embedding_contract_id": active.embedding_contract_id,
                "baseline_index_version_id": active.index_version_id,
            },
        )
        self.repo.add_index_version(version)
        create_hybrid = getattr(self.vector_store, "create_hybrid_collection", None)
        if not callable(create_hybrid):
            self.repo.update_index_version(
                version_id,
                status="failed",
                failed_at=_now(),
                error="Vector store does not support named dense+sparse collections",
                delete_after=_now(),
            )
            raise TypeError("Vector store does not support create_hybrid_collection()")
        try:
            create_hybrid(
                physical,
                dim=dim,
                distance=distance,
                dense_name=dense_name,
                sparse_name=sparse_name,
                sparse_modifier=encoder.spec.modifier,
            )
        except Exception as exc:
            self.repo.update_index_version(
                version_id,
                status="failed",
                failed_at=_now(),
                error=str(exc),
                delete_after=_now(),
            )
            raise
        return self.repo.get_index_version(version_id) or version

    def build(
        self,
        index_version_id: str,
        *,
        batch_size: int = 100,
        progress=None,
    ) -> IndexVersion:
        version = self._require(index_version_id)
        sparse_schema = dict((version.vector_schema or {}).get("sparse") or {})
        if not sparse_schema:
            return super().build(
                index_version_id, batch_size=batch_size, progress=progress
            )
        if version.status not in {"building", "failed"}:
            raise ValueError(
                f"Index version is not buildable: {index_version_id} ({version.status})"
            )
        encoder = self._require_sparse(str(sparse_schema.get("contract_id") or ""))
        embedder = self.embedding_router.get(version.embedding_contract_id)
        dense_schema = dict((version.vector_schema or {}).get("dense") or {})
        dense_name = str(dense_schema.get("name") or "dense")
        sparse_name = str(sparse_schema.get("name") or encoder.spec.vector_name)
        dim = int(dense_schema.get("dimension") or 0)
        if int(embedder.dimension) != dim:
            raise RuntimeError("Candidate embedder dimension does not match IndexVersion schema")
        create_hybrid = getattr(self.vector_store, "create_hybrid_collection", None)
        upsert_hybrid = getattr(self.vector_store, "upsert_hybrid_to_collection", None)
        schema_getter = getattr(self.vector_store, "collection_vector_schema", None)
        if not callable(create_hybrid) or not callable(upsert_hybrid):
            raise TypeError("Vector store does not support dense+sparse IndexVersion builds")

        if version.status == "failed":
            self.vector_store.delete_collection(version.physical_collection)
            create_hybrid(
                version.physical_collection,
                dim=dim,
                distance=str(dense_schema.get("distance") or "cosine"),
                dense_name=dense_name,
                sparse_name=sparse_name,
                sparse_modifier=encoder.spec.modifier,
            )
        if callable(schema_getter):
            physical_schema = schema_getter(version.physical_collection)
            if dense_name not in (physical_schema.get("dense") or {}):
                raise RuntimeError("Physical collection is missing the named dense vector")
            if sparse_name not in (physical_schema.get("sparse") or {}):
                raise RuntimeError("Physical collection is missing the named sparse vector")

        catalog_fingerprint = _catalog_fingerprint(self.repo)
        base_stats = dict(version.build_stats or {})
        base_stats.update(
            {
                "state": "building",
                "vectors_written": 0,
                "catalog_fingerprint": catalog_fingerprint,
                "sparse_contract_id": encoder.contract_id,
            }
        )
        self.repo.update_index_version(
            index_version_id,
            status="building",
            build_stats=base_stats,
            error=None,
            failed_at=None,
        )
        total = _count_chunks(self.repo)
        written = 0
        started = time.perf_counter()
        batch: list[Any] = []
        parser_contracts: dict[str, dict[str, Any]] = {}
        chunking_contracts: dict[str, dict[str, Any]] = {}

        def flush() -> None:
            nonlocal batch, written
            if not batch:
                return
            texts = [chunk.text for chunk in batch]
            dense_vectors = embedder.embed_batch(texts)
            sparse_vectors = encoder.embed_documents(texts)
            if len(dense_vectors) != len(batch) or len(sparse_vectors) != len(batch):
                raise RuntimeError("Dense/sparse encoder result count does not match build batch")
            points = []
            for chunk, dense, sparse in zip(batch, dense_vectors, sparse_vectors):
                if len(dense) != dim:
                    raise RuntimeError(
                        f"Embedding dimension mismatch for {chunk.chunk_id}: {len(dense)}"
                    )
                payload = _build_payload(chunk, embedder.model_name)
                payload["embedding_contract_id"] = version.embedding_contract_id
                payload["embedding_dimension"] = dim
                payload["sparse_contract_id"] = encoder.contract_id
                payload["index_version_id"] = version.index_version_id
                point_id = normalize_qdrant_point_id(
                    chunk.qdrant_point_id, chunk.chunk_id
                )
                points.append((point_id, dense, sparse, payload))
                _capture_contracts(chunk, parser_contracts, chunking_contracts)
            upsert_hybrid(
                version.physical_collection,
                points,
                dense_name=dense_name,
                sparse_name=sparse_name,
            )
            written += len(points)
            elapsed = max(1e-9, time.perf_counter() - started)
            rate = written / elapsed
            remaining = max(0, total - written) if total is not None else None
            stats = {
                **base_stats,
                "state": "building",
                "vectors_written": written,
                "chunks_total": total,
                "vectors_per_second": round(rate, 3),
                "elapsed_seconds": round(elapsed, 3),
                "estimated_remaining_seconds": (
                    round(remaining / rate, 3) if remaining is not None and rate > 0 else None
                ),
            }
            self.repo.update_index_version(index_version_id, build_stats=stats)
            if progress:
                progress(dict(stats))
            batch = []

        try:
            for chunk in self.repo.iter_chunks():
                batch.append(chunk)
                if len(batch) >= max(1, int(batch_size)):
                    flush()
            flush()
            elapsed = max(1e-9, time.perf_counter() - started)
            if _catalog_fingerprint(self.repo) != catalog_fingerprint:
                raise RuntimeError(
                    "Knowledge catalog changed while the IndexVersion was building; rebuild against a stable active snapshot"
                )
            stats = {
                **base_stats,
                "state": "built",
                "vectors_written": written,
                "chunks_total": total,
                "vectors_per_second": round(written / elapsed, 3),
                "elapsed_seconds": round(elapsed, 3),
                "physical_count": int(
                    getattr(self.vector_store, "count_collection")(version.physical_collection)
                )
                if callable(getattr(self.vector_store, "count_collection", None))
                else written,
            }
            self.repo.update_index_version(
                index_version_id,
                status="validating",
                validating_at=_now(),
                build_stats=stats,
                parser_contracts=list(parser_contracts.values()),
                chunking_contracts=list(chunking_contracts.values()),
                error=None,
            )
            if progress:
                progress(dict(stats))
        except Exception as exc:
            self.repo.update_index_version(
                index_version_id,
                status="failed",
                failed_at=_now(),
                error=str(exc),
                build_stats={
                    **base_stats,
                    "state": "failed",
                    "vectors_written": written,
                    "catalog_fingerprint": catalog_fingerprint,
                },
            )
            raise
        return self._require(index_version_id)

    def _require_sparse(self, contract_id: str) -> SparseEncoder:
        if self.sparse_encoder is None:
            raise ValueError(
                "Sparse encoder is not configured; set RAGBOT_SPARSE_ENABLED=true"
            )
        if str(self.sparse_encoder.contract_id) != str(contract_id):
            raise KeyError(
                f"Unknown sparse contract {contract_id!r}; configured={self.sparse_encoder.contract_id}"
            )
        return self.sparse_encoder
