from __future__ import annotations

import hashlib
import math
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Optional

from services.api.app.storage.index_support import ensure_index_repository, supports_index_lifecycle
from services.api.app.storage.models import IndexVersion
from services.worker.jobs.embed_and_upsert import _build_payload

from .embedding_contract import embedding_contract_id
from .embedding_router import EmbeddingRouter
from .qdrant import normalize_qdrant_point_id

ProgressCallback = Callable[[dict[str, Any]], None]


class IndexLifecycleService:
    """Control versioned vector indexes behind one stable Qdrant alias.

    Qdrant's alias target is query-visible authority. PostgreSQL records build,
    validation, activation and retention history and is reconciled to the alias
    after partial failures.
    """

    def __init__(
        self,
        repo: Any,
        vector_store: Any,
        embedding_router: EmbeddingRouter,
        *,
        alias_name: Optional[str] = None,
    ) -> None:
        ensure_index_repository(repo)
        self.repo = repo
        self.vector_store = vector_store
        self.embedding_router = embedding_router
        self.alias_name = str(
            alias_name or getattr(vector_store, "alias_name", None) or ""
        ).strip()
        if not self.alias_name:
            raise ValueError("Index lifecycle requires a stable vector alias")
        required_vector_methods = (
            "active_collection_name",
            "create_physical_collection",
            "upsert_to_collection",
            "search_collection",
            "switch_alias",
            "delete_collection",
            "collection_dimension",
        )
        missing = [
            name for name in required_vector_methods if not callable(getattr(vector_store, name, None))
        ]
        if missing:
            raise TypeError(f"Vector store does not support index lifecycle: missing={missing}")
        if not supports_index_lifecycle(repo):
            raise TypeError("Repository does not support IndexLifecycleRepo")

    def bootstrap_current(self, default_embedder: Any) -> IndexVersion:
        """Register/reconcile the pre-lifecycle collection without reindexing."""
        physical = self.vector_store.active_collection_name()
        existing = self.repo.get_index_version_by_collection(self.alias_name, physical)
        active = self.repo.get_active_index_version(self.alias_name)
        if existing is not None:
            if active is None or active.index_version_id != existing.index_version_id:
                self.repo.activate_index_version(
                    existing.index_version_id,
                    previous_index_version_id=active.index_version_id if active else None,
                )
            return self.repo.get_index_version(existing.index_version_id) or existing

        if active is not None:
            raise RuntimeError(
                "Qdrant alias points to an unregistered physical collection while PostgreSQL already has an active IndexVersion: "
                f"alias={self.alias_name}, collection={physical}. Run index reconcile after restoring the expected version record."
            )

        now = _now()
        contract_id = embedding_contract_id(default_embedder)
        spec = _embedding_spec(default_embedder)
        version = IndexVersion(
            index_version_id=f"idx-legacy-{uuid.uuid4().hex[:12]}",
            alias_name=self.alias_name,
            physical_collection=physical,
            embedding_contract_id=contract_id,
            embedding_spec=spec,
            vector_schema={
                "dense": {
                    "dimension": int(getattr(default_embedder, "dimension", 0) or 0),
                    "distance": str(spec.get("distance") or "cosine"),
                }
            },
            status="active",
            created_at=now,
            ready_at=now,
            activated_at=now,
            build_stats={
                "state": "bootstrap",
                "catalog_fingerprint": _catalog_fingerprint(self.repo),
            },
            validation_evidence={"bootstrap": "legacy-existing-collection"},
        )
        self.repo.add_index_version(version)
        return self.repo.get_index_version(version.index_version_id) or version

    def create_candidate(
        self,
        embedding_contract_id_value: str,
        *,
        index_version_id: Optional[str] = None,
        physical_collection: Optional[str] = None,
    ) -> IndexVersion:
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
        version = IndexVersion(
            index_version_id=version_id,
            alias_name=self.alias_name,
            physical_collection=physical,
            embedding_contract_id=embedding_contract_id_value,
            embedding_spec=spec,
            vector_schema={"dense": {"dimension": dim, "distance": distance}},
            status="building",
            created_at=_now(),
        )
        self.repo.add_index_version(version)
        try:
            self.vector_store.create_physical_collection(
                physical, dim=dim, distance=distance
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
        progress: Optional[ProgressCallback] = None,
    ) -> IndexVersion:
        version = self._require(index_version_id)
        if version.status not in {"building", "failed"}:
            raise ValueError(
                f"Index version is not buildable: {index_version_id} ({version.status})"
            )
        embedder = self.embedding_router.get(version.embedding_contract_id)
        if int(embedder.dimension) != _index_dimension(version):
            raise RuntimeError("Candidate embedder dimension does not match IndexVersion schema")

        if version.status == "failed":
            self.vector_store.delete_collection(version.physical_collection)
            self.vector_store.create_physical_collection(
                version.physical_collection,
                dim=_index_dimension(version),
                distance=str((version.vector_schema.get("dense") or {}).get("distance") or "cosine"),
            )

        actual_dim = self.vector_store.collection_dimension(version.physical_collection)
        if actual_dim != _index_dimension(version):
            raise RuntimeError(
                f"Physical collection dimension mismatch: actual={actual_dim}, expected={_index_dimension(version)}"
            )

        catalog_fingerprint = _catalog_fingerprint(self.repo)
        self.repo.update_index_version(
            index_version_id,
            status="building",
            build_stats={
                "state": "building",
                "vectors_written": 0,
                "catalog_fingerprint": catalog_fingerprint,
            },
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
            vectors = embedder.embed_batch([chunk.text for chunk in batch])
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Embedder returned {len(vectors)} vectors for {len(batch)} chunks"
                )
            points = []
            for chunk, vector in zip(batch, vectors):
                if len(vector) != _index_dimension(version):
                    raise RuntimeError(
                        f"Embedding dimension mismatch for {chunk.chunk_id}: {len(vector)}"
                    )
                payload = _build_payload(chunk, embedder.model_name)
                payload["embedding_contract_id"] = version.embedding_contract_id
                payload["embedding_dimension"] = _index_dimension(version)
                payload["index_version_id"] = version.index_version_id
                point_id = normalize_qdrant_point_id(
                    chunk.qdrant_point_id, chunk.chunk_id
                )
                points.append((point_id, vector, payload))
                _capture_contracts(chunk, parser_contracts, chunking_contracts)
            self.vector_store.upsert_to_collection(version.physical_collection, points)
            written += len(points)
            elapsed = max(1e-9, time.perf_counter() - started)
            rate = written / elapsed
            remaining = max(0, total - written) if total is not None else None
            stats = {
                "state": "building",
                "vectors_written": written,
                "chunks_total": total,
                "catalog_fingerprint": catalog_fingerprint,
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
            final_fingerprint = _catalog_fingerprint(self.repo)
            if final_fingerprint != catalog_fingerprint:
                raise RuntimeError(
                    "Knowledge catalog changed while the IndexVersion was building; rebuild against a stable active snapshot"
                )
            stats = {
                "state": "built",
                "vectors_written": written,
                "chunks_total": total,
                "catalog_fingerprint": catalog_fingerprint,
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
                    "state": "failed",
                    "vectors_written": written,
                    "catalog_fingerprint": catalog_fingerprint,
                },
            )
            raise
        return self._require(index_version_id)

    def shadow_compare(
        self,
        index_version_id: str,
        cases: Iterable[dict[str, Any]],
        *,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        candidate = self._require(index_version_id)
        baseline = self.repo.get_active_index_version(self.alias_name)
        if baseline is None:
            raise RuntimeError("No active baseline index is registered")
        candidate_embedder = self.embedding_router.get(candidate.embedding_contract_id)
        baseline_embedder = self.embedding_router.get(baseline.embedding_contract_id)
        rows: list[dict[str, Any]] = []
        baseline_latencies: list[float] = []
        candidate_latencies: list[float] = []
        baseline_rr: list[float] = []
        candidate_rr: list[float] = []
        baseline_hits = 0
        candidate_hits = 0
        labeled = 0

        for raw_case in cases:
            query = str(raw_case.get("query") or "").strip()
            if not query:
                continue
            expected = {str(item) for item in raw_case.get("expected_chunk_ids") or []}
            query_filters = dict(filters or {})
            query_filters.update(raw_case.get("filters") or {})

            started = time.perf_counter()
            baseline_vec = _embed_query(baseline_embedder, query)
            baseline_results = self.vector_store.search_collection(
                baseline.physical_collection, baseline_vec, query_filters, top_k
            )
            baseline_ms = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            candidate_vec = _embed_query(candidate_embedder, query)
            candidate_results = self.vector_store.search_collection(
                candidate.physical_collection, candidate_vec, query_filters, top_k
            )
            candidate_ms = (time.perf_counter() - started) * 1000

            base_ids = [_logical_chunk_id(hit) for hit in baseline_results]
            cand_ids = [_logical_chunk_id(hit) for hit in candidate_results]
            baseline_latencies.append(baseline_ms)
            candidate_latencies.append(candidate_ms)
            row = {
                "query": query,
                "baseline_ids": base_ids,
                "candidate_ids": cand_ids,
                "overlap_at_k": round(
                    len(set(base_ids).intersection(cand_ids)) / max(1, top_k), 4
                ),
                "baseline_ms": round(baseline_ms, 3),
                "candidate_ms": round(candidate_ms, 3),
            }
            if expected:
                labeled += 1
                base_rank = _first_relevant_rank(base_ids, expected)
                cand_rank = _first_relevant_rank(cand_ids, expected)
                baseline_hits += int(base_rank is not None)
                candidate_hits += int(cand_rank is not None)
                baseline_rr.append(1.0 / base_rank if base_rank else 0.0)
                candidate_rr.append(1.0 / cand_rank if cand_rank else 0.0)
                row.update(
                    {
                        "baseline_relevant_rank": base_rank,
                        "candidate_relevant_rank": cand_rank,
                    }
                )
            rows.append(row)

        result = {
            "baseline_index_version_id": baseline.index_version_id,
            "candidate_index_version_id": candidate.index_version_id,
            "cases": len(rows),
            "labeled_cases": labeled,
            "top_k": top_k,
            "baseline_p95_ms": round(_percentile(baseline_latencies, 0.95), 3),
            "candidate_p95_ms": round(_percentile(candidate_latencies, 0.95), 3),
            "mean_overlap_at_k": round(
                sum(row["overlap_at_k"] for row in rows) / max(1, len(rows)), 4
            ),
            "rows": rows,
        }
        if labeled:
            result.update(
                {
                    "baseline_hit_at_k": round(baseline_hits / labeled, 4),
                    "candidate_hit_at_k": round(candidate_hits / labeled, 4),
                    "baseline_mrr": round(sum(baseline_rr) / labeled, 4),
                    "candidate_mrr": round(sum(candidate_rr) / labeled, 4),
                }
            )
        return result

    def mark_ready(
        self,
        index_version_id: str,
        evidence: dict[str, Any],
        *,
        approved: bool,
    ) -> IndexVersion:
        version = self._require(index_version_id)
        if version.status != "validating":
            raise ValueError(
                f"Index version is not validating: {index_version_id} ({version.status})"
            )
        if not evidence:
            raise ValueError("Validation evidence is required before promotion")
        if not approved:
            raise ValueError("Explicit validation approval is required before marking ready")
        self._assert_catalog_snapshot_current(version)
        self.repo.update_index_version(
            index_version_id,
            status="ready",
            ready_at=_now(),
            validation_evidence=dict(evidence),
            error=None,
        )
        return self._require(index_version_id)

    def activate(
        self,
        index_version_id: str,
        *,
        retention_seconds: int = 7 * 24 * 3600,
    ) -> IndexVersion:
        candidate = self._require(index_version_id)
        if candidate.status != "ready":
            raise ValueError(
                f"Index version must be ready before activation: {candidate.status}"
            )
        if not self.embedding_router.has(candidate.embedding_contract_id):
            raise RuntimeError(
                f"Candidate embedding contract is not loaded: {candidate.embedding_contract_id}"
            )
        self._assert_catalog_snapshot_current(candidate)
        baseline = self.repo.get_active_index_version(self.alias_name)
        previous_physical = self.vector_store.active_collection_name()
        if self.vector_store.collection_dimension(candidate.physical_collection) != _index_dimension(candidate):
            raise RuntimeError("Candidate physical collection schema does not match IndexVersion")

        self.vector_store.switch_alias(candidate.physical_collection)
        delete_after = (
            datetime.now(timezone.utc) + timedelta(seconds=max(0, retention_seconds))
        ).isoformat()
        try:
            self.repo.activate_index_version(
                index_version_id,
                previous_index_version_id=baseline.index_version_id if baseline else None,
                delete_after=delete_after,
            )
        except Exception:
            try:
                self.vector_store.switch_alias(previous_physical)
            except Exception:
                pass
            raise
        return self._require(index_version_id)

    def rollback(
        self,
        target_index_version_id: str,
        *,
        retention_seconds: int = 7 * 24 * 3600,
    ) -> IndexVersion:
        target = self._require(target_index_version_id)
        if target.status not in {"retired", "ready"}:
            raise ValueError(
                f"Rollback target must be retired/ready: {target.index_version_id} ({target.status})"
            )
        if not self.embedding_router.has(target.embedding_contract_id):
            raise RuntimeError(
                f"Rollback embedding contract is not loaded: {target.embedding_contract_id}"
            )
        self._assert_catalog_snapshot_current(target)
        current = self.repo.get_active_index_version(self.alias_name)
        previous_physical = self.vector_store.active_collection_name()
        self.vector_store.switch_alias(target.physical_collection)
        delete_after = (
            datetime.now(timezone.utc) + timedelta(seconds=max(0, retention_seconds))
        ).isoformat()
        try:
            self.repo.activate_index_version(
                target.index_version_id,
                previous_index_version_id=current.index_version_id if current else None,
                delete_after=delete_after,
            )
        except Exception:
            try:
                self.vector_store.switch_alias(previous_physical)
            except Exception:
                pass
            raise
        return self._require(target.index_version_id)

    def reconcile(self) -> dict[str, Any]:
        physical = self.vector_store.active_collection_name()
        visible = self.repo.get_index_version_by_collection(self.alias_name, physical)
        active = self.repo.get_active_index_version(self.alias_name)
        if visible is None:
            raise RuntimeError(
                "Qdrant alias points to an unregistered physical collection: "
                f"alias={self.alias_name}, collection={physical}"
            )
        if active is None or active.index_version_id != visible.index_version_id:
            self.repo.activate_index_version(
                visible.index_version_id,
                previous_index_version_id=active.index_version_id if active else None,
            )
            action = "postgres_pointer_repaired"
        else:
            action = "consistent"
        return {
            "alias_name": self.alias_name,
            "physical_collection": physical,
            "active_index_version_id": visible.index_version_id,
            "action": action,
        }

    def prune_retired(self, *, now_iso: Optional[str] = None) -> dict[str, Any]:
        now_iso = now_iso or _now()
        active_physical = self.vector_store.active_collection_name()
        deleted: list[str] = []
        skipped: list[str] = []
        for version in self.repo.list_prunable_index_versions(now_iso):
            if version.physical_collection == active_physical:
                skipped.append(version.index_version_id)
                continue
            self.vector_store.delete_collection(version.physical_collection)
            self.repo.update_index_version(
                version.index_version_id,
                status="deleted",
                deleted_at=_now(),
                error=None,
            )
            deleted.append(version.index_version_id)
        return {"deleted": deleted, "skipped_active": skipped}

    def _assert_catalog_snapshot_current(self, version: IndexVersion) -> None:
        expected = str((version.build_stats or {}).get("catalog_fingerprint") or "")
        if not expected:
            if version.status == "active":
                return
            raise RuntimeError(
                f"IndexVersion has no catalog snapshot fingerprint: {version.index_version_id}"
            )
        actual = _catalog_fingerprint(self.repo)
        if actual != expected:
            raise RuntimeError(
                "Knowledge catalog changed since this IndexVersion was built; rebuild before activation/rollback: "
                f"index={version.index_version_id}, built={expected}, current={actual}"
            )

    def _require(self, index_version_id: str) -> IndexVersion:
        version = self.repo.get_index_version(index_version_id)
        if version is None:
            raise ValueError(f"Unknown index version: {index_version_id}")
        if version.alias_name != self.alias_name:
            raise ValueError(
                f"Index version belongs to another alias: {version.alias_name} != {self.alias_name}"
            )
        return version


def _embedding_spec(embedder: Any) -> dict[str, Any]:
    spec = getattr(embedder, "spec", None)
    as_public = getattr(spec, "as_public_dict", None)
    if callable(as_public):
        return dict(as_public())
    return {
        "provider_id": type(embedder).__name__.lower(),
        "model": str(getattr(embedder, "model_name", "unknown")),
        "dimension": int(getattr(embedder, "dimension", 0) or 0),
        "distance": "cosine",
    }


def _physical_name(alias_name: str, contract_id: str, version_id: str) -> str:
    base = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in alias_name)
    return f"{base}__{contract_id.replace('-', '_')[:16]}__{version_id[-8:]}"[:200]


def _index_dimension(version: IndexVersion) -> int:
    dense = version.vector_schema.get("dense") or {}
    return int(dense.get("dimension") or version.embedding_spec.get("dimension") or 0)


def _embed_query(embedder: Any, query: str) -> list[float]:
    method = getattr(embedder, "embed_query", None)
    return method(query) if callable(method) else embedder.embed(query)


def _logical_chunk_id(hit: tuple[str, float, dict[str, Any]]) -> str:
    point_id, _score, payload = hit
    return str(payload.get("chunk_id") or point_id)


def _first_relevant_rank(ids: list[str], expected: set[str]) -> Optional[int]:
    for rank, chunk_id in enumerate(ids, 1):
        if chunk_id in expected:
            return rank
    return None


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def _capture_contracts(
    chunk: Any,
    parsers: dict[str, dict[str, Any]],
    chunkers: dict[str, dict[str, Any]],
) -> None:
    metadata = chunk.metadata or {}
    parser_hash = str(metadata.get("parser_config_hash") or "")
    if parser_hash and parser_hash not in parsers:
        parsers[parser_hash] = {
            "provider": metadata.get("parser_provider"),
            "strategy": metadata.get("parser_strategy"),
            "version": metadata.get("parser_version"),
            "config_hash": parser_hash,
        }
    chunker_hash = str(metadata.get("chunker_config_hash") or "")
    if chunker_hash and chunker_hash not in chunkers:
        chunkers[chunker_hash] = {
            "provider": metadata.get("chunker_provider"),
            "strategy": metadata.get("chunker_strategy"),
            "version": metadata.get("chunker_version"),
            "config_hash": chunker_hash,
            "chunk_size": metadata.get("chunk_size"),
            "chunk_overlap": metadata.get("chunk_overlap"),
        }


def _catalog_fingerprint(repo: Any) -> str:
    rows: list[tuple[str, str, str]] = []
    if hasattr(repo, "_pool"):
        with repo._pool.connection() as conn:
            db_rows = conn.execute(
                "SELECT chunk_id, qdrant_point_id, checksum FROM chunks ORDER BY chunk_id"
            ).fetchall()
        for row in db_rows:
            if isinstance(row, dict):
                rows.append(
                    (
                        str(row.get("chunk_id") or ""),
                        str(row.get("qdrant_point_id") or ""),
                        str(row.get("checksum") or ""),
                    )
                )
            else:
                rows.append(tuple(str(item or "") for item in row[:3]))
    else:
        rows = sorted(
            (
                str(chunk.chunk_id),
                str(chunk.qdrant_point_id or ""),
                str(chunk.checksum or ""),
            )
            for chunk in repo.iter_chunks()
        )
    digest = hashlib.sha256()
    for row in rows:
        digest.update("\x1f".join(row).encode("utf-8"))
        digest.update(b"\n")
    return f"cat-{digest.hexdigest()[:24]}"


def _count_chunks(repo: Any) -> Optional[int]:
    if hasattr(repo, "_pool"):
        try:
            with repo._pool.connection() as conn:
                row = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
            if isinstance(row, dict):
                return int(row.get("n") or 0)
            return int(row[0])
        except Exception:
            return None
    if hasattr(repo, "_chunks"):
        return len(repo._chunks)
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
