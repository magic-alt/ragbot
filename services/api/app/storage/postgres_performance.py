from __future__ import annotations

import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from ..retrieval.lexical import contains_cjk, lexicalize
from .models import Chunk, Document, IngestionJob
from .query_support import ensure_query_repository


def ensure_database_performance(repo: Any) -> Any:
    """Install Ragbot's bounded PostgreSQL production hot paths.

    The production worker/API composition calls this after staged-generation
    support is attached and before the worker publication claim gate is wrapped.
    In-memory repositories only receive the bounded query helpers.
    """

    ensure_query_repository(repo)
    if not hasattr(repo, "_pool"):
        return repo
    if getattr(repo, "_ragbot_postgres_performance_installed", False):
        return repo

    setattr(repo, "_ragbot_perf_lock", threading.Lock())
    setattr(
        repo,
        "_ragbot_perf_metrics",
        {
            "claim_attempts": 0,
            "claim_hits": 0,
            "claim_empty": 0,
            "claim_total_ms": 0.0,
            "claim_max_ms": 0.0,
            "copy_chunk_batches": 0,
            "copy_chunk_rows": 0,
            "copy_generation_batches": 0,
            "copy_generation_rows": 0,
        },
    )

    # Migration 010 made source/generation ownership first-class on documents,
    # but the historical pg_repo adapter predates those columns. Keep the
    # production path schema-aligned here until the compatibility façade can be
    # narrowed further.
    _replace(repo, "add_document", _PostgresPerformanceMixin.add_document)
    _replace(
        repo,
        "delete_documents_by_source",
        _PostgresPerformanceMixin.delete_documents_by_source,
    )
    _replace(repo, "claim_next_job", _PostgresPerformanceMixin.claim_next_job)
    _replace(repo, "add_chunks", _PostgresPerformanceMixin.add_chunks)
    if callable(getattr(repo, "stage_knowledge_generation", None)):
        _replace(
            repo,
            "stage_knowledge_generation",
            _PostgresPerformanceMixin.stage_knowledge_generation,
        )
    setattr(
        repo,
        "database_runtime_metrics",
        _PostgresPerformanceMixin.database_runtime_metrics.__get__(repo, type(repo)),
    )
    setattr(repo, "_ragbot_postgres_performance_installed", True)
    return repo


def _replace(repo: Any, name: str, method: Any) -> None:
    legacy_name = f"_ragbot_legacy_{name}"
    if not hasattr(repo, legacy_name):
        setattr(repo, legacy_name, getattr(repo, name))
    setattr(repo, name, method.__get__(repo, type(repo)))


class _PostgresPerformanceMixin:
    def add_document(self, doc: Document) -> None:
        sql = """
            INSERT INTO documents (
                doc_id, tenant_id, source_type, title, uri, version,
                doc_updated_at, ingested_at, tags, acl_policy_id, status,
                source_id, generation_id
            ) VALUES (
                %(doc_id)s, %(tenant_id)s, %(source_type)s, %(title)s, %(uri)s,
                %(version)s, %(doc_updated_at)s, %(ingested_at)s,
                %(tags)s, %(acl_policy_id)s, %(status)s,
                %(source_id)s, %(generation_id)s
            )
            ON CONFLICT (doc_id) DO UPDATE SET
                tenant_id = EXCLUDED.tenant_id,
                source_type = EXCLUDED.source_type,
                title = EXCLUDED.title,
                uri = EXCLUDED.uri,
                version = EXCLUDED.version,
                doc_updated_at = EXCLUDED.doc_updated_at,
                ingested_at = EXCLUDED.ingested_at,
                tags = EXCLUDED.tags,
                acl_policy_id = EXCLUDED.acl_policy_id,
                status = EXCLUDED.status,
                source_id = EXCLUDED.source_id,
                generation_id = EXCLUDED.generation_id
        """
        params = asdict(doc)
        params["tags"] = list(params["tags"])
        with self._pool.connection() as conn:
            conn.execute(sql, params)

    def delete_documents_by_source(self, source_id: str) -> list[str]:
        pattern = f"source://{source_id}%"
        local_fs_pattern = f"doc-{source_id}:%"
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                DELETE FROM documents
                WHERE source_id = %s OR (source_id IS NULL AND (uri LIKE %s OR doc_id LIKE %s))
                RETURNING doc_id
                """,
                (source_id, pattern, local_fs_pattern),
            ).fetchall()
        return [str(row["doc_id"]) for row in rows]

    def claim_next_job(
        self,
        worker_id: str,
        lease_seconds: int = 120,
        max_attempts: int = 3,
    ) -> Optional[IngestionJob]:
        """Claim exactly one ready job without queue-wide reconciliation writes.

        Expired leases / exhausted attempts are repaired by the worker's
        periodic reconciliation pass. Keeping that repair out of every claim is
        important when many workers poll an empty or lightly loaded queue.
        """

        started = time.perf_counter()
        row = None
        try:
            with self._pool.connection() as conn:
                with conn.transaction():
                    row = conn.execute(
                        """
                        WITH candidate AS (
                            SELECT job_id
                            FROM ingestion_jobs
                            WHERE status = 'pending'
                              AND available_at <= NOW()
                              AND attempts < %s
                            ORDER BY available_at, created_at, job_id
                            FOR UPDATE SKIP LOCKED
                            LIMIT 1
                        )
                        UPDATE ingestion_jobs AS j
                        SET status = 'running',
                            attempts = j.attempts + 1,
                            started_at = COALESCE(j.started_at, NOW()),
                            lease_owner = %s,
                            heartbeat_at = NOW(),
                            lease_expires_at = NOW() + (%s * INTERVAL '1 second'),
                            error = NULL,
                            failure_class = NULL,
                            dead_lettered_at = NULL
                        FROM candidate
                        WHERE j.job_id = candidate.job_id
                        RETURNING j.*
                        """,
                        (max(1, int(max_attempts)), worker_id, max(1, int(lease_seconds))),
                    ).fetchone()
            return self._row_to_job(row, None) if row else None
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            _observe(
                self,
                "claim_attempts",
                1,
                elapsed_ms=elapsed_ms,
                hit=row is not None,
            )

    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        items = list(chunks)
        if not items:
            return 0
        threshold = max(1, int(getattr(self, "_ragbot_pg_copy_min_rows", 256)))
        if len(items) < threshold:
            return int(self._ragbot_legacy_add_chunks(items))

        now = datetime.now(timezone.utc).isoformat()
        columns = (
            "chunk_id", "doc_id", "tenant_id", "chunk_index", "text",
            "path", "url", "page", "section", "checksum", "qdrant_point_id",
            "created_at", "metadata", "fts_text", "source_id", "generation_id",
        )
        with self._pool.connection() as conn:
            with conn.transaction():
                conn.execute(
                    "CREATE TEMP TABLE ragbot_chunk_batch "
                    "(LIKE chunks INCLUDING DEFAULTS) ON COMMIT DROP"
                )
                with conn.cursor() as cur:
                    with cur.copy(
                        "COPY ragbot_chunk_batch (" + ", ".join(columns) + ") FROM STDIN"
                    ) as copy:
                        for chunk in items:
                            copy.write_row(
                                (
                                    chunk.chunk_id,
                                    chunk.doc_id,
                                    chunk.tenant_id,
                                    chunk.chunk_index,
                                    chunk.text,
                                    chunk.path,
                                    chunk.url,
                                    chunk.page,
                                    chunk.section,
                                    chunk.checksum,
                                    chunk.qdrant_point_id,
                                    chunk.created_at or now,
                                    self._jsonb(chunk.metadata or {}),
                                    lexicalize(chunk.text) if contains_cjk(chunk.text) else chunk.text,
                                    chunk.source_id,
                                    chunk.generation_id,
                                )
                            )
                conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, doc_id, tenant_id, chunk_index, text,
                        path, url, page, section, checksum, qdrant_point_id,
                        created_at, metadata, fts_text, source_id, generation_id
                    )
                    SELECT
                        chunk_id, doc_id, tenant_id, chunk_index, text,
                        path, url, page, section, checksum, qdrant_point_id,
                        created_at, metadata, fts_text, source_id, generation_id
                    FROM ragbot_chunk_batch
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        doc_id = EXCLUDED.doc_id,
                        tenant_id = EXCLUDED.tenant_id,
                        chunk_index = EXCLUDED.chunk_index,
                        text = EXCLUDED.text,
                        path = EXCLUDED.path,
                        url = EXCLUDED.url,
                        page = EXCLUDED.page,
                        section = EXCLUDED.section,
                        checksum = EXCLUDED.checksum,
                        qdrant_point_id = EXCLUDED.qdrant_point_id,
                        metadata = EXCLUDED.metadata,
                        fts_text = EXCLUDED.fts_text,
                        source_id = EXCLUDED.source_id,
                        generation_id = EXCLUDED.generation_id
                    """
                )
        _metric(self, "copy_chunk_batches", 1)
        _metric(self, "copy_chunk_rows", len(items))
        return len(items)

    def stage_knowledge_generation(
        self,
        generation_id: str,
        documents: Iterable[Document],
        chunks: Iterable[Chunk],
    ) -> dict[str, int]:
        docs = list(documents)
        items = list(chunks)
        threshold = max(1, int(getattr(self, "_ragbot_pg_copy_min_rows", 256)))
        if max(len(docs), len(items)) < threshold:
            return dict(self._ragbot_legacy_stage_knowledge_generation(generation_id, docs, items))

        now = datetime.now(timezone.utc).isoformat()
        with self._pool.connection() as conn:
            with conn.transaction():
                row = conn.execute(
                    "SELECT status, source_id FROM knowledge_generations "
                    "WHERE generation_id = %s FOR UPDATE",
                    (generation_id,),
                ).fetchone()
                if not row:
                    raise ValueError(f"Unknown knowledge generation: {generation_id}")
                status = str(row["status"])
                if status not in {"staging", "prepared"}:
                    raise ValueError(
                        f"Generation {generation_id} cannot be staged from status={status}"
                    )
                source_id = str(row["source_id"])
                conn.execute("DELETE FROM staged_chunks WHERE generation_id = %s", (generation_id,))
                conn.execute("DELETE FROM staged_documents WHERE generation_id = %s", (generation_id,))

                if docs:
                    with conn.cursor() as cur:
                        with cur.copy(
                            """
                            COPY staged_documents (
                                generation_id, source_id, doc_id, tenant_id, source_type,
                                title, uri, version, doc_updated_at, ingested_at,
                                tags, acl_policy_id, status
                            ) FROM STDIN
                            """
                        ) as copy:
                            for doc in docs:
                                copy.write_row(
                                    (
                                        generation_id,
                                        doc.source_id or source_id,
                                        doc.doc_id,
                                        doc.tenant_id,
                                        doc.source_type,
                                        doc.title,
                                        doc.uri,
                                        doc.version,
                                        doc.doc_updated_at,
                                        doc.ingested_at or now,
                                        list(doc.tags or []),
                                        doc.acl_policy_id,
                                        doc.status,
                                    )
                                )

                if items:
                    with conn.cursor() as cur:
                        with cur.copy(
                            """
                            COPY staged_chunks (
                                generation_id, source_id, chunk_id, doc_id, tenant_id,
                                chunk_index, text, path, url, page, section, checksum,
                                qdrant_point_id, created_at, metadata, fts_text
                            ) FROM STDIN
                            """
                        ) as copy:
                            for chunk in items:
                                copy.write_row(
                                    (
                                        generation_id,
                                        chunk.source_id or source_id,
                                        chunk.chunk_id,
                                        chunk.doc_id,
                                        chunk.tenant_id,
                                        chunk.chunk_index,
                                        chunk.text,
                                        chunk.path,
                                        chunk.url,
                                        chunk.page,
                                        chunk.section,
                                        chunk.checksum,
                                        chunk.qdrant_point_id,
                                        chunk.created_at or now,
                                        self._jsonb(chunk.metadata or {}),
                                        lexicalize(chunk.text) if contains_cjk(chunk.text) else chunk.text,
                                    )
                                )
        _metric(self, "copy_generation_batches", 1)
        _metric(self, "copy_generation_rows", len(docs) + len(items))
        return {"documents": len(docs), "chunks": len(items)}

    def database_runtime_metrics(self) -> dict[str, Any]:
        pool_stats = {}
        getter = getattr(self._pool, "get_stats", None)
        if callable(getter):
            try:
                pool_stats = dict(getter())
            except Exception:
                pool_stats = {}
        lock = getattr(self, "_ragbot_perf_lock")
        with lock:
            custom = dict(getattr(self, "_ragbot_perf_metrics"))
        attempts = int(custom.get("claim_attempts", 0) or 0)
        custom["claim_mean_ms"] = round(
            float(custom.get("claim_total_ms", 0.0) or 0.0) / max(1, attempts), 3
        )
        custom["claim_max_ms"] = round(float(custom.get("claim_max_ms", 0.0) or 0.0), 3)
        return {
            "pool": pool_stats,
            "hot_path": custom,
            "config": {
                "pool_min": int(getattr(self, "_ragbot_pg_pool_min", 2)),
                "pool_max": int(getattr(self, "_ragbot_pg_pool_max", 10)),
                "connect_timeout_seconds": int(getattr(self, "_ragbot_pg_connect_timeout_seconds", 5)),
                "statement_timeout_ms": int(getattr(self, "_ragbot_pg_statement_timeout_ms", 30000)),
                "lock_timeout_ms": int(getattr(self, "_ragbot_pg_lock_timeout_ms", 5000)),
                "idle_transaction_timeout_ms": int(
                    getattr(self, "_ragbot_pg_idle_transaction_timeout_ms", 30000)
                ),
                "copy_min_rows": int(getattr(self, "_ragbot_pg_copy_min_rows", 256)),
            },
        }


def _observe(repo: Any, key: str, amount: int, *, elapsed_ms: float, hit: bool) -> None:
    lock = getattr(repo, "_ragbot_perf_lock")
    with lock:
        metrics = getattr(repo, "_ragbot_perf_metrics")
        metrics[key] += amount
        metrics["claim_total_ms"] += elapsed_ms
        metrics["claim_max_ms"] = max(float(metrics["claim_max_ms"]), elapsed_ms)
        metrics["claim_hits" if hit else "claim_empty"] += 1


def _metric(repo: Any, key: str, amount: int) -> None:
    lock = getattr(repo, "_ragbot_perf_lock")
    with lock:
        getattr(repo, "_ragbot_perf_metrics")[key] += int(amount)
