"""Schema-aligned PostgreSQL repository used by the runtime factory.

This adapter keeps SQL types aligned with the current migrations and adds the
production-only operations that should execute in PostgreSQL rather than in
application memory (bulk lifecycle cleanup, durable job leasing and full-text
search).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..retrieval.lexical import build_or_tsquery, contains_cjk, lexicalize
from .models import ACLPolicy, Chunk, Document, IngestionJob, Source
from .postgres_repo import PostgresRepo as _LegacyPostgresRepo

logger = logging.getLogger(__name__)


class PostgresRepo(_LegacyPostgresRepo):
    """PostgreSQL repository with current-schema type adaptation."""

    def __init__(self, dsn: str, pool_min: int = 2, pool_max: int = 10) -> None:
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise RuntimeError(
                "psycopg and psycopg_pool are required for PostgresRepo. "
                "Install with: pip install 'psycopg[binary]' psycopg_pool"
            ) from exc

        self._pool = ConnectionPool(
            dsn,
            min_size=pool_min,
            max_size=pool_max,
            kwargs={"row_factory": dict_row},
        )
        logger.info(
            "PostgresRepo connected with dict rows: pool_min=%d, pool_max=%d",
            pool_min,
            pool_max,
        )

    @staticmethod
    def _jsonb(value: Any) -> Any:
        from psycopg.types.json import Jsonb
        return Jsonb(value)

    def healthcheck(self) -> bool:
        try:
            with self._pool.connection() as conn:
                row = conn.execute("SELECT 1 AS ok").fetchone()
            return bool(row and row.get("ok") == 1)
        except Exception:
            logger.exception("PostgreSQL repository healthcheck failed")
            return False

    # ── Documents ──────────────────────────────────────────────────────

    def add_document(self, doc: Document) -> None:
        sql = """
            INSERT INTO documents (
                doc_id, tenant_id, source_type, title, uri, version,
                doc_updated_at, ingested_at, tags, acl_policy_id, status
            ) VALUES (
                %(doc_id)s, %(tenant_id)s, %(source_type)s, %(title)s, %(uri)s,
                %(version)s, %(doc_updated_at)s, %(ingested_at)s,
                %(tags)s, %(acl_policy_id)s, %(status)s
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
                status = EXCLUDED.status
        """
        params = asdict(doc)
        params["tags"] = list(params["tags"])
        with self._pool.connection() as conn:
            conn.execute(sql, params)

    def delete_documents(self, doc_ids: Iterable[str]) -> int:
        ids = list(dict.fromkeys(doc_ids))
        if not ids:
            return 0
        with self._pool.connection() as conn:
            result = conn.execute("DELETE FROM documents WHERE doc_id = ANY(%s)", (ids,))
            return result.rowcount or 0

    def delete_documents_by_source(self, source_id: str) -> List[str]:
        pattern = f"source://{source_id}%"
        local_fs_pattern = f"doc-{source_id}:%"
        sql = """
            DELETE FROM documents
            WHERE uri LIKE %s OR doc_id LIKE %s
            RETURNING doc_id
        """
        with self._pool.connection() as conn:
            rows = conn.execute(sql, (pattern, local_fs_pattern)).fetchall()
            return [row["doc_id"] if isinstance(row, dict) else row[0] for row in rows]

    # ── Chunks ─────────────────────────────────────────────────────────

    def add_chunk(self, chunk: Chunk) -> None:
        sql = """
            INSERT INTO chunks (
                chunk_id, doc_id, tenant_id, chunk_index, text,
                path, url, page, section, checksum, qdrant_point_id,
                created_at, metadata, fts_text
            ) VALUES (
                %(chunk_id)s, %(doc_id)s, %(tenant_id)s, %(chunk_index)s, %(text)s,
                %(path)s, %(url)s, %(page)s, %(section)s, %(checksum)s,
                %(qdrant_point_id)s, COALESCE(%(created_at)s, NOW()), %(metadata)s,
                %(fts_text)s
            )
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
                fts_text = EXCLUDED.fts_text
        """
        params = asdict(chunk)
        params["metadata"] = self._jsonb(params["metadata"])
        params["fts_text"] = lexicalize(chunk.text) if contains_cjk(chunk.text) else chunk.text
        with self._pool.connection() as conn:
            conn.execute(sql, params)

    def delete_chunks(self, chunk_ids: Iterable[str]) -> int:
        ids = list(dict.fromkeys(chunk_ids))
        if not ids:
            return 0
        with self._pool.connection() as conn:
            result = conn.execute("DELETE FROM chunks WHERE chunk_id = ANY(%s)", (ids,))
            return result.rowcount or 0

    def search_chunks_fts(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int,
    ) -> List[Tuple[Chunk, float]]:
        """Use PostgreSQL GIN-backed lexical retrieval with CJK bigram support."""
        cjk = contains_cjk(query)
        params: Dict[str, Any] = {"query": query, "limit": top_k}
        vector_expr = "to_tsvector('simple', COALESCE(NULLIF(c.fts_text, ''), c.text))"
        if cjk:
            fts_query = build_or_tsquery(query)
            if not fts_query:
                return []
            params["fts_query"] = fts_query
            query_expr = "to_tsquery('simple', %(fts_query)s)"
        else:
            query_expr = "plainto_tsquery('simple', %(query)s)"

        conditions = [f"{vector_expr} @@ {query_expr}"]

        tenant_id = filters.get("tenant_id")
        if tenant_id:
            conditions.append("c.tenant_id = %(tenant_id)s")
            params["tenant_id"] = tenant_id

        source_types = filters.get("source_types")
        if source_types:
            conditions.append("c.metadata->>'source_type' = ANY(%(source_types)s)")
            params["source_types"] = list(source_types)

        doc_ids = filters.get("doc_ids")
        if doc_ids:
            conditions.append("c.doc_id = ANY(%(doc_ids)s)")
            params["doc_ids"] = list(doc_ids)

        tags = filters.get("tags")
        if tags:
            conditions.append("COALESCE(c.metadata->'tags', '[]'::jsonb) ?| %(tags)s")
            params["tags"] = list(tags)

        path_prefix = filters.get("path_prefix")
        if path_prefix:
            conditions.append("LEFT(COALESCE(c.path, ''), LENGTH(%(path_prefix)s)) = %(path_prefix)s")
            params["path_prefix"] = path_prefix

        url_prefix = filters.get("url_prefix")
        if url_prefix:
            conditions.append("LEFT(COALESCE(c.url, ''), LENGTH(%(url_prefix)s)) = %(url_prefix)s")
            params["url_prefix"] = url_prefix

        time_range = filters.get("time_range") or {}
        if time_range.get("start"):
            conditions.append("c.created_at >= %(time_start)s::timestamptz")
            params["time_start"] = time_range["start"]
        if time_range.get("end"):
            conditions.append("c.created_at <= %(time_end)s::timestamptz")
            params["time_end"] = time_range["end"]

        security_scope = filters.get("security_scope")
        if security_scope:
            conditions.append("COALESCE(c.metadata->>'acl_hash', 'public') = ANY(%(security_scope)s)")
            params["security_scope"] = list(security_scope)

        sql = f"""
            SELECT c.*,
                   ts_rank_cd({vector_expr}, {query_expr}) AS fts_score
            FROM chunks AS c
            WHERE {' AND '.join(conditions)}
            ORDER BY fts_score DESC, c.chunk_id
            LIMIT %(limit)s
        """
        with self._pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            (self._row_to_chunk(row, None), float(row.get("fts_score") or 0.0))
            for row in rows
        ]

    # ── Policies ───────────────────────────────────────────────────────

    def add_policy(self, policy: ACLPolicy) -> None:
        sql = """
            INSERT INTO acl_policies (acl_policy_id, tenant_id, rules, policy_hash)
            VALUES (%(acl_policy_id)s, %(tenant_id)s, %(rules)s, %(policy_hash)s)
            ON CONFLICT (acl_policy_id) DO UPDATE SET
                tenant_id = EXCLUDED.tenant_id,
                rules = EXCLUDED.rules,
                policy_hash = EXCLUDED.policy_hash
        """
        params = asdict(policy)
        params["rules"] = self._jsonb(params["rules"])
        with self._pool.connection() as conn:
            conn.execute(sql, params)

    def get_policy_hash(self, acl_policy_id: Optional[str] = None) -> Optional[str]:
        if not acl_policy_id:
            return None
        sql = "SELECT policy_hash FROM acl_policies WHERE acl_policy_id = %s"
        with self._pool.connection() as conn:
            row = conn.execute(sql, (acl_policy_id,)).fetchone()
            if not row:
                return None
            return row["policy_hash"] if isinstance(row, dict) else row[0]

    # ── Sources ────────────────────────────────────────────────────────

    def add_source(self, source: Source) -> None:
        sql = """
            INSERT INTO sources (
                source_id, tenant_id, source_type, name, config,
                status, acl_policy_id, tags, created_at, updated_at
            ) VALUES (
                %(source_id)s, %(tenant_id)s, %(source_type)s, %(name)s,
                %(config)s, %(status)s, %(acl_policy_id)s, %(tags)s,
                COALESCE(%(created_at)s, NOW()), COALESCE(%(updated_at)s, NOW())
            )
            ON CONFLICT (source_id) DO UPDATE SET
                tenant_id = EXCLUDED.tenant_id,
                source_type = EXCLUDED.source_type,
                name = EXCLUDED.name,
                config = EXCLUDED.config,
                status = EXCLUDED.status,
                acl_policy_id = EXCLUDED.acl_policy_id,
                tags = EXCLUDED.tags,
                updated_at = EXCLUDED.updated_at
        """
        params = asdict(source)
        params["config"] = self._jsonb(params["config"])
        params["tags"] = list(params["tags"])
        with self._pool.connection() as conn:
            conn.execute(sql, params)

    def update_source(self, source_id: str, **kwargs: Any) -> Optional[Source]:
        if not kwargs:
            return self.get_source(source_id)
        allowed = {"name", "config", "status", "acl_policy_id", "tags", "updated_at"}
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(f"Unsupported source fields: {sorted(unknown)}")

        set_clauses = []
        params: Dict[str, Any] = {"source_id": source_id}
        for key, value in kwargs.items():
            set_clauses.append(f"{key} = %({key})s")
            if key == "config":
                params[key] = self._jsonb(value)
            elif key == "tags":
                params[key] = list(value)
            else:
                params[key] = value
        sql = f"UPDATE sources SET {', '.join(set_clauses)} WHERE source_id = %(source_id)s"
        with self._pool.connection() as conn:
            conn.execute(sql, params)
        return self.get_source(source_id)

    # ── Jobs ───────────────────────────────────────────────────────────

    def add_job(self, job: IngestionJob) -> None:
        sql = """
            INSERT INTO ingestion_jobs (
                job_id, tenant_id, source_id, source_type, source_config,
                status, doc_count, chunk_count, error,
                started_at, completed_at, created_at, stats,
                attempts, available_at, lease_owner, lease_expires_at, heartbeat_at
            ) VALUES (
                %(job_id)s, %(tenant_id)s, %(source_id)s, %(source_type)s,
                %(source_config)s, %(status)s, %(doc_count)s, %(chunk_count)s,
                %(error)s, %(started_at)s, %(completed_at)s,
                COALESCE(%(created_at)s, NOW()), %(stats)s,
                %(attempts)s, COALESCE(%(available_at)s, NOW()), %(lease_owner)s,
                %(lease_expires_at)s, %(heartbeat_at)s
            )
            ON CONFLICT (job_id) DO UPDATE SET
                status = EXCLUDED.status,
                doc_count = EXCLUDED.doc_count,
                chunk_count = EXCLUDED.chunk_count,
                error = EXCLUDED.error,
                started_at = EXCLUDED.started_at,
                completed_at = EXCLUDED.completed_at,
                stats = EXCLUDED.stats,
                attempts = EXCLUDED.attempts,
                available_at = EXCLUDED.available_at,
                lease_owner = EXCLUDED.lease_owner,
                lease_expires_at = EXCLUDED.lease_expires_at,
                heartbeat_at = EXCLUDED.heartbeat_at
        """
        params = asdict(job)
        params["source_config"] = self._jsonb(params["source_config"])
        params["stats"] = self._jsonb(params["stats"])
        with self._pool.connection() as conn:
            conn.execute(sql, params)

    def update_job(self, job_id: str, **kwargs: Any) -> Optional[IngestionJob]:
        if not kwargs:
            return self.get_job(job_id)
        allowed = {
            "status", "doc_count", "chunk_count", "error", "started_at",
            "completed_at", "stats", "attempts", "available_at", "lease_owner",
            "lease_expires_at", "heartbeat_at",
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(f"Unsupported job fields: {sorted(unknown)}")

        set_clauses = []
        params: Dict[str, Any] = {"job_id": job_id}
        for key, value in kwargs.items():
            set_clauses.append(f"{key} = %({key})s")
            params[key] = self._jsonb(value) if key == "stats" else value
        sql = f"UPDATE ingestion_jobs SET {', '.join(set_clauses)} WHERE job_id = %(job_id)s"
        with self._pool.connection() as conn:
            conn.execute(sql, params)
        return self.get_job(job_id)

    def claim_next_job(
        self,
        worker_id: str,
        lease_seconds: int = 120,
        max_attempts: int = 3,
    ) -> Optional[IngestionJob]:
        """Atomically recover expired leases and claim one pending job."""
        with self._pool.connection() as conn:
            with conn.transaction():
                conn.execute(
                    """
                    UPDATE ingestion_jobs
                    SET status = 'failed',
                        error = 'Worker lease expired and maximum attempts were exhausted',
                        completed_at = NOW(),
                        lease_owner = NULL,
                        lease_expires_at = NULL
                    WHERE status = 'running'
                      AND lease_expires_at IS NOT NULL
                      AND lease_expires_at <= NOW()
                      AND attempts >= %s
                    """,
                    (max_attempts,),
                )
                conn.execute(
                    """
                    UPDATE ingestion_jobs
                    SET status = 'pending', lease_owner = NULL, lease_expires_at = NULL
                    WHERE status = 'running'
                      AND lease_expires_at IS NOT NULL
                      AND lease_expires_at <= NOW()
                      AND attempts < %s
                    """,
                    (max_attempts,),
                )
                row = conn.execute(
                    """
                    WITH candidate AS (
                        SELECT job_id
                        FROM ingestion_jobs
                        WHERE status = 'pending'
                          AND available_at <= NOW()
                          AND attempts < %s
                        ORDER BY available_at, created_at
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
                        error = NULL
                    FROM candidate
                    WHERE j.job_id = candidate.job_id
                    RETURNING j.*
                    """,
                    (max_attempts, worker_id, lease_seconds),
                ).fetchone()
        return self._row_to_job(row, None) if row else None

    def heartbeat_job(self, job_id: str, worker_id: str, lease_seconds: int = 120) -> bool:
        with self._pool.connection() as conn:
            result = conn.execute(
                """
                UPDATE ingestion_jobs
                SET heartbeat_at = NOW(),
                    lease_expires_at = NOW() + (%s * INTERVAL '1 second')
                WHERE job_id = %s AND status = 'running' AND lease_owner = %s
                """,
                (lease_seconds, job_id, worker_id),
            )
            return (result.rowcount or 0) > 0

    def release_job_lease(self, job_id: str, worker_id: str) -> bool:
        with self._pool.connection() as conn:
            result = conn.execute(
                """
                UPDATE ingestion_jobs
                SET lease_owner = NULL, lease_expires_at = NULL
                WHERE job_id = %s AND lease_owner = %s
                """,
                (job_id, worker_id),
            )
            return (result.rowcount or 0) > 0

    @staticmethod
    def _row_to_job(row: Any, conn: Any) -> IngestionJob:
        if hasattr(row, "keys"):
            d = dict(row)
        elif hasattr(row, "_asdict"):
            d = row._asdict()
        else:
            cols = [
                "job_id", "tenant_id", "source_id", "source_type", "source_config",
                "status", "doc_count", "chunk_count", "error", "started_at",
                "completed_at", "created_at", "stats", "attempts", "available_at",
                "lease_owner", "lease_expires_at", "heartbeat_at",
            ]
            d = dict(zip(cols, row))
        source_config = d.get("source_config", {})
        if isinstance(source_config, str):
            source_config = json.loads(source_config)
        stats = d.get("stats", {})
        if isinstance(stats, str):
            stats = json.loads(stats)
        return IngestionJob(
            job_id=d["job_id"],
            tenant_id=d["tenant_id"],
            source_id=d["source_id"],
            source_type=d["source_type"],
            source_config=source_config,
            status=d.get("status", "pending"),
            doc_count=d.get("doc_count", 0),
            chunk_count=d.get("chunk_count", 0),
            error=d.get("error"),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            created_at=d.get("created_at"),
            stats=stats,
            attempts=d.get("attempts", 0),
            available_at=d.get("available_at"),
            lease_owner=d.get("lease_owner"),
            lease_expires_at=d.get("lease_expires_at"),
            heartbeat_at=d.get("heartbeat_at"),
        )

    # ── Debug ──────────────────────────────────────────────────────────

    def export_state(self) -> Dict[str, List[dict]]:
        result: Dict[str, List[dict]] = {}
        tables = [
            ("documents", "SELECT * FROM documents"),
            ("chunks", "SELECT * FROM chunks"),
            ("acl_policies", "SELECT * FROM acl_policies"),
            ("sources", "SELECT * FROM sources"),
            ("ingestion_jobs", "SELECT * FROM ingestion_jobs"),
        ]
        with self._pool.connection() as conn:
            for name, sql in tables:
                rows = conn.execute(sql).fetchall()
                result[name] = [dict(row) for row in rows]
        return result
