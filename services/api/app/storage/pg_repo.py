"""Authoritative PostgreSQL repository for Ragbot runtime persistence.

This module owns the current migration-aligned PostgreSQL implementation.  It
no longer subclasses the historical ``postgres_repo`` adapter; that module is a
compatibility import only. Product-specific scheduling/DLQ operations remain in
``ManagedPostgresRepo`` as a thin extension until they are moved behind focused
stores.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..retrieval.lexical import build_or_tsquery, contains_cjk, lexicalize
from .models import ACLPolicy, Chunk, Document, IngestionJob, Source, TableData

logger = logging.getLogger(__name__)


class PostgresRepo:
    """Migration-aligned PostgreSQL implementation with dict-row semantics."""

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
            open=True,
            kwargs={"row_factory": dict_row},
        )
        logger.info(
            "PostgresRepo connected with dict rows: pool_min=%d, pool_max=%d",
            pool_min,
            pool_max,
        )

    def close(self) -> None:
        self._pool.close()

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

    # ── Documents / chunks ─────────────────────────────────────────────

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

    def get_document(self, doc_id: str) -> Optional[Document]:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = %s", (doc_id,)).fetchone()
        return self._row_to_document(row, None) if row else None

    def list_documents(self, tenant_id: Optional[str] = None) -> List[Document]:
        if tenant_id:
            sql, params = "SELECT * FROM documents WHERE tenant_id = %s ORDER BY ingested_at DESC", (tenant_id,)
        else:
            sql, params = "SELECT * FROM documents ORDER BY ingested_at DESC", ()
        with self._pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_document(row, None) for row in rows]

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
        return [str(row["doc_id"]) for row in rows]

    @staticmethod
    def _chunk_upsert_sql() -> str:
        return """
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

    def _chunk_params(self, chunk: Chunk) -> Dict[str, Any]:
        params = asdict(chunk)
        params["metadata"] = self._jsonb(params["metadata"])
        params["fts_text"] = lexicalize(chunk.text) if contains_cjk(chunk.text) else chunk.text
        return params

    def add_chunk(self, chunk: Chunk) -> None:
        self.add_chunks([chunk])

    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        items = list(chunks)
        if not items:
            return 0
        params = [self._chunk_params(chunk) for chunk in items]
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(self._chunk_upsert_sql(), params)
        return len(items)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE chunk_id = %s", (chunk_id,)).fetchone()
        return self._row_to_chunk(row, None) if row else None

    def list_chunks(self, doc_id: Optional[str] = None) -> List[Chunk]:
        if doc_id:
            sql, params = "SELECT * FROM chunks WHERE doc_id = %s ORDER BY chunk_index", (doc_id,)
        else:
            sql, params = "SELECT * FROM chunks ORDER BY created_at", ()
        with self._pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_chunk(row, None) for row in rows]

    def delete_chunks(self, chunk_ids: Iterable[str]) -> int:
        ids = list(dict.fromkeys(chunk_ids))
        if not ids:
            return 0
        with self._pool.connection() as conn:
            result = conn.execute("DELETE FROM chunks WHERE chunk_id = ANY(%s)", (ids,))
            return result.rowcount or 0

    def delete_chunks_by_doc(self, doc_id: str) -> int:
        with self._pool.connection() as conn:
            result = conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            return result.rowcount or 0

    def iter_chunks(self) -> Iterable[Chunk]:
        with self._pool.connection() as conn:
            with conn.cursor(name="iter_chunks") as cur:
                cur.execute("SELECT * FROM chunks ORDER BY created_at")
                for row in cur:
                    yield self._row_to_chunk(row, None)

    def search_chunks_fts(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int,
    ) -> List[Tuple[Chunk, float]]:
        """Use the stored GIN-backed tsvector with CJK bigram support."""
        cjk = contains_cjk(query)
        params: Dict[str, Any] = {"query": query, "limit": top_k}
        if cjk:
            fts_query = build_or_tsquery(query)
            if not fts_query:
                return []
            params["fts_query"] = fts_query
            query_expr = "to_tsquery('simple', %(fts_query)s)"
        else:
            query_expr = "plainto_tsquery('simple', %(query)s)"

        conditions = [f"c.fts_document @@ {query_expr}"]
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
            SELECT c.*, ts_rank_cd(c.fts_document, {query_expr}) AS fts_score
            FROM chunks AS c
            WHERE {' AND '.join(conditions)}
            ORDER BY fts_score DESC, c.chunk_id
            LIMIT %(limit)s
        """
        with self._pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [(self._row_to_chunk(row, None), float(row.get("fts_score") or 0.0)) for row in rows]

    # ── ACL / sources ──────────────────────────────────────────────────

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
        with self._pool.connection() as conn:
            row = conn.execute("SELECT policy_hash FROM acl_policies WHERE acl_policy_id = %s", (acl_policy_id,)).fetchone()
        return str(row["policy_hash"]) if row else None

    def list_policies(self, tenant_id: Optional[str] = None) -> List[ACLPolicy]:
        if tenant_id:
            sql, params = "SELECT * FROM acl_policies WHERE tenant_id = %s", (tenant_id,)
        else:
            sql, params = "SELECT * FROM acl_policies", ()
        with self._pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_policy(row, None) for row in rows]

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

    def get_source(self, source_id: str) -> Optional[Source]:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT * FROM sources WHERE source_id = %s", (source_id,)).fetchone()
        return self._row_to_source(row, None) if row else None

    def list_sources(self, tenant_id: Optional[str] = None) -> List[Source]:
        if tenant_id:
            sql, params = "SELECT * FROM sources WHERE tenant_id = %s ORDER BY created_at", (tenant_id,)
        else:
            sql, params = "SELECT * FROM sources ORDER BY created_at", ()
        with self._pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_source(row, None) for row in rows]

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
        with self._pool.connection() as conn:
            conn.execute(f"UPDATE sources SET {', '.join(set_clauses)} WHERE source_id = %(source_id)s", params)
        return self.get_source(source_id)

    def delete_source(self, source_id: str) -> bool:
        with self._pool.connection() as conn:
            result = conn.execute(
                "UPDATE sources SET status = 'deleted' WHERE source_id = %s AND status != 'deleted'",
                (source_id,),
            )
        return (result.rowcount or 0) > 0

    # ── Durable jobs ───────────────────────────────────────────────────

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

    def add_job_if_absent(self, job: IngestionJob) -> bool:
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
            ) ON CONFLICT (job_id) DO NOTHING
        """
        params = asdict(job)
        params["source_config"] = self._jsonb(params["source_config"])
        params["stats"] = self._jsonb(params["stats"])
        with self._pool.connection() as conn:
            result = conn.execute(sql, params)
        return (result.rowcount or 0) > 0

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT * FROM ingestion_jobs WHERE job_id = %s", (job_id,)).fetchone()
        return self._row_to_job(row, None) if row else None

    def list_jobs(self, tenant_id: Optional[str] = None, source_id: Optional[str] = None) -> List[IngestionJob]:
        conditions: list[str] = []
        params: list[Any] = []
        if tenant_id:
            conditions.append("tenant_id = %s")
            params.append(tenant_id)
        if source_id:
            conditions.append("source_id = %s")
            params.append(source_id)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._pool.connection() as conn:
            rows = conn.execute(f"SELECT * FROM ingestion_jobs{where} ORDER BY created_at DESC", tuple(params)).fetchall()
        return [self._row_to_job(row, None) for row in rows]

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
        with self._pool.connection() as conn:
            conn.execute(f"UPDATE ingestion_jobs SET {', '.join(set_clauses)} WHERE job_id = %(job_id)s", params)
        return self.get_job(job_id)

    def claim_next_job(
        self,
        worker_id: str,
        lease_seconds: int = 120,
        max_attempts: int = 3,
    ) -> Optional[IngestionJob]:
        with self._pool.connection() as conn:
            with conn.transaction():
                conn.execute(
                    """
                    UPDATE ingestion_jobs
                    SET status = 'failed',
                        error = 'Worker lease expired and maximum attempts were exhausted',
                        completed_at = NOW(), lease_owner = NULL, lease_expires_at = NULL
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
                        SELECT job_id FROM ingestion_jobs
                        WHERE status = 'pending'
                          AND available_at <= NOW()
                          AND attempts < %s
                        ORDER BY available_at, created_at
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    UPDATE ingestion_jobs AS j
                    SET status = 'running', attempts = j.attempts + 1,
                        started_at = COALESCE(j.started_at, NOW()),
                        lease_owner = %s, heartbeat_at = NOW(),
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

    def reconcile_ingestion_jobs(self, max_attempts: int = 3) -> Dict[str, int]:
        # ManagedPostgresRepo owns production reconciliation/DLQ semantics. The
        # base store intentionally has no hidden control-plane side effects.
        return {
            "recovered_running": 0,
            "recovered_failed": 0,
            "dead_lettered_running": 0,
            "dead_lettered_exhausted": 0,
        }

    # ── Development compatibility / diagnostics ────────────────────────

    def register_table(self, table: TableData) -> None:
        return None

    def get_table(self, name: str) -> Optional[TableData]:
        return None

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

    # ── Row mapping ────────────────────────────────────────────────────

    @staticmethod
    def _decode_json(value: Any, default: Any) -> Any:
        if value is None:
            return default
        if isinstance(value, str):
            return json.loads(value)
        return value

    @classmethod
    def _row_to_document(cls, row: Any, _conn: Any) -> Document:
        d = dict(row)
        return Document(
            doc_id=d["doc_id"],
            tenant_id=d["tenant_id"],
            source_type=d["source_type"],
            title=d.get("title", ""),
            uri=d.get("uri", ""),
            version=d.get("version", "1.0"),
            doc_updated_at=d.get("doc_updated_at"),
            ingested_at=d.get("ingested_at"),
            tags=list(cls._decode_json(d.get("tags"), []) or []),
            acl_policy_id=d.get("acl_policy_id"),
            status=d.get("status", "active"),
            source_id=d.get("source_id"),
            generation_id=d.get("generation_id"),
        )

    @classmethod
    def _row_to_chunk(cls, row: Any, _conn: Any) -> Chunk:
        d = dict(row)
        return Chunk(
            chunk_id=d["chunk_id"],
            doc_id=d["doc_id"],
            tenant_id=d["tenant_id"],
            chunk_index=int(d.get("chunk_index", 0)),
            text=d.get("text", ""),
            path=d.get("path"),
            url=d.get("url"),
            page=d.get("page"),
            section=d.get("section"),
            checksum=d.get("checksum"),
            qdrant_point_id=d.get("qdrant_point_id"),
            created_at=d.get("created_at"),
            metadata=dict(cls._decode_json(d.get("metadata"), {}) or {}),
            source_id=d.get("source_id"),
            generation_id=d.get("generation_id"),
        )

    @classmethod
    def _row_to_policy(cls, row: Any, _conn: Any) -> ACLPolicy:
        d = dict(row)
        return ACLPolicy(
            acl_policy_id=d["acl_policy_id"],
            tenant_id=d["tenant_id"],
            rules=dict(cls._decode_json(d.get("rules"), {}) or {}),
            policy_hash=d["policy_hash"],
        )

    @classmethod
    def _row_to_source(cls, row: Any, _conn: Any) -> Source:
        d = dict(row)
        return Source(
            source_id=d["source_id"],
            tenant_id=d["tenant_id"],
            source_type=d["source_type"],
            name=d.get("name", ""),
            config=dict(cls._decode_json(d.get("config"), {}) or {}),
            status=d.get("status", "active"),
            acl_policy_id=d.get("acl_policy_id"),
            tags=list(cls._decode_json(d.get("tags"), []) or []),
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
            sync_enabled=bool(d.get("sync_enabled", False)),
            sync_interval_seconds=d.get("sync_interval_seconds"),
            sync_next_at=d.get("sync_next_at"),
            sync_last_enqueued_at=d.get("sync_last_enqueued_at"),
        )

    @classmethod
    def _row_to_job(cls, row: Any, _conn: Any) -> IngestionJob:
        d = dict(row)
        return IngestionJob(
            job_id=d["job_id"],
            tenant_id=d["tenant_id"],
            source_id=d["source_id"],
            source_type=d["source_type"],
            source_config=dict(cls._decode_json(d.get("source_config"), {}) or {}),
            status=d.get("status", "pending"),
            doc_count=int(d.get("doc_count", 0) or 0),
            chunk_count=int(d.get("chunk_count", 0) or 0),
            error=d.get("error"),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            created_at=d.get("created_at"),
            stats=dict(cls._decode_json(d.get("stats"), {}) or {}),
            attempts=int(d.get("attempts", 0) or 0),
            available_at=d.get("available_at"),
            lease_owner=d.get("lease_owner"),
            lease_expires_at=d.get("lease_expires_at"),
            heartbeat_at=d.get("heartbeat_at"),
            failure_class=d.get("failure_class"),
            dead_lettered_at=d.get("dead_lettered_at"),
        )
