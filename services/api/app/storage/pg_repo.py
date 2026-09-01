"""Schema-aligned PostgreSQL repository used by the runtime factory.

The original repository implementation predates several schema migrations.
This compatibility layer keeps the public repository surface while making the
PostgreSQL path safe for the current schema:

- connections use ``dict_row`` so decoding is independent of physical column
  order after ALTER TABLE migrations;
- JSONB values use psycopg's ``Jsonb`` adapter rather than text serialization;
- PostgreSQL ``TEXT[]`` tag columns receive Python lists;
- nullable dataclass timestamps allow PostgreSQL defaults to apply;
- mapping rows are handled for scalar/RETURNING helpers and debug export.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

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
                title = EXCLUDED.title,
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

    def add_chunk(self, chunk: Chunk) -> None:
        sql = """
            INSERT INTO chunks (
                chunk_id, doc_id, tenant_id, chunk_index, text,
                path, url, page, section, checksum, qdrant_point_id,
                created_at, metadata
            ) VALUES (
                %(chunk_id)s, %(doc_id)s, %(tenant_id)s, %(chunk_index)s, %(text)s,
                %(path)s, %(url)s, %(page)s, %(section)s, %(checksum)s,
                %(qdrant_point_id)s, COALESCE(%(created_at)s, NOW()), %(metadata)s
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                text = EXCLUDED.text,
                path = EXCLUDED.path,
                url = EXCLUDED.url,
                page = EXCLUDED.page,
                section = EXCLUDED.section,
                checksum = EXCLUDED.checksum,
                qdrant_point_id = EXCLUDED.qdrant_point_id,
                metadata = EXCLUDED.metadata
        """
        params = asdict(chunk)
        params["metadata"] = self._jsonb(params["metadata"])
        with self._pool.connection() as conn:
            conn.execute(sql, params)

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

    def add_job(self, job: IngestionJob) -> None:
        sql = """
            INSERT INTO ingestion_jobs (
                job_id, tenant_id, source_id, source_type, source_config,
                status, doc_count, chunk_count, error,
                started_at, completed_at, created_at, stats
            ) VALUES (
                %(job_id)s, %(tenant_id)s, %(source_id)s, %(source_type)s,
                %(source_config)s, %(status)s, %(doc_count)s, %(chunk_count)s,
                %(error)s, %(started_at)s, %(completed_at)s,
                COALESCE(%(created_at)s, NOW()), %(stats)s
            )
            ON CONFLICT (job_id) DO UPDATE SET
                status = EXCLUDED.status,
                doc_count = EXCLUDED.doc_count,
                chunk_count = EXCLUDED.chunk_count,
                error = EXCLUDED.error,
                started_at = EXCLUDED.started_at,
                completed_at = EXCLUDED.completed_at,
                stats = EXCLUDED.stats
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
            "status",
            "doc_count",
            "chunk_count",
            "error",
            "started_at",
            "completed_at",
            "stats",
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
