"""PostgreSQL-backed repository implementation.

Uses psycopg (v3) with connection pooling. Maps to the schema defined
in infra/migrations/001-003.

Requires: psycopg[binary,pool] >= 3.1
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional

from .models import ACLPolicy, Chunk, Document, IngestionJob, Source, TableData

logger = logging.getLogger(__name__)


class PostgresRepo:
    """Production repository backed by PostgreSQL."""

    def __init__(self, dsn: str, pool_min: int = 2, pool_max: int = 10) -> None:
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise RuntimeError(
                "psycopg_pool is required for PostgresRepo. "
                "Install with: pip install 'psycopg[binary,pool]'"
            ) from exc

        self._pool = ConnectionPool(dsn, min_size=pool_min, max_size=pool_max)
        logger.info("PostgresRepo connected: pool_min=%d, pool_max=%d", pool_min, pool_max)

    def close(self) -> None:
        self._pool.close()

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
                title = EXCLUDED.title,
                version = EXCLUDED.version,
                doc_updated_at = EXCLUDED.doc_updated_at,
                ingested_at = EXCLUDED.ingested_at,
                tags = EXCLUDED.tags,
                status = EXCLUDED.status
        """
        params = asdict(doc)
        params["tags"] = json.dumps(params["tags"])
        with self._pool.connection() as conn:
            conn.execute(sql, params)

    def get_document(self, doc_id: str) -> Optional[Document]:
        sql = "SELECT * FROM documents WHERE doc_id = %s"
        with self._pool.connection() as conn:
            row = conn.execute(sql, (doc_id,)).fetchone()
            if not row:
                return None
            return self._row_to_document(row, conn)

    def list_documents(self, tenant_id: Optional[str] = None) -> List[Document]:
        if tenant_id:
            sql = "SELECT * FROM documents WHERE tenant_id = %s ORDER BY ingested_at DESC"
            params: tuple = (tenant_id,)
        else:
            sql = "SELECT * FROM documents ORDER BY ingested_at DESC"
            params = ()
        with self._pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_document(row, conn) for row in rows]

    def delete_documents_by_source(self, source_id: str) -> List[str]:
        pattern = f"source://{source_id}%"
        sql = "DELETE FROM documents WHERE uri LIKE %s RETURNING doc_id"
        with self._pool.connection() as conn:
            rows = conn.execute(sql, (pattern,)).fetchall()
            return [row[0] for row in rows]

    # ── Chunks ─────────────────────────────────────────────────────────

    def add_chunk(self, chunk: Chunk) -> None:
        sql = """
            INSERT INTO chunks (
                chunk_id, doc_id, tenant_id, chunk_index, text,
                path, url, page, section, checksum, qdrant_point_id,
                created_at, metadata
            ) VALUES (
                %(chunk_id)s, %(doc_id)s, %(tenant_id)s, %(chunk_index)s, %(text)s,
                %(path)s, %(url)s, %(page)s, %(section)s, %(checksum)s,
                %(qdrant_point_id)s, %(created_at)s, %(metadata)s
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                text = EXCLUDED.text,
                checksum = EXCLUDED.checksum,
                qdrant_point_id = EXCLUDED.qdrant_point_id,
                metadata = EXCLUDED.metadata
        """
        params = asdict(chunk)
        params["metadata"] = json.dumps(params["metadata"])
        with self._pool.connection() as conn:
            conn.execute(sql, params)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        sql = "SELECT * FROM chunks WHERE chunk_id = %s"
        with self._pool.connection() as conn:
            row = conn.execute(sql, (chunk_id,)).fetchone()
            if not row:
                return None
            return self._row_to_chunk(row, conn)

    def list_chunks(self, doc_id: Optional[str] = None) -> List[Chunk]:
        if doc_id:
            sql = "SELECT * FROM chunks WHERE doc_id = %s ORDER BY chunk_index"
            params: tuple = (doc_id,)
        else:
            sql = "SELECT * FROM chunks ORDER BY created_at"
            params = ()
        with self._pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_chunk(row, conn) for row in rows]

    def delete_chunks_by_doc(self, doc_id: str) -> int:
        sql = "DELETE FROM chunks WHERE doc_id = %s"
        with self._pool.connection() as conn:
            result = conn.execute(sql, (doc_id,))
            return result.rowcount or 0

    def iter_chunks(self) -> Iterable[Chunk]:
        sql = "SELECT * FROM chunks ORDER BY created_at"
        with self._pool.connection() as conn:
            with conn.cursor(name="iter_chunks") as cur:
                cur.execute(sql)
                for row in cur:
                    yield self._row_to_chunk(row, conn)

    # ── Policies ───────────────────────────────────────────────────────

    def add_policy(self, policy: ACLPolicy) -> None:
        sql = """
            INSERT INTO acl_policies (acl_policy_id, tenant_id, rules, policy_hash)
            VALUES (%(acl_policy_id)s, %(tenant_id)s, %(rules)s, %(policy_hash)s)
            ON CONFLICT (acl_policy_id) DO UPDATE SET
                rules = EXCLUDED.rules,
                policy_hash = EXCLUDED.policy_hash
        """
        params = asdict(policy)
        params["rules"] = json.dumps(params["rules"])
        with self._pool.connection() as conn:
            conn.execute(sql, params)

    def get_policy_hash(self, acl_policy_id: Optional[str] = None) -> Optional[str]:
        if not acl_policy_id:
            return None
        sql = "SELECT policy_hash FROM acl_policies WHERE acl_policy_id = %s"
        with self._pool.connection() as conn:
            row = conn.execute(sql, (acl_policy_id,)).fetchone()
            return row[0] if row else None

    def list_policies(self, tenant_id: Optional[str] = None) -> List[ACLPolicy]:
        if tenant_id:
            sql = "SELECT * FROM acl_policies WHERE tenant_id = %s"
            params: tuple = (tenant_id,)
        else:
            sql = "SELECT * FROM acl_policies"
            params = ()
        with self._pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_policy(row, conn) for row in rows]

    # ── Sources ────────────────────────────────────────────────────────

    def add_source(self, source: Source) -> None:
        sql = """
            INSERT INTO sources (
                source_id, tenant_id, source_type, name, config,
                status, acl_policy_id, tags, created_at, updated_at
            ) VALUES (
                %(source_id)s, %(tenant_id)s, %(source_type)s, %(name)s,
                %(config)s, %(status)s, %(acl_policy_id)s, %(tags)s,
                %(created_at)s, %(updated_at)s
            )
            ON CONFLICT (source_id) DO UPDATE SET
                name = EXCLUDED.name,
                config = EXCLUDED.config,
                status = EXCLUDED.status,
                tags = EXCLUDED.tags,
                updated_at = EXCLUDED.updated_at
        """
        params = asdict(source)
        params["config"] = json.dumps(params["config"])
        params["tags"] = json.dumps(params["tags"])
        with self._pool.connection() as conn:
            conn.execute(sql, params)

    def get_source(self, source_id: str) -> Optional[Source]:
        sql = "SELECT * FROM sources WHERE source_id = %s"
        with self._pool.connection() as conn:
            row = conn.execute(sql, (source_id,)).fetchone()
            if not row:
                return None
            return self._row_to_source(row, conn)

    def list_sources(self, tenant_id: Optional[str] = None) -> List[Source]:
        if tenant_id:
            sql = "SELECT * FROM sources WHERE tenant_id = %s ORDER BY created_at"
            params: tuple = (tenant_id,)
        else:
            sql = "SELECT * FROM sources ORDER BY created_at"
            params = ()
        with self._pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_source(row, conn) for row in rows]

    def update_source(self, source_id: str, **kwargs: Any) -> Optional[Source]:
        if not kwargs:
            return self.get_source(source_id)
        set_clauses = []
        params: Dict[str, Any] = {"source_id": source_id}
        for key, value in kwargs.items():
            set_clauses.append(f"{key} = %({key})s")
            if key in ("config", "tags"):
                params[key] = json.dumps(value)
            else:
                params[key] = value
        sql = f"UPDATE sources SET {', '.join(set_clauses)} WHERE source_id = %(source_id)s"
        with self._pool.connection() as conn:
            conn.execute(sql, params)
        return self.get_source(source_id)

    def delete_source(self, source_id: str) -> bool:
        sql = "UPDATE sources SET status = 'deleted' WHERE source_id = %s AND status != 'deleted'"
        with self._pool.connection() as conn:
            result = conn.execute(sql, (source_id,))
            return (result.rowcount or 0) > 0

    # ── Jobs ───────────────────────────────────────────────────────────

    def add_job(self, job: IngestionJob) -> None:
        sql = """
            INSERT INTO ingestion_jobs (
                job_id, tenant_id, source_id, source_type, source_config,
                status, doc_count, chunk_count, error,
                started_at, completed_at, created_at, stats
            ) VALUES (
                %(job_id)s, %(tenant_id)s, %(source_id)s, %(source_type)s,
                %(source_config)s, %(status)s, %(doc_count)s, %(chunk_count)s,
                %(error)s, %(started_at)s, %(completed_at)s, %(created_at)s, %(stats)s
            )
            ON CONFLICT (job_id) DO UPDATE SET
                status = EXCLUDED.status,
                doc_count = EXCLUDED.doc_count,
                chunk_count = EXCLUDED.chunk_count,
                error = EXCLUDED.error,
                completed_at = EXCLUDED.completed_at,
                stats = EXCLUDED.stats
        """
        params = asdict(job)
        params["source_config"] = json.dumps(params["source_config"])
        params["stats"] = json.dumps(params["stats"])
        with self._pool.connection() as conn:
            conn.execute(sql, params)

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        sql = "SELECT * FROM ingestion_jobs WHERE job_id = %s"
        with self._pool.connection() as conn:
            row = conn.execute(sql, (job_id,)).fetchone()
            if not row:
                return None
            return self._row_to_job(row, conn)

    def list_jobs(self, tenant_id: Optional[str] = None, source_id: Optional[str] = None) -> List[IngestionJob]:
        conditions = []
        params: list = []
        if tenant_id:
            conditions.append("tenant_id = %s")
            params.append(tenant_id)
        if source_id:
            conditions.append("source_id = %s")
            params.append(source_id)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM ingestion_jobs{where} ORDER BY created_at DESC"
        with self._pool.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            return [self._row_to_job(row, conn) for row in rows]

    def update_job(self, job_id: str, **kwargs: Any) -> Optional[IngestionJob]:
        if not kwargs:
            return self.get_job(job_id)
        set_clauses = []
        params: Dict[str, Any] = {"job_id": job_id}
        for key, value in kwargs.items():
            set_clauses.append(f"{key} = %({key})s")
            if key in ("source_config", "stats"):
                params[key] = json.dumps(value)
            else:
                params[key] = value
        sql = f"UPDATE ingestion_jobs SET {', '.join(set_clauses)} WHERE job_id = %(job_id)s"
        with self._pool.connection() as conn:
            conn.execute(sql, params)
        return self.get_job(job_id)

    # ── Tables (no-op for PostgresRepo) ────────────────────────────────

    def register_table(self, table: TableData) -> None:
        pass

    def get_table(self, name: str) -> Optional[TableData]:
        return None

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
                cols = [desc.name for desc in conn.execute(sql).description or []]
                result[name] = [dict(zip(cols, row)) for row in rows]
        return result

    # ── Row mapping helpers ────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: Any, conn: Any) -> Dict[str, Any]:
        """Convert a row tuple to a dict using cursor description."""
        if isinstance(row, dict):
            return row
        # psycopg3 with row_factory=dict_row would return dicts
        # For tuple rows, we need column names from the last executed query
        return dict(row) if hasattr(row, "keys") else {}

    @staticmethod
    def _row_to_document(row: Any, conn: Any) -> Document:
        if hasattr(row, "keys"):
            d = dict(row)
        elif hasattr(row, "_asdict"):
            d = row._asdict()
        else:
            # Fallback: assume tuple order matches Document fields
            cols = [
                "doc_id", "tenant_id", "source_type", "title", "uri",
                "version", "doc_updated_at", "ingested_at", "tags",
                "acl_policy_id", "status",
            ]
            d = dict(zip(cols, row))
        tags = d.get("tags", "[]")
        if isinstance(tags, str):
            tags = json.loads(tags)
        return Document(
            doc_id=d["doc_id"],
            tenant_id=d["tenant_id"],
            source_type=d["source_type"],
            title=d.get("title", ""),
            uri=d.get("uri", ""),
            version=d.get("version", ""),
            doc_updated_at=d.get("doc_updated_at", ""),
            ingested_at=d.get("ingested_at", ""),
            tags=tags,
            acl_policy_id=d.get("acl_policy_id"),
            status=d.get("status", "active"),
        )

    @staticmethod
    def _row_to_chunk(row: Any, conn: Any) -> Chunk:
        if hasattr(row, "keys"):
            d = dict(row)
        elif hasattr(row, "_asdict"):
            d = row._asdict()
        else:
            cols = [
                "chunk_id", "doc_id", "tenant_id", "chunk_index", "text",
                "path", "url", "page", "section", "checksum",
                "qdrant_point_id", "created_at", "metadata",
            ]
            d = dict(zip(cols, row))
        metadata = d.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return Chunk(
            chunk_id=d["chunk_id"],
            doc_id=d["doc_id"],
            tenant_id=d["tenant_id"],
            chunk_index=d.get("chunk_index", 0),
            text=d.get("text", ""),
            path=d.get("path"),
            url=d.get("url"),
            page=d.get("page"),
            section=d.get("section"),
            checksum=d.get("checksum"),
            qdrant_point_id=d.get("qdrant_point_id"),
            created_at=d.get("created_at"),
            metadata=metadata,
        )

    @staticmethod
    def _row_to_policy(row: Any, conn: Any) -> ACLPolicy:
        if hasattr(row, "keys"):
            d = dict(row)
        elif hasattr(row, "_asdict"):
            d = row._asdict()
        else:
            cols = ["acl_policy_id", "tenant_id", "rules", "policy_hash"]
            d = dict(zip(cols, row))
        rules = d.get("rules", "{}")
        if isinstance(rules, str):
            rules = json.loads(rules)
        return ACLPolicy(
            acl_policy_id=d["acl_policy_id"],
            tenant_id=d["tenant_id"],
            rules=rules,
            policy_hash=d["policy_hash"],
        )

    @staticmethod
    def _row_to_source(row: Any, conn: Any) -> Source:
        if hasattr(row, "keys"):
            d = dict(row)
        elif hasattr(row, "_asdict"):
            d = row._asdict()
        else:
            cols = [
                "source_id", "tenant_id", "source_type", "name", "config",
                "status", "acl_policy_id", "tags", "created_at", "updated_at",
            ]
            d = dict(zip(cols, row))
        config = d.get("config", "{}")
        if isinstance(config, str):
            config = json.loads(config)
        tags = d.get("tags", "[]")
        if isinstance(tags, str):
            tags = json.loads(tags)
        return Source(
            source_id=d["source_id"],
            tenant_id=d["tenant_id"],
            source_type=d["source_type"],
            name=d.get("name", ""),
            config=config,
            status=d.get("status", "active"),
            acl_policy_id=d.get("acl_policy_id"),
            tags=tags,
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
        )

    @staticmethod
    def _row_to_job(row: Any, conn: Any) -> IngestionJob:
        if hasattr(row, "keys"):
            d = dict(row)
        elif hasattr(row, "_asdict"):
            d = row._asdict()
        else:
            cols = [
                "job_id", "tenant_id", "source_id", "source_type",
                "source_config", "status", "doc_count", "chunk_count",
                "error", "started_at", "completed_at", "created_at", "stats",
            ]
            d = dict(zip(cols, row))
        source_config = d.get("source_config", "{}")
        if isinstance(source_config, str):
            source_config = json.loads(source_config)
        stats = d.get("stats", "{}")
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
        )
