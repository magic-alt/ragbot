"""Schema-aligned PostgreSQL repository used by the runtime factory.

This module keeps compatibility with the original ``postgres_repo``
implementation while fixing PostgreSQL-specific adaptation details:

- connections use ``dict_row`` so row decoding is independent of migration
  column order;
- PostgreSQL ``TEXT[]`` tag columns receive Python lists, not JSON strings;
- methods that previously indexed tuple rows support mapping rows;
- debug export works with mapping rows.

The legacy import path remains available for compatibility, while new runtime
construction should import ``PostgresRepo`` from this module.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .models import Document, Source
from .postgres_repo import PostgresRepo as _LegacyPostgresRepo

logger = logging.getLogger(__name__)


class PostgresRepo(_LegacyPostgresRepo):
    """PostgreSQL repository with schema-safe row and array adaptation."""

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
        # psycopg adapts Python lists directly to PostgreSQL TEXT[].
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
        params["tags"] = list(params["tags"])
        with self._pool.connection() as conn:
            conn.execute(sql, params)

    def update_source(self, source_id: str, **kwargs: Any) -> Optional[Source]:
        if not kwargs:
            return self.get_source(source_id)
        set_clauses = []
        params: Dict[str, Any] = {"source_id": source_id}
        for key, value in kwargs.items():
            set_clauses.append(f"{key} = %({key})s")
            if key == "config":
                params[key] = json.dumps(value)
            elif key == "tags":
                params[key] = list(value)
            else:
                params[key] = value
        sql = f"UPDATE sources SET {', '.join(set_clauses)} WHERE source_id = %(source_id)s"
        with self._pool.connection() as conn:
            conn.execute(sql, params)
        return self.get_source(source_id)

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
