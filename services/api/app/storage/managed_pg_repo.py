"""PostgreSQL repository extensions for the product control plane.

The core :mod:`pg_repo` remains focused on retrieval/ingestion primitives. This
adapter adds source scheduling state and atomic scheduled-job insertion without
changing the existing repository API used by lower-level tests and libraries.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .models import IngestionJob, Source
from .pg_repo import PostgresRepo


class ManagedPostgresRepo(PostgresRepo):
    """Production repository with source scheduling/control-plane operations."""

    def add_source(self, source: Source) -> None:
        sql = """
            INSERT INTO sources (
                source_id, tenant_id, source_type, name, config,
                status, acl_policy_id, tags, created_at, updated_at,
                sync_enabled, sync_interval_seconds, sync_next_at,
                sync_last_enqueued_at
            ) VALUES (
                %(source_id)s, %(tenant_id)s, %(source_type)s, %(name)s,
                %(config)s, %(status)s, %(acl_policy_id)s, %(tags)s,
                COALESCE(%(created_at)s, NOW()), COALESCE(%(updated_at)s, NOW()),
                %(sync_enabled)s, %(sync_interval_seconds)s, %(sync_next_at)s,
                %(sync_last_enqueued_at)s
            )
            ON CONFLICT (source_id) DO UPDATE SET
                tenant_id = EXCLUDED.tenant_id,
                source_type = EXCLUDED.source_type,
                name = EXCLUDED.name,
                config = EXCLUDED.config,
                status = EXCLUDED.status,
                acl_policy_id = EXCLUDED.acl_policy_id,
                tags = EXCLUDED.tags,
                updated_at = EXCLUDED.updated_at,
                sync_enabled = EXCLUDED.sync_enabled,
                sync_interval_seconds = EXCLUDED.sync_interval_seconds,
                sync_next_at = EXCLUDED.sync_next_at,
                sync_last_enqueued_at = EXCLUDED.sync_last_enqueued_at
        """
        params = asdict(source)
        params["config"] = self._jsonb(params["config"])
        params["tags"] = list(params["tags"])
        with self._pool.connection() as conn:
            conn.execute(sql, params)

    def update_source(self, source_id: str, **kwargs: Any) -> Optional[Source]:
        if not kwargs:
            return self.get_source(source_id)
        allowed = {
            "name", "config", "status", "acl_policy_id", "tags", "updated_at",
            "sync_enabled", "sync_interval_seconds", "sync_next_at",
            "sync_last_enqueued_at",
        }
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
            conn.execute(
                f"UPDATE sources SET {', '.join(set_clauses)} WHERE source_id = %(source_id)s",
                params,
            )
        return self.get_source(source_id)

    @staticmethod
    def _row_to_source(row: Any, conn: Any) -> Source:
        if hasattr(row, "keys"):
            data = dict(row)
        elif hasattr(row, "_asdict"):
            data = row._asdict()
        else:
            cols = [
                "source_id", "tenant_id", "source_type", "name", "config",
                "status", "acl_policy_id", "tags", "created_at", "updated_at",
                "sync_enabled", "sync_interval_seconds", "sync_next_at",
                "sync_last_enqueued_at",
            ]
            data = dict(zip(cols, row))
        config = data.get("config", {})
        if isinstance(config, str):
            config = json.loads(config)
        tags = data.get("tags", [])
        if isinstance(tags, str):
            tags = json.loads(tags)
        return Source(
            source_id=data["source_id"],
            tenant_id=data["tenant_id"],
            source_type=data["source_type"],
            name=data.get("name", ""),
            config=config,
            status=data.get("status", "active"),
            acl_policy_id=data.get("acl_policy_id"),
            tags=list(tags or []),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            sync_enabled=bool(data.get("sync_enabled", False)),
            sync_interval_seconds=data.get("sync_interval_seconds"),
            sync_next_at=data.get("sync_next_at"),
            sync_last_enqueued_at=data.get("sync_last_enqueued_at"),
        )

    def add_job_if_absent(self, job: IngestionJob) -> bool:
        """Atomically persist one scheduled job without reviving an old job."""
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
            ON CONFLICT (job_id) DO NOTHING
        """
        params = asdict(job)
        params["source_config"] = self._jsonb(params["source_config"])
        params["stats"] = self._jsonb(params["stats"])
        with self._pool.connection() as conn:
            result = conn.execute(sql, params)
            return (result.rowcount or 0) > 0

    def list_due_sources(self, now_iso: str, limit: int = 100) -> List[Source]:
        sql = """
            SELECT * FROM sources
            WHERE status = 'active'
              AND sync_enabled = TRUE
              AND sync_interval_seconds IS NOT NULL
              AND sync_next_at IS NOT NULL
              AND sync_next_at <= %s::timestamptz
            ORDER BY sync_next_at, source_id
            LIMIT %s
        """
        with self._pool.connection() as conn:
            rows = conn.execute(sql, (now_iso, limit)).fetchall()
        return [self._row_to_source(row, None) for row in rows]
