from __future__ import annotations

from typing import Any, Mapping, Optional


def ensure_connector_state_repository(repo: Any) -> Any:
    """Attach restart-safe connector checkpoint state to built-in repositories."""

    if callable(getattr(repo, "get_connector_checkpoint", None)) and callable(
        getattr(repo, "set_connector_checkpoint", None)
    ):
        return repo

    if hasattr(repo, "_pool"):
        setattr(
            repo,
            "get_connector_checkpoint",
            _get_postgres.__get__(repo, type(repo)),
        )
        setattr(
            repo,
            "set_connector_checkpoint",
            _set_postgres.__get__(repo, type(repo)),
        )
        return repo

    if hasattr(repo, "_lock"):
        if not hasattr(repo, "_connector_checkpoints"):
            setattr(repo, "_connector_checkpoints", {})
        setattr(
            repo,
            "get_connector_checkpoint",
            _get_memory.__get__(repo, type(repo)),
        )
        setattr(
            repo,
            "set_connector_checkpoint",
            _set_memory.__get__(repo, type(repo)),
        )
    return repo


def _get_postgres(self: Any, source_id: str) -> Optional[dict[str, Any]]:
    with self._pool.connection() as conn:
        row = conn.execute(
            "SELECT checkpoint FROM connector_checkpoints WHERE source_id = %s",
            (source_id,),
        ).fetchone()
    if not row:
        return None
    value = row["checkpoint"] if isinstance(row, dict) else row[0]
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    try:
        import json

        return dict(json.loads(value))
    except Exception:
        return dict(value)


def _set_postgres(
    self: Any,
    source_id: str,
    tenant_id: str,
    checkpoint: Mapping[str, Any],
) -> None:
    with self._pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO connector_checkpoints(source_id, tenant_id, checkpoint, updated_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (source_id) DO UPDATE SET
                tenant_id = EXCLUDED.tenant_id,
                checkpoint = EXCLUDED.checkpoint,
                updated_at = NOW()
            """,
            (source_id, tenant_id, self._jsonb(dict(checkpoint))),
        )


def _get_memory(self: Any, source_id: str) -> Optional[dict[str, Any]]:
    with self._lock:
        value = self._connector_checkpoints.get(source_id)
        return dict(value["checkpoint"]) if value else None


def _set_memory(
    self: Any,
    source_id: str,
    tenant_id: str,
    checkpoint: Mapping[str, Any],
) -> None:
    with self._lock:
        self._connector_checkpoints[source_id] = {
            "tenant_id": tenant_id,
            "checkpoint": dict(checkpoint),
        }
