from __future__ import annotations

from typing import Any, Mapping, Optional


def ensure_connector_state_repository(repo: Any) -> Any:
    """Attach restart-safe connector checkpoint state to built-in repositories.

    Candidate checkpoints are staged after a connector stream is fully consumed.
    The active checkpoint advances only after knowledge-generation activation
    succeeds. If checkpoint promotion fails after activation, a retry safely
    replays provider changes rather than losing them.
    """

    if getattr(repo, "_ragbot_connector_state_installed", False):
        return repo

    if hasattr(repo, "_pool"):
        setattr(repo, "get_connector_checkpoint", _get_postgres.__get__(repo, type(repo)))
        setattr(repo, "stage_connector_checkpoint", _stage_postgres.__get__(repo, type(repo)))
        setattr(repo, "promote_connector_checkpoint", _promote_postgres.__get__(repo, type(repo)))
        setattr(repo, "set_connector_checkpoint", _set_postgres.__get__(repo, type(repo)))
    elif hasattr(repo, "_lock"):
        if not hasattr(repo, "_connector_checkpoints"):
            setattr(repo, "_connector_checkpoints", {})
        setattr(repo, "get_connector_checkpoint", _get_memory.__get__(repo, type(repo)))
        setattr(repo, "stage_connector_checkpoint", _stage_memory.__get__(repo, type(repo)))
        setattr(repo, "promote_connector_checkpoint", _promote_memory.__get__(repo, type(repo)))
        setattr(repo, "set_connector_checkpoint", _set_memory.__get__(repo, type(repo)))
    else:
        return repo

    activate = getattr(repo, "activate_knowledge_generation", None)
    if callable(activate) and not hasattr(repo, "_ragbot_connector_legacy_activate"):
        setattr(repo, "_ragbot_connector_legacy_activate", activate)
        setattr(
            repo,
            "activate_knowledge_generation",
            _activate_with_checkpoint.__get__(repo, type(repo)),
        )

    setattr(repo, "_ragbot_connector_state_installed", True)
    return repo


def _activate_with_checkpoint(
    self: Any,
    source_id: str,
    generation_id: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    legacy = getattr(self, "_ragbot_connector_legacy_activate")
    result = legacy(source_id, generation_id, *args, **kwargs)
    promote = getattr(self, "promote_connector_checkpoint", None)
    if callable(promote):
        promote(source_id)
    return result


def _get_postgres(self: Any, source_id: str) -> Optional[dict[str, Any]]:
    with self._pool.connection() as conn:
        row = conn.execute(
            "SELECT checkpoint FROM connector_checkpoints WHERE source_id = %s",
            (source_id,),
        ).fetchone()
    if not row:
        return None
    value = row["checkpoint"] if isinstance(row, dict) else row[0]
    return _mapping(value)


def _stage_postgres(
    self: Any,
    source_id: str,
    tenant_id: str,
    checkpoint: Mapping[str, Any],
) -> None:
    with self._pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO connector_checkpoints(
                source_id, tenant_id, checkpoint, pending_checkpoint, updated_at
            ) VALUES (%s, %s, NULL, %s, NOW())
            ON CONFLICT (source_id) DO UPDATE SET
                tenant_id = EXCLUDED.tenant_id,
                pending_checkpoint = EXCLUDED.pending_checkpoint,
                updated_at = NOW()
            """,
            (source_id, tenant_id, self._jsonb(dict(checkpoint))),
        )


def _promote_postgres(self: Any, source_id: str) -> bool:
    with self._pool.connection() as conn:
        result = conn.execute(
            """
            UPDATE connector_checkpoints
            SET checkpoint = pending_checkpoint,
                pending_checkpoint = NULL,
                updated_at = NOW()
            WHERE source_id = %s
              AND pending_checkpoint IS NOT NULL
            """,
            (source_id,),
        )
    return bool(result.rowcount or 0)


def _set_postgres(
    self: Any,
    source_id: str,
    tenant_id: str,
    checkpoint: Mapping[str, Any],
) -> None:
    """Administrative/direct setter; normal SDK execution uses stage+promote."""
    with self._pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO connector_checkpoints(
                source_id, tenant_id, checkpoint, pending_checkpoint, updated_at
            ) VALUES (%s, %s, %s, NULL, NOW())
            ON CONFLICT (source_id) DO UPDATE SET
                tenant_id = EXCLUDED.tenant_id,
                checkpoint = EXCLUDED.checkpoint,
                pending_checkpoint = NULL,
                updated_at = NOW()
            """,
            (source_id, tenant_id, self._jsonb(dict(checkpoint))),
        )


def _get_memory(self: Any, source_id: str) -> Optional[dict[str, Any]]:
    with self._lock:
        value = self._connector_checkpoints.get(source_id)
        if not value or value.get("checkpoint") is None:
            return None
        return dict(value["checkpoint"])


def _stage_memory(
    self: Any,
    source_id: str,
    tenant_id: str,
    checkpoint: Mapping[str, Any],
) -> None:
    with self._lock:
        entry = dict(self._connector_checkpoints.get(source_id) or {})
        entry["tenant_id"] = tenant_id
        entry["pending_checkpoint"] = dict(checkpoint)
        entry.setdefault("checkpoint", None)
        self._connector_checkpoints[source_id] = entry


def _promote_memory(self: Any, source_id: str) -> bool:
    with self._lock:
        entry = self._connector_checkpoints.get(source_id)
        if not entry or entry.get("pending_checkpoint") is None:
            return False
        entry["checkpoint"] = dict(entry["pending_checkpoint"])
        entry["pending_checkpoint"] = None
        return True


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
            "pending_checkpoint": None,
        }


def _mapping(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    try:
        import json

        parsed = json.loads(value)
        return dict(parsed) if parsed is not None else None
    except Exception:
        return dict(value)
