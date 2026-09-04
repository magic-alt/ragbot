from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from .store import build_upload_store_from_env
from .uri import is_upload_uri, upload_object_id


def source_upload_object_id(source: Any) -> str | None:
    config = dict(getattr(source, "config", {}) or {})
    explicit = config.get("upload_object_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    path = config.get("path")
    if isinstance(path, str) and is_upload_uri(path):
        return upload_object_id(path)
    return None


def retire_uploaded_object_for_source(repo: Any, source: Any) -> bool:
    object_id = source_upload_object_id(source)
    if not object_id or not callable(getattr(repo, "get_uploaded_object", None)):
        return False
    obj = repo.get_uploaded_object(object_id)
    if obj is None or obj.state == "deleted":
        return False
    if obj.tenant_id != getattr(source, "tenant_id", None):
        raise ValueError("Uploaded object tenant does not match Source tenant")
    now = datetime.now(timezone.utc).isoformat()
    repo.update_uploaded_object(
        object_id,
        state="retired",
        ref_count=max(0, int(obj.ref_count or 0) - 1),
        retired_at=now,
    )
    return True


def gc_uploaded_objects(
    repo: Any,
    *,
    tenant_id: str | None = None,
    retention_seconds: int | None = None,
) -> dict[str, int]:
    """Delete unreferenced uploaded objects after a retention window.

    ``tenant_id`` scopes operator-triggered GC. Passing ``None`` is reserved for
    trusted internal maintenance paths. Source deletion only retires an object;
    storage deletion occurs after retention so failed submissions and accidental
    deletes remain diagnosable.
    """
    list_objects = getattr(repo, "list_uploaded_objects", None)
    update = getattr(repo, "update_uploaded_object", None)
    if not callable(list_objects) or not callable(update):
        return {"scanned": 0, "deleted": 0, "errors": 0}
    if retention_seconds is None:
        retention_seconds = _nonnegative_int_env("RAGBOT_UPLOAD_RETENTION_SECONDS", 86400)
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=retention_seconds)
    objects = list_objects(tenant_id=tenant_id) if tenant_id is not None else list_objects()
    candidates = [
        obj for obj in objects
        if obj.state in {"orphaned", "retired"}
        and int(obj.ref_count or 0) == 0
        and _timestamp(obj.retired_at or obj.created_at) <= cutoff
    ]
    stats = {"scanned": len(candidates), "deleted": 0, "errors": 0}
    if not candidates:
        return stats
    try:
        store = build_upload_store_from_env()
    except Exception:
        stats["errors"] = len(candidates)
        return stats
    for obj in candidates:
        try:
            store.delete_object(obj.object_id, sha256=obj.sha256)
            update(
                obj.object_id,
                state="deleted",
                retired_at=obj.retired_at or datetime.now(timezone.utc).isoformat(),
            )
            stats["deleted"] += 1
        except Exception:
            stats["errors"] += 1
    return stats


def _timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _nonnegative_int_env(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value
