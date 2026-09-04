from __future__ import annotations

import threading
from dataclasses import asdict
from types import MethodType
from typing import Any, Optional

from .models import UploadedObject


def ensure_upload_repository(repo: Any) -> Any:
    """Install the optional uploaded-object lifecycle capability on a repository.

    Ragbot keeps the base Repo surface stable while layering server-managed
    upload lifecycle operations onto both the in-memory development repository
    and the production PostgreSQL repository.
    """
    if callable(getattr(repo, "add_uploaded_object", None)):
        return repo
    if hasattr(repo, "_pool"):
        repo.add_uploaded_object = MethodType(_pg_add, repo)
        repo.get_uploaded_object = MethodType(_pg_get, repo)
        repo.list_uploaded_objects = MethodType(_pg_list, repo)
        repo.update_uploaded_object = MethodType(_pg_update, repo)
        return repo

    if not hasattr(repo, "_uploaded_objects"):
        repo._uploaded_objects = {}
    if not hasattr(repo, "_upload_lock"):
        repo._upload_lock = threading.Lock()
    repo.add_uploaded_object = MethodType(_mem_add, repo)
    repo.get_uploaded_object = MethodType(_mem_get, repo)
    repo.list_uploaded_objects = MethodType(_mem_list, repo)
    repo.update_uploaded_object = MethodType(_mem_update, repo)
    return repo


def _mem_add(self, obj: UploadedObject) -> None:
    with self._upload_lock:
        self._uploaded_objects[obj.object_id] = obj


def _mem_get(self, object_id: str) -> Optional[UploadedObject]:
    with self._upload_lock:
        return self._uploaded_objects.get(object_id)


def _mem_list(self, tenant_id: str | None = None, state: str | None = None) -> list[UploadedObject]:
    with self._upload_lock:
        objects = list(self._uploaded_objects.values())
    if tenant_id is not None:
        objects = [obj for obj in objects if obj.tenant_id == tenant_id]
    if state is not None:
        objects = [obj for obj in objects if obj.state == state]
    return objects


def _mem_update(self, object_id: str, **kwargs: Any) -> Optional[UploadedObject]:
    with self._upload_lock:
        obj = self._uploaded_objects.get(object_id)
        if obj is None:
            return None
        for key, value in kwargs.items():
            if not hasattr(obj, key):
                raise ValueError(f"Unsupported uploaded object field: {key}")
            setattr(obj, key, value)
        return obj


def _pg_add(self, obj: UploadedObject) -> None:
    params = asdict(obj)
    with self._pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO uploaded_objects (
                object_id, tenant_id, sha256, storage_backend, storage_key,
                size_bytes, media_type, original_filename, state, ref_count,
                created_at, last_referenced_at, retired_at
            ) VALUES (
                %(object_id)s, %(tenant_id)s, %(sha256)s, %(storage_backend)s,
                %(storage_key)s, %(size_bytes)s, %(media_type)s,
                %(original_filename)s, %(state)s, %(ref_count)s,
                COALESCE(%(created_at)s, NOW()), %(last_referenced_at)s, %(retired_at)s
            )
            ON CONFLICT (object_id) DO UPDATE SET
                state = EXCLUDED.state,
                ref_count = EXCLUDED.ref_count,
                last_referenced_at = EXCLUDED.last_referenced_at,
                retired_at = EXCLUDED.retired_at
            """,
            params,
        )


def _pg_get(self, object_id: str) -> Optional[UploadedObject]:
    with self._pool.connection() as conn:
        row = conn.execute(
            """
            SELECT object_id, tenant_id, sha256, storage_backend, storage_key,
                   size_bytes, media_type, original_filename, state, ref_count,
                   created_at, last_referenced_at, retired_at
            FROM uploaded_objects WHERE object_id = %s
            """,
            (object_id,),
        ).fetchone()
    return _row_to_uploaded_object(row) if row else None


def _pg_list(self, tenant_id: str | None = None, state: str | None = None) -> list[UploadedObject]:
    where: list[str] = []
    params: list[Any] = []
    if tenant_id is not None:
        where.append("tenant_id = %s")
        params.append(tenant_id)
    if state is not None:
        where.append("state = %s")
        params.append(state)
    sql = """
        SELECT object_id, tenant_id, sha256, storage_backend, storage_key,
               size_bytes, media_type, original_filename, state, ref_count,
               created_at, last_referenced_at, retired_at
        FROM uploaded_objects
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at"
    with self._pool.connection() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return [_row_to_uploaded_object(row) for row in rows]


def _pg_update(self, object_id: str, **kwargs: Any) -> Optional[UploadedObject]:
    allowed = {"state", "ref_count", "last_referenced_at", "retired_at"}
    unknown = set(kwargs) - allowed
    if unknown:
        raise ValueError(f"Unsupported uploaded object fields: {sorted(unknown)}")
    if not kwargs:
        return _pg_get(self, object_id)
    clauses = [f"{key} = %({key})s" for key in kwargs]
    params = {"object_id": object_id, **kwargs}
    with self._pool.connection() as conn:
        conn.execute(
            f"UPDATE uploaded_objects SET {', '.join(clauses)} WHERE object_id = %(object_id)s",
            params,
        )
    return _pg_get(self, object_id)


def _row_to_uploaded_object(row: Any) -> UploadedObject:
    if hasattr(row, "keys"):
        data = dict(row)
    elif hasattr(row, "_asdict"):
        data = row._asdict()
    else:
        columns = [
            "object_id", "tenant_id", "sha256", "storage_backend", "storage_key",
            "size_bytes", "media_type", "original_filename", "state", "ref_count",
            "created_at", "last_referenced_at", "retired_at",
        ]
        data = dict(zip(columns, row))
    return UploadedObject(**data)
