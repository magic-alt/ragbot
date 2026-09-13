from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

from .models import IndexVersion

_INDEX_METHODS = (
    "add_index_version",
    "get_index_version",
    "get_index_version_by_collection",
    "get_active_index_version",
    "list_index_versions",
    "update_index_version",
    "activate_index_version",
    "list_prunable_index_versions",
)

_ALLOWED_UPDATE_FIELDS = {
    "status",
    "build_stats",
    "validation_evidence",
    "validating_at",
    "ready_at",
    "activated_at",
    "retired_at",
    "failed_at",
    "delete_after",
    "deleted_at",
    "error",
    "parser_contracts",
    "chunking_contracts",
}
_JSON_FIELDS = {
    "embedding_spec",
    "vector_schema",
    "parser_contracts",
    "chunking_contracts",
    "build_stats",
    "validation_evidence",
}


def ensure_index_repository(repo: Any) -> Any:
    """Attach IndexLifecycleRepo methods to Ragbot's built-in repositories.

    The lifecycle capability is deliberately additive, matching GenerationRepo:
    third-party repositories are not forced to pretend they persist index state.
    """
    if all(callable(getattr(repo, name, None)) for name in _INDEX_METHODS):
        return repo

    if hasattr(repo, "_pool"):
        backend = _PostgresIndexRepo
    elif hasattr(repo, "_lock") and hasattr(repo, "_documents"):
        backend = _InMemoryIndexRepo
    else:
        return repo

    for name in _INDEX_METHODS:
        method = getattr(backend, name)
        setattr(repo, name, method.__get__(repo, type(repo)))
    if backend is _PostgresIndexRepo:
        setattr(repo, "_row_to_index_version", _PostgresIndexRepo._row_to_index_version)
    else:
        helper = getattr(backend, "_ensure_index_state")
        setattr(repo, "_ensure_index_state", helper.__get__(repo, type(repo)))
    return repo


def supports_index_lifecycle(repo: Any) -> bool:
    ensure_index_repository(repo)
    return all(callable(getattr(repo, name, None)) for name in _INDEX_METHODS)


class _InMemoryIndexRepo:
    def _ensure_index_state(self) -> None:
        if not hasattr(self, "_index_versions"):
            self._index_versions: dict[str, IndexVersion] = {}

    def add_index_version(self, version: IndexVersion) -> None:
        with self._lock:
            self._ensure_index_state()
            if version.index_version_id in self._index_versions:
                raise ValueError(f"Index version already exists: {version.index_version_id}")
            if any(
                item.physical_collection == version.physical_collection
                for item in self._index_versions.values()
            ):
                raise ValueError(f"Physical collection already registered: {version.physical_collection}")
            self._index_versions[version.index_version_id] = version

    def get_index_version(self, index_version_id: str) -> Optional[IndexVersion]:
        with self._lock:
            self._ensure_index_state()
            return self._index_versions.get(index_version_id)

    def get_index_version_by_collection(
        self, alias_name: str, physical_collection: str
    ) -> Optional[IndexVersion]:
        with self._lock:
            self._ensure_index_state()
            return next(
                (
                    item
                    for item in self._index_versions.values()
                    if item.alias_name == alias_name
                    and item.physical_collection == physical_collection
                ),
                None,
            )

    def get_active_index_version(
        self, alias_name: str, tenant_id: Optional[str] = None, scope_key: str = "global"
    ) -> Optional[IndexVersion]:
        with self._lock:
            self._ensure_index_state()
            return next(
                (
                    item
                    for item in self._index_versions.values()
                    if item.alias_name == alias_name
                    and item.tenant_id == tenant_id
                    and item.scope_key == scope_key
                    and item.status == "active"
                ),
                None,
            )

    def list_index_versions(
        self,
        alias_name: Optional[str] = None,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[IndexVersion]:
        with self._lock:
            self._ensure_index_state()
            items = list(self._index_versions.values())
            if alias_name is not None:
                items = [item for item in items if item.alias_name == alias_name]
            if tenant_id is not None:
                items = [item for item in items if item.tenant_id == tenant_id]
            if status is not None:
                items = [item for item in items if item.status == status]
            return sorted(items, key=lambda item: item.created_at or "", reverse=True)

    def update_index_version(self, index_version_id: str, **kwargs: Any) -> Optional[IndexVersion]:
        unknown = set(kwargs) - _ALLOWED_UPDATE_FIELDS
        if unknown:
            raise ValueError(f"Unsupported index version fields: {sorted(unknown)}")
        with self._lock:
            self._ensure_index_state()
            item = self._index_versions.get(index_version_id)
            if item is None:
                return None
            for key, value in kwargs.items():
                setattr(item, key, value)
            return item

    def activate_index_version(
        self,
        index_version_id: str,
        *,
        previous_index_version_id: Optional[str] = None,
        delete_after: Optional[str] = None,
    ) -> IndexVersion:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._ensure_index_state()
            target = self._index_versions.get(index_version_id)
            if target is None:
                raise ValueError(f"Unknown index version: {index_version_id}")
            if target.status not in {"ready", "retired", "active"}:
                raise ValueError(f"Index version is not activatable: {index_version_id} ({target.status})")
            active = next(
                (
                    item
                    for item in self._index_versions.values()
                    if item.index_version_id != index_version_id
                    and item.alias_name == target.alias_name
                    and item.tenant_id == target.tenant_id
                    and item.scope_key == target.scope_key
                    and item.status == "active"
                ),
                None,
            )
            if previous_index_version_id is not None:
                actual = active.index_version_id if active else None
                if actual != previous_index_version_id:
                    raise RuntimeError(
                        f"Active index changed during activation: expected={previous_index_version_id}, actual={actual}"
                    )
            if active is not None:
                active.status = "retired"
                active.retired_at = now
                active.delete_after = delete_after
            target.status = "active"
            target.activated_at = now
            target.retired_at = None
            target.delete_after = None
            target.error = None
            return target

    def list_prunable_index_versions(self, now_iso: str) -> list[IndexVersion]:
        now = _parse_time(now_iso)
        with self._lock:
            self._ensure_index_state()
            return [
                item
                for item in self._index_versions.values()
                if item.status in {"retired", "failed"}
                and item.delete_after
                and _parse_time(item.delete_after) <= now
            ]


class _PostgresIndexRepo:
    def add_index_version(self, version: IndexVersion) -> None:
        params = asdict(version)
        for key in _JSON_FIELDS:
            params[key] = self._jsonb(params[key])
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO vector_index_versions (
                    index_version_id, tenant_id, scope_key, vector_backend,
                    alias_name, physical_collection, embedding_contract_id,
                    embedding_spec, vector_schema, parser_contracts,
                    chunking_contracts, status, build_stats,
                    validation_evidence, created_at, validating_at, ready_at,
                    activated_at, retired_at, failed_at, delete_after,
                    deleted_at, error
                ) VALUES (
                    %(index_version_id)s, %(tenant_id)s, %(scope_key)s,
                    %(vector_backend)s, %(alias_name)s, %(physical_collection)s,
                    %(embedding_contract_id)s, %(embedding_spec)s,
                    %(vector_schema)s, %(parser_contracts)s,
                    %(chunking_contracts)s, %(status)s, %(build_stats)s,
                    %(validation_evidence)s, COALESCE(%(created_at)s, NOW()),
                    %(validating_at)s, %(ready_at)s, %(activated_at)s,
                    %(retired_at)s, %(failed_at)s, %(delete_after)s,
                    %(deleted_at)s, %(error)s
                )
                """,
                params,
            )

    def get_index_version(self, index_version_id: str) -> Optional[IndexVersion]:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM vector_index_versions WHERE index_version_id = %s",
                (index_version_id,),
            ).fetchone()
        return self._row_to_index_version(row) if row else None

    def get_index_version_by_collection(
        self, alias_name: str, physical_collection: str
    ) -> Optional[IndexVersion]:
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM vector_index_versions
                WHERE alias_name = %s AND physical_collection = %s
                """,
                (alias_name, physical_collection),
            ).fetchone()
        return self._row_to_index_version(row) if row else None

    def get_active_index_version(
        self, alias_name: str, tenant_id: Optional[str] = None, scope_key: str = "global"
    ) -> Optional[IndexVersion]:
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM vector_index_versions
                WHERE alias_name = %(alias_name)s
                  AND tenant_id IS NOT DISTINCT FROM %(tenant_id)s
                  AND scope_key = %(scope_key)s
                  AND status = 'active'
                ORDER BY activated_at DESC NULLS LAST, created_at DESC
                LIMIT 1
                """,
                {"alias_name": alias_name, "tenant_id": tenant_id, "scope_key": scope_key},
            ).fetchone()
        return self._row_to_index_version(row) if row else None

    def list_index_versions(
        self,
        alias_name: Optional[str] = None,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[IndexVersion]:
        conditions: list[str] = []
        params: list[Any] = []
        if alias_name is not None:
            conditions.append("alias_name = %s")
            params.append(alias_name)
        if tenant_id is not None:
            conditions.append("tenant_id = %s")
            params.append(tenant_id)
        if status is not None:
            conditions.append("status = %s")
            params.append(status)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM vector_index_versions{where} ORDER BY created_at DESC",
                tuple(params),
            ).fetchall()
        return [self._row_to_index_version(row) for row in rows]

    def update_index_version(self, index_version_id: str, **kwargs: Any) -> Optional[IndexVersion]:
        unknown = set(kwargs) - _ALLOWED_UPDATE_FIELDS
        if unknown:
            raise ValueError(f"Unsupported index version fields: {sorted(unknown)}")
        if not kwargs:
            return self.get_index_version(index_version_id)
        clauses: list[str] = []
        params: dict[str, Any] = {"index_version_id": index_version_id}
        for key, value in kwargs.items():
            clauses.append(f"{key} = %({key})s")
            params[key] = self._jsonb(value) if key in _JSON_FIELDS else value
        with self._pool.connection() as conn:
            conn.execute(
                f"UPDATE vector_index_versions SET {', '.join(clauses)} WHERE index_version_id = %(index_version_id)s",
                params,
            )
        return self.get_index_version(index_version_id)

    def activate_index_version(
        self,
        index_version_id: str,
        *,
        previous_index_version_id: Optional[str] = None,
        delete_after: Optional[str] = None,
    ) -> IndexVersion:
        with self._pool.connection() as conn:
            with conn.transaction():
                target = conn.execute(
                    "SELECT * FROM vector_index_versions WHERE index_version_id = %s FOR UPDATE",
                    (index_version_id,),
                ).fetchone()
                if not target:
                    raise ValueError(f"Unknown index version: {index_version_id}")
                target_status = str(target["status"])
                if target_status not in {"ready", "retired", "active"}:
                    raise ValueError(
                        f"Index version is not activatable: {index_version_id} ({target_status})"
                    )
                active = conn.execute(
                    """
                    SELECT * FROM vector_index_versions
                    WHERE alias_name = %(alias_name)s
                      AND tenant_id IS NOT DISTINCT FROM %(tenant_id)s
                      AND scope_key = %(scope_key)s
                      AND status = 'active'
                      AND index_version_id <> %(index_version_id)s
                    FOR UPDATE
                    """,
                    {
                        "alias_name": target["alias_name"],
                        "tenant_id": target["tenant_id"],
                        "scope_key": target["scope_key"],
                        "index_version_id": index_version_id,
                    },
                ).fetchone()
                actual_previous = str(active["index_version_id"]) if active else None
                if previous_index_version_id is not None and actual_previous != previous_index_version_id:
                    raise RuntimeError(
                        "Active index changed during activation: "
                        f"expected={previous_index_version_id}, actual={actual_previous}"
                    )
                if active:
                    conn.execute(
                        """
                        UPDATE vector_index_versions
                        SET status = 'retired', retired_at = NOW(), delete_after = %s
                        WHERE index_version_id = %s
                        """,
                        (delete_after, actual_previous),
                    )
                conn.execute(
                    """
                    UPDATE vector_index_versions
                    SET status = 'active', activated_at = NOW(), retired_at = NULL,
                        delete_after = NULL, error = NULL
                    WHERE index_version_id = %s
                    """,
                    (index_version_id,),
                )
        result = self.get_index_version(index_version_id)
        if result is None:  # pragma: no cover
            raise RuntimeError(f"Activated index disappeared: {index_version_id}")
        return result

    def list_prunable_index_versions(self, now_iso: str) -> list[IndexVersion]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM vector_index_versions
                WHERE status IN ('retired', 'failed')
                  AND delete_after IS NOT NULL
                  AND delete_after <= %s::timestamptz
                ORDER BY delete_after
                """,
                (now_iso,),
            ).fetchall()
        return [self._row_to_index_version(row) for row in rows]

    @staticmethod
    def _row_to_index_version(row: Any) -> IndexVersion:
        data = dict(row)
        for key in _JSON_FIELDS:
            value = data.get(key)
            if isinstance(value, str):
                data[key] = json.loads(value)
        return IndexVersion(
            index_version_id=str(data["index_version_id"]),
            tenant_id=data.get("tenant_id"),
            scope_key=str(data.get("scope_key") or "global"),
            vector_backend=str(data.get("vector_backend") or "qdrant"),
            alias_name=str(data["alias_name"]),
            physical_collection=str(data["physical_collection"]),
            embedding_contract_id=str(data["embedding_contract_id"]),
            embedding_spec=dict(data.get("embedding_spec") or {}),
            vector_schema=dict(data.get("vector_schema") or {}),
            parser_contracts=list(data.get("parser_contracts") or []),
            chunking_contracts=list(data.get("chunking_contracts") or []),
            status=str(data.get("status") or "building"),
            build_stats=dict(data.get("build_stats") or {}),
            validation_evidence=dict(data.get("validation_evidence") or {}),
            created_at=_iso(data.get("created_at")),
            validating_at=_iso(data.get("validating_at")),
            ready_at=_iso(data.get("ready_at")),
            activated_at=_iso(data.get("activated_at")),
            retired_at=_iso(data.get("retired_at")),
            failed_at=_iso(data.get("failed_at")),
            delete_after=_iso(data.get("delete_after")),
            deleted_at=_iso(data.get("deleted_at")),
            error=data.get("error"),
        )


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
