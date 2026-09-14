from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generic, Iterable, Optional, TypeVar

from .models import Document, IngestionJob, KnowledgeGeneration, Source

T = TypeVar("T")


@dataclass(frozen=True)
class KeysetPage(Generic[T]):
    items: list[T]
    next_cursor: Optional[str]
    total: int


def ensure_query_repository(repo: Any) -> Any:
    """Attach bounded/keyset query helpers to Ragbot built-in repositories.

    The historical ``list_*`` methods remain source-compatible for tests and
    internal tooling. Product HTTP/control-plane paths should prefer these
    bounded methods so cardinality growth cannot turn a request into an
    unbounded application-memory scan.
    """

    required = (
        "page_sources",
        "page_jobs",
        "page_documents",
        "page_generations",
        "documents_for_source",
        "latest_active_job",
        "source_job_summaries",
        "control_plane_overview",
    )
    if all(callable(getattr(repo, name, None)) for name in required):
        return repo

    if hasattr(repo, "_pool"):
        backend = _PostgresQueryMixin
    elif hasattr(repo, "_lock") and hasattr(repo, "_sources"):
        backend = _InMemoryQueryMixin
    else:
        return repo

    for name in required:
        if callable(getattr(repo, name, None)):
            continue
        method = getattr(backend, name)
        setattr(repo, name, method.__get__(repo, type(repo)))
    return repo


def encode_cursor(kind: str, timestamp: Any, identity: str) -> str:
    payload = {
        "v": 1,
        "k": str(kind),
        "t": _iso(timestamp),
        "i": str(identity),
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: Optional[str], kind: str) -> Optional[tuple[str, str]]:
    if not cursor:
        return None
    try:
        raw = str(cursor).strip()
        raw += "=" * (-len(raw) % 4)
        payload = json.loads(base64.urlsafe_b64decode(raw.encode("ascii")).decode("utf-8"))
    except Exception as exc:
        raise ValueError("Invalid pagination cursor") from exc
    if payload.get("v") != 1 or payload.get("k") != kind:
        raise ValueError("Pagination cursor does not match this resource")
    timestamp = str(payload.get("t") or "").strip()
    identity = str(payload.get("i") or "").strip()
    if not timestamp or not identity:
        raise ValueError("Invalid pagination cursor payload")
    return timestamp, identity


class _PostgresQueryMixin:
    def page_sources(
        self,
        *,
        tenant_ids: Optional[Iterable[str]] = None,
        status: Optional[str] = None,
        source_type: Optional[str] = None,
        q: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
        include_deleted: bool = False,
    ) -> KeysetPage[Source]:
        limit = _limit(limit)
        tenants = _tenant_list(tenant_ids)
        if tenants == []:
            return KeysetPage([], None, 0)
        conditions: list[str] = []
        params: dict[str, Any] = {"limit": limit + 1}
        if tenants is not None:
            conditions.append("tenant_id = ANY(%(tenant_ids)s)")
            params["tenant_ids"] = tenants
        if not include_deleted:
            conditions.append("status <> 'deleted'")
        if status:
            conditions.append("status = %(status)s")
            params["status"] = status
        if source_type:
            conditions.append("source_type = %(source_type)s")
            params["source_type"] = source_type
        needle = str(q or "").strip()
        if needle:
            conditions.append(
                "(name ILIKE %(q)s OR source_id ILIKE %(q)s OR source_type ILIKE %(q)s "
                "OR array_to_string(tags, ' ') ILIKE %(q)s)"
            )
            params["q"] = f"%{needle}%"

        count_where = _where(conditions)
        count_params = {key: value for key, value in params.items() if key != "limit"}
        cursor_value = decode_cursor(cursor, "sources")
        page_conditions = list(conditions)
        if cursor_value:
            page_conditions.append(
                "(created_at, source_id) < (%(cursor_time)s::timestamptz, %(cursor_id)s)"
            )
            params["cursor_time"], params["cursor_id"] = cursor_value

        with self._pool.connection() as conn:
            total = int(
                conn.execute(
                    f"SELECT COUNT(*) AS n FROM sources{count_where}", count_params
                ).fetchone()["n"]
            )
            rows = conn.execute(
                f"SELECT * FROM sources{_where(page_conditions)} "
                "ORDER BY created_at DESC, source_id DESC LIMIT %(limit)s",
                params,
            ).fetchall()
        items = [self._row_to_source(row, None) for row in rows[:limit]]
        next_cursor = None
        if len(rows) > limit and items:
            last = items[-1]
            next_cursor = encode_cursor("sources", last.created_at, last.source_id)
        return KeysetPage(items, next_cursor, total)

    def page_jobs(
        self,
        *,
        tenant_ids: Optional[Iterable[str]] = None,
        source_id: Optional[str] = None,
        status: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> KeysetPage[IngestionJob]:
        limit = _limit(limit)
        tenants = _tenant_list(tenant_ids)
        if tenants == []:
            return KeysetPage([], None, 0)
        conditions: list[str] = []
        params: dict[str, Any] = {"limit": limit + 1}
        if tenants is not None:
            conditions.append("tenant_id = ANY(%(tenant_ids)s)")
            params["tenant_ids"] = tenants
        if source_id:
            conditions.append("source_id = %(source_id)s")
            params["source_id"] = source_id
        if status:
            conditions.append("status = %(status)s")
            params["status"] = status
        count_where = _where(conditions)
        count_params = {key: value for key, value in params.items() if key != "limit"}
        cursor_value = decode_cursor(cursor, "jobs")
        page_conditions = list(conditions)
        if cursor_value:
            page_conditions.append(
                "(created_at, job_id) < (%(cursor_time)s::timestamptz, %(cursor_id)s)"
            )
            params["cursor_time"], params["cursor_id"] = cursor_value
        with self._pool.connection() as conn:
            total = int(
                conn.execute(
                    f"SELECT COUNT(*) AS n FROM ingestion_jobs{count_where}", count_params
                ).fetchone()["n"]
            )
            rows = conn.execute(
                f"SELECT * FROM ingestion_jobs{_where(page_conditions)} "
                "ORDER BY created_at DESC, job_id DESC LIMIT %(limit)s",
                params,
            ).fetchall()
        items = [self._row_to_job(row, None) for row in rows[:limit]]
        next_cursor = None
        if len(rows) > limit and items:
            last = items[-1]
            next_cursor = encode_cursor("jobs", last.created_at, last.job_id)
        return KeysetPage(items, next_cursor, total)

    def page_documents(
        self,
        *,
        tenant_ids: Optional[Iterable[str]] = None,
        source_id: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> KeysetPage[Document]:
        limit = _limit(limit)
        tenants = _tenant_list(tenant_ids)
        if tenants == []:
            return KeysetPage([], None, 0)
        conditions = ["status <> 'deleted'"]
        params: dict[str, Any] = {"limit": limit + 1}
        if tenants is not None:
            conditions.append("tenant_id = ANY(%(tenant_ids)s)")
            params["tenant_ids"] = tenants
        if source_id:
            conditions.append("source_id = %(source_id)s")
            params["source_id"] = source_id
        count_where = _where(conditions)
        count_params = {key: value for key, value in params.items() if key != "limit"}
        cursor_value = decode_cursor(cursor, "documents")
        page_conditions = list(conditions)
        if cursor_value:
            page_conditions.append(
                "(ingested_at, doc_id) < (%(cursor_time)s::timestamptz, %(cursor_id)s)"
            )
            params["cursor_time"], params["cursor_id"] = cursor_value
        with self._pool.connection() as conn:
            total = int(
                conn.execute(
                    f"SELECT COUNT(*) AS n FROM documents{count_where}", count_params
                ).fetchone()["n"]
            )
            rows = conn.execute(
                f"SELECT * FROM documents{_where(page_conditions)} "
                "ORDER BY ingested_at DESC, doc_id DESC LIMIT %(limit)s",
                params,
            ).fetchall()
        items = [self._row_to_document(row, None) for row in rows[:limit]]
        next_cursor = None
        if len(rows) > limit and items:
            last = items[-1]
            next_cursor = encode_cursor("documents", last.ingested_at, last.doc_id)
        return KeysetPage(items, next_cursor, total)

    def page_generations(
        self,
        *,
        tenant_ids: Optional[Iterable[str]] = None,
        source_id: Optional[str] = None,
        status: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> KeysetPage[KnowledgeGeneration]:
        limit = _limit(limit)
        tenants = _tenant_list(tenant_ids)
        if tenants == []:
            return KeysetPage([], None, 0)
        conditions: list[str] = []
        params: dict[str, Any] = {"limit": limit + 1}
        if tenants is not None:
            conditions.append("tenant_id = ANY(%(tenant_ids)s)")
            params["tenant_ids"] = tenants
        if source_id:
            conditions.append("source_id = %(source_id)s")
            params["source_id"] = source_id
        if status:
            conditions.append("status = %(status)s")
            params["status"] = status
        count_where = _where(conditions)
        count_params = {key: value for key, value in params.items() if key != "limit"}
        cursor_value = decode_cursor(cursor, "generations")
        page_conditions = list(conditions)
        if cursor_value:
            page_conditions.append(
                "(created_at, generation_id) < (%(cursor_time)s::timestamptz, %(cursor_id)s)"
            )
            params["cursor_time"], params["cursor_id"] = cursor_value
        with self._pool.connection() as conn:
            total = int(
                conn.execute(
                    f"SELECT COUNT(*) AS n FROM knowledge_generations{count_where}", count_params
                ).fetchone()["n"]
            )
            rows = conn.execute(
                f"SELECT * FROM knowledge_generations{_where(page_conditions)} "
                "ORDER BY created_at DESC, generation_id DESC LIMIT %(limit)s",
                params,
            ).fetchall()
        items = [_generation(row) for row in rows[:limit]]
        next_cursor = None
        if len(rows) > limit and items:
            last = items[-1]
            next_cursor = encode_cursor("generations", last.created_at, last.generation_id)
        return KeysetPage(items, next_cursor, total)

    def documents_for_source(
        self,
        source_id: str,
        tenant_id: str,
        *,
        base_doc_id: Optional[str] = None,
    ) -> list[Document]:
        base = str(base_doc_id or f"doc-{source_id}")
        params = {
            "source_id": source_id,
            "tenant_id": tenant_id,
            "uri_prefix": f"source://{source_id}%",
            "base_doc_id": base,
            "doc_prefix": f"{base}:%",
        }
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM documents
                WHERE tenant_id = %(tenant_id)s
                  AND (
                    source_id = %(source_id)s
                    OR (
                      source_id IS NULL
                      AND (
                        uri LIKE %(uri_prefix)s
                        OR doc_id = %(base_doc_id)s
                        OR doc_id LIKE %(doc_prefix)s
                      )
                    )
                  )
                ORDER BY ingested_at, doc_id
                """,
                params,
            ).fetchall()
        return [self._row_to_document(row, None) for row in rows]

    def latest_active_job(self, tenant_id: str, source_id: str) -> Optional[IngestionJob]:
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM ingestion_jobs
                WHERE tenant_id = %s AND source_id = %s
                  AND status IN ('pending', 'running')
                ORDER BY created_at DESC, job_id DESC
                LIMIT 1
                """,
                (tenant_id, source_id),
            ).fetchone()
        return self._row_to_job(row, None) if row else None

    def source_job_summaries(self, source_ids: Iterable[str]) -> dict[str, dict[str, Optional[IngestionJob]]]:
        ids = list(dict.fromkeys(str(item) for item in source_ids if item))
        if not ids:
            return {}
        result: dict[str, dict[str, Optional[IngestionJob]]] = {
            source_id: {"latest": None, "latest_completed": None} for source_id in ids
        }
        with self._pool.connection() as conn:
            latest = conn.execute(
                """
                SELECT DISTINCT ON (source_id) *
                FROM ingestion_jobs
                WHERE source_id = ANY(%s)
                ORDER BY source_id, created_at DESC, job_id DESC
                """,
                (ids,),
            ).fetchall()
            completed = conn.execute(
                """
                SELECT DISTINCT ON (source_id) *
                FROM ingestion_jobs
                WHERE source_id = ANY(%s) AND status = 'completed'
                ORDER BY source_id, created_at DESC, job_id DESC
                """,
                (ids,),
            ).fetchall()
        for row in latest:
            job = self._row_to_job(row, None)
            result.setdefault(job.source_id, {})["latest"] = job
        for row in completed:
            job = self._row_to_job(row, None)
            result.setdefault(job.source_id, {})["latest_completed"] = job
        return result

    def control_plane_overview(
        self,
        tenant_ids: Optional[Iterable[str]] = None,
    ) -> dict[str, Any]:
        tenants = _tenant_list(tenant_ids)
        if tenants == []:
            return _empty_overview_raw()
        source_where, source_params = _tenant_where(tenants, prefix="")
        job_where, job_params = _tenant_where(tenants, prefix="")
        with self._pool.connection() as conn:
            source_row = conn.execute(
                f"""
                SELECT
                    COUNT(*) FILTER (WHERE status <> 'deleted') AS total,
                    COUNT(*) FILTER (WHERE status = 'active') AS active,
                    COUNT(*) FILTER (WHERE status = 'paused') AS paused,
                    COUNT(*) FILTER (WHERE status <> 'deleted' AND sync_enabled) AS scheduled,
                    MIN(sync_next_at) FILTER (WHERE status <> 'deleted' AND sync_enabled) AS next_sync_at
                FROM sources{source_where}
                """,
                source_params,
            ).fetchone()
            job_row = conn.execute(
                f"""
                SELECT
                    COUNT(*) FILTER (WHERE status = 'pending') AS pending,
                    COUNT(*) FILTER (WHERE status = 'running') AS running,
                    COUNT(*) FILTER (WHERE status = 'failed') AS failed,
                    COUNT(*) FILTER (WHERE status = 'dead_lettered') AS dead_lettered,
                    MIN(created_at) FILTER (WHERE status = 'pending') AS oldest_pending_at,
                    COUNT(*) FILTER (
                        WHERE status = 'running' AND lease_expires_at IS NOT NULL
                          AND lease_expires_at <= NOW()
                    ) AS stale_running,
                    COUNT(*) FILTER (
                        WHERE status = 'completed' AND completed_at >= NOW() - INTERVAL '24 hours'
                    ) AS completed_24h,
                    COUNT(*) FILTER (
                        WHERE status = 'failed' AND completed_at >= NOW() - INTERVAL '24 hours'
                    ) AS failed_24h,
                    COUNT(*) FILTER (
                        WHERE status = 'dead_lettered' AND dead_lettered_at >= NOW() - INTERVAL '24 hours'
                    ) AS dead_lettered_24h
                FROM ingestion_jobs{job_where}
                """,
                job_params,
            ).fetchone()
            knowledge = conn.execute(
                f"""
                WITH latest AS (
                    SELECT DISTINCT ON (source_id)
                        source_id, doc_count, chunk_count, stats
                    FROM ingestion_jobs
                    {job_where}{' AND' if job_where else ' WHERE'} status = 'completed'
                    ORDER BY source_id, created_at DESC, job_id DESC
                )
                SELECT
                    COALESCE(SUM(doc_count), 0) AS documents,
                    COALESCE(SUM(
                        COALESCE(NULLIF(stats->>'chunks_total', '')::bigint, chunk_count, 0)
                    ), 0) AS chunks
                FROM latest
                """,
                job_params,
            ).fetchone()
            failures = conn.execute(
                f"""
                SELECT * FROM ingestion_jobs
                {job_where}{' AND' if job_where else ' WHERE'} status IN ('failed', 'dead_lettered')
                ORDER BY created_at DESC, job_id DESC
                LIMIT 10
                """,
                job_params,
            ).fetchall()
        return {
            "sources": dict(source_row or {}),
            "queue": dict(job_row or {}),
            "knowledge": dict(knowledge or {}),
            "recent_failures": [self._row_to_job(row, None) for row in failures],
        }


class _InMemoryQueryMixin:
    def page_sources(
        self,
        *,
        tenant_ids: Optional[Iterable[str]] = None,
        status: Optional[str] = None,
        source_type: Optional[str] = None,
        q: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
        include_deleted: bool = False,
    ) -> KeysetPage[Source]:
        tenants = _tenant_list(tenant_ids)
        if tenants == []:
            return KeysetPage([], None, 0)
        needle = str(q or "").strip().lower()
        with self._lock:
            items = list(self._sources.values())
        if tenants is not None:
            allowed = set(tenants)
            items = [item for item in items if item.tenant_id in allowed]
        if not include_deleted:
            items = [item for item in items if item.status != "deleted"]
        if status:
            items = [item for item in items if item.status == status]
        if source_type:
            items = [item for item in items if item.source_type == source_type]
        if needle:
            items = [
                item for item in items
                if needle in " ".join(
                    [item.name, item.source_id, item.source_type, *item.tags]
                ).lower()
            ]
        return _memory_page(items, "sources", "created_at", "source_id", cursor, limit)

    def page_jobs(
        self,
        *,
        tenant_ids: Optional[Iterable[str]] = None,
        source_id: Optional[str] = None,
        status: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> KeysetPage[IngestionJob]:
        tenants = _tenant_list(tenant_ids)
        if tenants == []:
            return KeysetPage([], None, 0)
        with self._lock:
            items = list(self._jobs.values())
        if tenants is not None:
            allowed = set(tenants)
            items = [item for item in items if item.tenant_id in allowed]
        if source_id:
            items = [item for item in items if item.source_id == source_id]
        if status:
            items = [item for item in items if item.status == status]
        return _memory_page(items, "jobs", "created_at", "job_id", cursor, limit)

    def page_documents(
        self,
        *,
        tenant_ids: Optional[Iterable[str]] = None,
        source_id: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> KeysetPage[Document]:
        tenants = _tenant_list(tenant_ids)
        if tenants == []:
            return KeysetPage([], None, 0)
        with self._lock:
            items = list(self._documents.values())
        if tenants is not None:
            allowed = set(tenants)
            items = [item for item in items if item.tenant_id in allowed]
        items = [item for item in items if item.status != "deleted"]
        if source_id:
            items = [item for item in items if item.source_id == source_id]
        return _memory_page(items, "documents", "ingested_at", "doc_id", cursor, limit)

    def page_generations(
        self,
        *,
        tenant_ids: Optional[Iterable[str]] = None,
        source_id: Optional[str] = None,
        status: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> KeysetPage[KnowledgeGeneration]:
        tenants = _tenant_list(tenant_ids)
        if tenants == []:
            return KeysetPage([], None, 0)
        ensure = getattr(self, "_ensure_generation_state", None)
        if callable(ensure):
            ensure()
        with self._lock:
            items = list(getattr(self, "_knowledge_generations", {}).values())
        if tenants is not None:
            allowed = set(tenants)
            items = [item for item in items if item.tenant_id in allowed]
        if source_id:
            items = [item for item in items if item.source_id == source_id]
        if status:
            items = [item for item in items if item.status == status]
        return _memory_page(items, "generations", "created_at", "generation_id", cursor, limit)

    def documents_for_source(
        self,
        source_id: str,
        tenant_id: str,
        *,
        base_doc_id: Optional[str] = None,
    ) -> list[Document]:
        base = str(base_doc_id or f"doc-{source_id}")
        with self._lock:
            docs = [
                doc for doc in self._documents.values()
                if doc.tenant_id == tenant_id
                and (
                    doc.source_id == source_id
                    or (
                        doc.source_id is None
                        and (
                            (doc.uri or "").startswith(f"source://{source_id}")
                            or doc.doc_id == base
                            or doc.doc_id.startswith(f"{base}:")
                        )
                    )
                )
            ]
        return sorted(docs, key=lambda item: (_time(item.ingested_at), item.doc_id))

    def latest_active_job(self, tenant_id: str, source_id: str) -> Optional[IngestionJob]:
        with self._lock:
            jobs = [
                job for job in self._jobs.values()
                if job.tenant_id == tenant_id
                and job.source_id == source_id
                and job.status in {"pending", "running"}
            ]
        if not jobs:
            return None
        return max(jobs, key=lambda item: (_time(item.created_at), item.job_id))

    def source_job_summaries(self, source_ids: Iterable[str]) -> dict[str, dict[str, Optional[IngestionJob]]]:
        ids = set(str(item) for item in source_ids if item)
        result: dict[str, dict[str, Optional[IngestionJob]]] = {
            source_id: {"latest": None, "latest_completed": None} for source_id in ids
        }
        with self._lock:
            jobs = [job for job in self._jobs.values() if job.source_id in ids]
        for source_id in ids:
            source_jobs = sorted(
                [job for job in jobs if job.source_id == source_id],
                key=lambda item: (_time(item.created_at), item.job_id),
                reverse=True,
            )
            if source_jobs:
                result[source_id]["latest"] = source_jobs[0]
            result[source_id]["latest_completed"] = next(
                (job for job in source_jobs if job.status == "completed"), None
            )
        return result

    def control_plane_overview(
        self,
        tenant_ids: Optional[Iterable[str]] = None,
    ) -> dict[str, Any]:
        tenants = _tenant_list(tenant_ids)
        if tenants == []:
            return _empty_overview_raw()
        allowed = set(tenants) if tenants is not None else None
        with self._lock:
            sources = list(self._sources.values())
            jobs = list(self._jobs.values())
        if allowed is not None:
            sources = [source for source in sources if source.tenant_id in allowed]
            jobs = [job for job in jobs if job.tenant_id in allowed]
        sources = [source for source in sources if source.status != "deleted"]
        now = datetime.now(timezone.utc)
        pending = [job for job in jobs if job.status == "pending"]
        running = [job for job in jobs if job.status == "running"]
        completed = [job for job in jobs if job.status == "completed"]
        failed = [job for job in jobs if job.status == "failed"]
        dead = [job for job in jobs if job.status == "dead_lettered"]
        scheduled = [source for source in sources if source.sync_enabled]
        latest_completed: dict[str, IngestionJob] = {}
        for job in sorted(completed, key=lambda item: (_time(item.created_at), item.job_id), reverse=True):
            latest_completed.setdefault(job.source_id, job)
        return {
            "sources": {
                "total": len(sources),
                "active": sum(1 for source in sources if source.status == "active"),
                "paused": sum(1 for source in sources if source.status == "paused"),
                "scheduled": len(scheduled),
                "next_sync_at": min(
                    (_time(source.sync_next_at) for source in scheduled if source.sync_next_at),
                    default=None,
                ),
            },
            "queue": {
                "pending": len(pending),
                "running": len(running),
                "failed": len(failed),
                "dead_lettered": len(dead),
                "oldest_pending_at": min(
                    (_time(job.created_at or job.available_at) for job in pending), default=None
                ),
                "stale_running": sum(
                    1 for job in running
                    if job.lease_expires_at and _time(job.lease_expires_at) <= now
                ),
                "completed_24h": sum(
                    1 for job in completed
                    if job.completed_at and (now - _time(job.completed_at)).total_seconds() <= 86400
                ),
                "failed_24h": sum(
                    1 for job in failed
                    if job.completed_at and (now - _time(job.completed_at)).total_seconds() <= 86400
                ),
                "dead_lettered_24h": sum(
                    1 for job in dead
                    if job.dead_lettered_at and (now - _time(job.dead_lettered_at)).total_seconds() <= 86400
                ),
            },
            "knowledge": {
                "documents": sum(int(job.doc_count or 0) for job in latest_completed.values()),
                "chunks": sum(
                    int((job.stats or {}).get("chunks_total", job.chunk_count or 0))
                    for job in latest_completed.values()
                ),
            },
            "recent_failures": sorted(
                [*failed, *dead],
                key=lambda item: (_time(item.created_at), item.job_id),
                reverse=True,
            )[:10],
        }


def _memory_page(
    items: list[T],
    kind: str,
    time_attr: str,
    id_attr: str,
    cursor: Optional[str],
    limit: int,
) -> KeysetPage[T]:
    limit = _limit(limit)
    items = sorted(
        items,
        key=lambda item: (_time(getattr(item, time_attr, None)), str(getattr(item, id_attr))),
        reverse=True,
    )
    total = len(items)
    cursor_value = decode_cursor(cursor, kind)
    if cursor_value:
        cursor_time = _time(cursor_value[0])
        cursor_id = cursor_value[1]
        items = [
            item for item in items
            if (_time(getattr(item, time_attr, None)), str(getattr(item, id_attr)))
            < (cursor_time, cursor_id)
        ]
    page_items = items[:limit]
    next_cursor = None
    if len(items) > limit and page_items:
        last = page_items[-1]
        next_cursor = encode_cursor(
            kind, getattr(last, time_attr, None), str(getattr(last, id_attr))
        )
    return KeysetPage(page_items, next_cursor, total)


def _generation(row: Any) -> KnowledgeGeneration:
    data = dict(row)
    stats = data.get("stats") or {}
    if isinstance(stats, str):
        stats = json.loads(stats)
    return KnowledgeGeneration(
        generation_id=str(data["generation_id"]),
        source_id=str(data["source_id"]),
        tenant_id=str(data["tenant_id"]),
        job_id=data.get("job_id"),
        status=str(data.get("status") or "staging"),
        created_at=_iso(data.get("created_at")),
        prepared_at=_iso(data.get("prepared_at")),
        activated_at=_iso(data.get("activated_at")),
        retired_at=_iso(data.get("retired_at")),
        failed_at=_iso(data.get("failed_at")),
        error=data.get("error"),
        stats=dict(stats),
    )


def _empty_overview_raw() -> dict[str, Any]:
    return {
        "sources": {
            "total": 0,
            "active": 0,
            "paused": 0,
            "scheduled": 0,
            "next_sync_at": None,
        },
        "queue": {
            "pending": 0,
            "running": 0,
            "failed": 0,
            "dead_lettered": 0,
            "oldest_pending_at": None,
            "stale_running": 0,
            "completed_24h": 0,
            "failed_24h": 0,
            "dead_lettered_24h": 0,
        },
        "knowledge": {"documents": 0, "chunks": 0},
        "recent_failures": [],
    }


def _tenant_where(tenants: Optional[list[str]], prefix: str = "") -> tuple[str, dict[str, Any]]:
    if tenants is None:
        return "", {}
    column = f"{prefix}tenant_id" if prefix else "tenant_id"
    return f" WHERE {column} = ANY(%(tenant_ids)s)", {"tenant_ids": tenants}


def _tenant_list(values: Optional[Iterable[str]]) -> Optional[list[str]]:
    if values is None:
        return None
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _where(conditions: list[str]) -> str:
    return f" WHERE {' AND '.join(conditions)}" if conditions else ""


def _limit(value: int) -> int:
    return max(1, min(500, int(value)))


def _iso(value: Any) -> str:
    if value is None:
        return datetime(1970, 1, 1, tzinfo=timezone.utc).isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    text = str(value)
    return _time(text).isoformat()


def _time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if not value:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
