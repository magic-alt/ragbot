from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Optional


def _canonical_timestamp(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value).strip()
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return raw
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _source_token(source_id: str, updated_at: Any, created_at: Any) -> str:
    updated = _canonical_timestamp(updated_at)
    created = _canonical_timestamp(created_at)
    return updated or created or f"legacy:{source_id}"


def _assert_activation_source(
    *,
    source_id: str,
    tenant_id: str,
    status: str,
    updated_at: Any,
    created_at: Any,
    expected_source_generation: Optional[str],
    generation_tenant_id: str,
) -> None:
    if tenant_id != generation_tenant_id:
        raise RuntimeError(f"Source tenant changed during ingestion: {source_id}")
    if status == "deleted":
        raise RuntimeError(f"Source is deleted during ingestion: {source_id}")
    if expected_source_generation:
        expected = _canonical_timestamp(expected_source_generation) or str(expected_source_generation)
        actual = _source_token(source_id, updated_at, created_at)
        if actual != expected:
            raise RuntimeError(
                "Source lifecycle generation changed during ingestion: "
                f"source={source_id} expected={expected} actual={actual}"
            )


def activate_postgres_generation(
    repo: Any,
    source_id: str,
    generation_id: str,
    cleanup_point_ids: Iterable[str] = (),
    previous_doc_ids: Iterable[str] = (),
    expected_source_generation: Optional[str] = None,
) -> Optional[str]:
    """Activate one prepared generation with the Source fence in the same PG tx."""
    previous_ids = list(dict.fromkeys(str(item) for item in previous_doc_ids if item))
    cleanup = list(dict.fromkeys(str(item) for item in cleanup_point_ids if item))
    with repo._pool.connection() as conn:
        with conn.transaction():
            generation = conn.execute(
                """
                SELECT source_id, tenant_id, status, stats
                FROM knowledge_generations
                WHERE generation_id = %s
                FOR UPDATE
                """,
                (generation_id,),
            ).fetchone()
            if not generation:
                raise ValueError(f"Unknown knowledge generation: {generation_id}")
            generation_source = str(generation["source_id"])
            generation_tenant = str(generation["tenant_id"])
            generation_status = str(generation["status"])
            generation_stats = dict(generation.get("stats") or {})
            durable_expected = expected_source_generation or generation_stats.get("source_generation")
            if generation_source != source_id:
                raise ValueError(
                    f"Generation/source mismatch: generation={generation_id} source={source_id}"
                )
            if generation_status != "prepared":
                raise ValueError(
                    f"Generation {generation_id} must be prepared before activation; status={generation_status}"
                )

            source_row = conn.execute(
                """
                SELECT source_id, tenant_id, status, updated_at, created_at
                FROM sources
                WHERE source_id = %s
                FOR UPDATE
                """,
                (source_id,),
            ).fetchone()
            if not source_row:
                raise RuntimeError(f"Source disappeared during ingestion: {source_id}")
            _assert_activation_source(
                source_id=source_id,
                tenant_id=str(source_row["tenant_id"]),
                status=str(source_row["status"]),
                updated_at=source_row.get("updated_at"),
                created_at=source_row.get("created_at"),
                expected_source_generation=durable_expected,
                generation_tenant_id=generation_tenant,
            )

            active = conn.execute(
                """
                SELECT generation_id
                FROM source_active_generations
                WHERE source_id = %s
                FOR UPDATE
                """,
                (source_id,),
            ).fetchone()
            old_generation_id = str(active["generation_id"]) if active else None

            if previous_ids:
                conn.execute(
                    "DELETE FROM documents WHERE source_id = %s OR doc_id = ANY(%s)",
                    (source_id, previous_ids),
                )
            else:
                conn.execute("DELETE FROM documents WHERE source_id = %s", (source_id,))

            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, tenant_id, source_type, title, uri, version,
                    doc_updated_at, ingested_at, tags, acl_policy_id, status,
                    source_id, generation_id
                )
                SELECT
                    doc_id, tenant_id, source_type, title, uri, version,
                    doc_updated_at, ingested_at, tags, acl_policy_id, status,
                    source_id, generation_id
                FROM staged_documents
                WHERE generation_id = %s
                """,
                (generation_id,),
            )
            conn.execute(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, tenant_id, chunk_index, text,
                    path, url, page, section, checksum, qdrant_point_id,
                    created_at, metadata, fts_text, source_id, generation_id
                )
                SELECT
                    chunk_id, doc_id, tenant_id, chunk_index, text,
                    path, url, page, section, checksum, qdrant_point_id,
                    created_at, metadata, fts_text, source_id, generation_id
                FROM staged_chunks
                WHERE generation_id = %s
                """,
                (generation_id,),
            )

            conn.execute(
                """
                INSERT INTO source_active_generations(source_id, generation_id, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (source_id) DO UPDATE SET
                    generation_id = EXCLUDED.generation_id,
                    updated_at = EXCLUDED.updated_at
                """,
                (source_id, generation_id),
            )
            conn.execute(
                """
                UPDATE knowledge_generations
                SET status = 'active', activated_at = NOW(), error = NULL
                WHERE generation_id = %s
                """,
                (generation_id,),
            )
            if old_generation_id and old_generation_id != generation_id:
                conn.execute(
                    """
                    UPDATE knowledge_generations
                    SET status = 'retired', retired_at = NOW()
                    WHERE generation_id = %s AND status = 'active'
                    """,
                    (old_generation_id,),
                )

            repo._enqueue_cleanup_events(
                conn,
                source_id=source_id,
                generation_id=old_generation_id,
                point_ids=cleanup,
            )
            conn.execute("DELETE FROM staged_documents WHERE generation_id = %s", (generation_id,))
            return old_generation_id


def activate_inmemory_generation(
    repo: Any,
    source_id: str,
    generation_id: str,
    cleanup_point_ids: Iterable[str] = (),
    previous_doc_ids: Iterable[str] = (),
    expected_source_generation: Optional[str] = None,
) -> Optional[str]:
    """In-memory equivalent of the fenced atomic activation transaction."""
    with repo._lock:
        repo._ensure_generation_state()
        generation = repo._knowledge_generations.get(generation_id)
        if generation is None or generation.status != "prepared" or generation.source_id != source_id:
            raise ValueError(f"Generation is not activatable: {generation_id}")
        source = repo._sources.get(source_id)
        if source is None:
            raise RuntimeError(f"Source disappeared during ingestion: {source_id}")
        durable_expected = expected_source_generation or (generation.stats or {}).get("source_generation")
        _assert_activation_source(
            source_id=source_id,
            tenant_id=str(source.tenant_id),
            status=str(source.status),
            updated_at=source.updated_at,
            created_at=source.created_at,
            expected_source_generation=durable_expected,
            generation_tenant_id=str(generation.tenant_id),
        )

        old_generation_id = repo._active_generations.get(source_id)
        previous = set(str(item) for item in previous_doc_ids if item)
        previous.update(
            doc_id
            for doc_id, doc in repo._documents.items()
            if getattr(doc, "source_id", None) == source_id
        )
        for doc_id in previous:
            repo._documents.pop(doc_id, None)
        for chunk_id in [cid for cid, chunk in repo._chunks.items() if chunk.doc_id in previous]:
            repo._chunks.pop(chunk_id, None)
        for doc in repo._staged_documents.get(generation_id, {}).values():
            repo._documents[doc.doc_id] = doc
        for chunk in repo._staged_chunks.get(generation_id, {}).values():
            repo._chunks[chunk.chunk_id] = chunk

        repo._active_generations[source_id] = generation_id
        generation.status = "active"
        generation.activated_at = datetime.now(timezone.utc).isoformat()
        if old_generation_id and old_generation_id != generation_id:
            old = repo._knowledge_generations.get(old_generation_id)
            if old is not None:
                old.status = "retired"
                old.retired_at = datetime.now(timezone.utc).isoformat()
        repo._enqueue_memory_cleanup(source_id, old_generation_id, cleanup_point_ids)
        repo._staged_documents.pop(generation_id, None)
        repo._staged_chunks.pop(generation_id, None)
        return old_generation_id
