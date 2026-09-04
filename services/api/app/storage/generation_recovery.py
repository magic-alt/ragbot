from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def fail_postgres_generation(
    repo: Any,
    generation_id: str,
    error: str,
    cleanup_point_ids: Iterable[str] = (),
) -> None:
    """Fail a candidate and durably enqueue every staged physical vector."""
    with repo._pool.connection() as conn:
        with conn.transaction():
            row = conn.execute(
                """
                SELECT source_id, status
                FROM knowledge_generations
                WHERE generation_id = %s
                FOR UPDATE
                """,
                (generation_id,),
            ).fetchone()
            if not row:
                return
            status = str(row["status"])
            if status in {"active", "retired"}:
                return
            source_id = str(row["source_id"])
            staged = conn.execute(
                """
                SELECT qdrant_point_id
                FROM staged_chunks
                WHERE generation_id = %s AND qdrant_point_id IS NOT NULL
                """,
                (generation_id,),
            ).fetchall()
            staged_ids = [str(item["qdrant_point_id"]) for item in staged]
            point_ids = _unique([*cleanup_point_ids, *staged_ids])
            conn.execute(
                """
                UPDATE knowledge_generations
                SET status = 'failed', failed_at = NOW(), error = %s
                WHERE generation_id = %s
                """,
                (str(error)[:4000], generation_id),
            )
            repo._enqueue_cleanup_events(
                conn,
                source_id=source_id,
                generation_id=generation_id,
                point_ids=point_ids,
            )
            conn.execute("DELETE FROM staged_chunks WHERE generation_id = %s", (generation_id,))
            conn.execute("DELETE FROM staged_documents WHERE generation_id = %s", (generation_id,))


def fail_inmemory_generation(
    repo: Any,
    generation_id: str,
    error: str,
    cleanup_point_ids: Iterable[str] = (),
) -> None:
    with repo._lock:
        repo._ensure_generation_state()
        generation = repo._knowledge_generations.get(generation_id)
        if generation is None or generation.status in {"active", "retired"}:
            return
        staged_ids = [
            str(chunk.qdrant_point_id)
            for chunk in repo._staged_chunks.get(generation_id, {}).values()
            if chunk.qdrant_point_id
        ]
        point_ids = _unique([*cleanup_point_ids, *staged_ids])
        generation.status = "failed"
        generation.failed_at = datetime.now(timezone.utc).isoformat()
        generation.error = str(error)
        repo._enqueue_memory_cleanup(generation.source_id, generation_id, point_ids)
        repo._staged_documents.pop(generation_id, None)
        repo._staged_chunks.pop(generation_id, None)


def reconcile_postgres_generations(repo: Any) -> Dict[str, int]:
    """Fail candidate generations whose durable ingestion Job is no longer running.

    Worker queue reconciliation changes a crashed Job from running to pending (or
    dead-lettered) after its lease expires. At that point no process owns the
    candidate generation, so its staged manifest can safely be failed and its
    physical Qdrant points sent to the publication outbox before the next Job
    attempt starts a fresh generation.
    """
    recovered = 0
    cleanup_events = 0
    with repo._pool.connection() as conn:
        with conn.transaction():
            rows = conn.execute(
                """
                SELECT kg.generation_id, kg.source_id
                FROM knowledge_generations AS kg
                LEFT JOIN ingestion_jobs AS j ON j.job_id = kg.job_id
                WHERE kg.status IN ('staging', 'prepared')
                  AND (j.job_id IS NULL OR j.status <> 'running')
                ORDER BY kg.created_at
                FOR UPDATE OF kg SKIP LOCKED
                """
            ).fetchall()
            for row in rows:
                generation_id = str(row["generation_id"])
                source_id = str(row["source_id"])
                point_rows = conn.execute(
                    """
                    SELECT qdrant_point_id
                    FROM staged_chunks
                    WHERE generation_id = %s AND qdrant_point_id IS NOT NULL
                    """,
                    (generation_id,),
                ).fetchall()
                point_ids = _unique(str(item["qdrant_point_id"]) for item in point_rows)
                conn.execute(
                    """
                    UPDATE knowledge_generations
                    SET status = 'failed', failed_at = NOW(),
                        error = COALESCE(error, 'orphaned candidate generation recovered after Job ownership ended')
                    WHERE generation_id = %s
                    """,
                    (generation_id,),
                )
                repo._enqueue_cleanup_events(
                    conn,
                    source_id=source_id,
                    generation_id=generation_id,
                    point_ids=point_ids,
                )
                if point_ids:
                    cleanup_events += 1
                conn.execute("DELETE FROM staged_chunks WHERE generation_id = %s", (generation_id,))
                conn.execute("DELETE FROM staged_documents WHERE generation_id = %s", (generation_id,))
                recovered += 1
    return {"recovered_generations": recovered, "cleanup_generations": cleanup_events}


def reconcile_inmemory_generations(repo: Any) -> Dict[str, int]:
    recovered = 0
    cleanup_generations = 0
    with repo._lock:
        repo._ensure_generation_state()
        candidates = []
        for generation in repo._knowledge_generations.values():
            if generation.status not in {"staging", "prepared"}:
                continue
            job = repo._jobs.get(generation.job_id) if generation.job_id else None
            if job is not None and job.status == "running":
                continue
            candidates.append(generation)
        for generation in candidates:
            point_ids = _unique(
                str(chunk.qdrant_point_id)
                for chunk in repo._staged_chunks.get(generation.generation_id, {}).values()
                if chunk.qdrant_point_id
            )
            generation.status = "failed"
            generation.failed_at = datetime.now(timezone.utc).isoformat()
            generation.error = generation.error or (
                "orphaned candidate generation recovered after Job ownership ended"
            )
            repo._enqueue_memory_cleanup(
                generation.source_id,
                generation.generation_id,
                point_ids,
            )
            if point_ids:
                cleanup_generations += 1
            repo._staged_documents.pop(generation.generation_id, None)
            repo._staged_chunks.pop(generation.generation_id, None)
            recovered += 1
    return {
        "recovered_generations": recovered,
        "cleanup_generations": cleanup_generations,
    }