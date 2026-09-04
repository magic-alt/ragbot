from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .generation_activation import activate_inmemory_generation, activate_postgres_generation
from .generation_recovery import (
    fail_inmemory_generation,
    fail_postgres_generation,
    reconcile_inmemory_generations,
    reconcile_postgres_generations,
)
from .generation_repo import InMemoryGenerationMixin, PostgresGenerationMixin

_GENERATION_METHODS = (
    "begin_knowledge_generation",
    "stage_knowledge_generation",
    "mark_knowledge_generation_prepared",
    "activate_knowledge_generation",
    "fail_knowledge_generation",
    "get_active_generation_id",
    "active_vector_points",
    "claim_publication_outbox",
    "complete_publication_outbox",
    "retry_publication_outbox",
    "reconcile_publication_outbox",
)
_POSTGRES_HELPERS = (
    "_generation_source_id",
    "_enqueue_cleanup_events",
)
_IN_MEMORY_HELPERS = (
    "_ensure_generation_state",
    "_enqueue_memory_cleanup",
)


def _mark_prepared_postgres(
    repo: Any,
    generation_id: str,
    stats: Optional[Dict[str, Any]] = None,
) -> None:
    """Prepare a generation without dropping its durable lifecycle snapshot."""
    with repo._pool.connection() as conn:
        result = conn.execute(
            """
            UPDATE knowledge_generations
            SET status = 'prepared',
                prepared_at = NOW(),
                stats = COALESCE(stats, '{}'::jsonb) || %(stats)s
            WHERE generation_id = %(generation_id)s
              AND status IN ('staging', 'prepared')
            """,
            {
                "generation_id": generation_id,
                "stats": repo._jsonb(stats or {}),
            },
        )
        if not (result.rowcount or 0):
            raise ValueError(f"Generation is not stageable/preparable: {generation_id}")


def _mark_prepared_inmemory(
    repo: Any,
    generation_id: str,
    stats: Optional[Dict[str, Any]] = None,
) -> None:
    with repo._lock:
        repo._ensure_generation_state()
        generation = repo._knowledge_generations.get(generation_id)
        if generation is None or generation.status not in {"staging", "prepared"}:
            raise ValueError(f"Generation is not preparable: {generation_id}")
        merged = dict(generation.stats or {})
        merged.update(stats or {})
        generation.stats = merged
        generation.status = "prepared"
        generation.prepared_at = datetime.now(timezone.utc).isoformat()


def _retry_publication_outbox_postgres(
    repo: Any,
    outbox_id: int,
    worker_id: str,
    error: str,
    delay_seconds: float,
    max_attempts: int = 10,
) -> bool:
    with repo._pool.connection() as conn:
        with conn.transaction():
            row = conn.execute(
                "SELECT attempts FROM publication_outbox WHERE outbox_id = %s AND lease_owner = %s FOR UPDATE",
                (int(outbox_id), worker_id),
            ).fetchone()
            if not row:
                return False
            attempts = int(row["attempts"] if isinstance(row, dict) else row[0])
            if attempts >= max_attempts:
                result = conn.execute(
                    """
                    UPDATE publication_outbox
                    SET status = 'failed', last_error = %s,
                        lease_owner = NULL, lease_expires_at = NULL
                    WHERE outbox_id = %s AND lease_owner = %s
                    """,
                    (str(error)[:4000], int(outbox_id), worker_id),
                )
            else:
                result = conn.execute(
                    """
                    UPDATE publication_outbox
                    SET status = 'pending', last_error = %s,
                        available_at = NOW() + (%s * INTERVAL '1 second'),
                        lease_owner = NULL, lease_expires_at = NULL
                    WHERE outbox_id = %s AND lease_owner = %s
                    """,
                    (str(error)[:4000], max(0.0, float(delay_seconds)), int(outbox_id), worker_id),
                )
            return (result.rowcount or 0) > 0


def _reconcile_publication_postgres(repo: Any, max_attempts: int = 10) -> Dict[str, int]:
    generation_stats = reconcile_postgres_generations(repo)
    outbox_stats = PostgresGenerationMixin.reconcile_publication_outbox(
        repo,
        max_attempts=max_attempts,
    )
    return {**outbox_stats, **generation_stats}


def _reconcile_publication_inmemory(repo: Any, max_attempts: int = 10) -> Dict[str, int]:
    generation_stats = reconcile_inmemory_generations(repo)
    outbox_stats = InMemoryGenerationMixin.reconcile_publication_outbox(
        repo,
        max_attempts=max_attempts,
    )
    return {**outbox_stats, **generation_stats}


def ensure_generation_repository(repo: Any) -> Any:
    """Attach staged-publication adapters to Ragbot's built-in repositories.

    Custom repositories that already implement the generation capability are
    left untouched. Repositories with only the baseline ``Repo`` contract keep
    using the explicit legacy direct-publication compatibility path.
    """
    if callable(getattr(repo, "begin_knowledge_generation", None)):
        return repo

    is_postgres = hasattr(repo, "_pool")
    if is_postgres:
        backend = PostgresGenerationMixin
        helpers = _POSTGRES_HELPERS
        fenced_activation = activate_postgres_generation
        prepared = _mark_prepared_postgres
        failed = fail_postgres_generation
        recover = reconcile_postgres_generations
        publication_reconcile = _reconcile_publication_postgres
    elif hasattr(repo, "_lock") and hasattr(repo, "_documents") and hasattr(repo, "_chunks"):
        backend = InMemoryGenerationMixin
        helpers = _IN_MEMORY_HELPERS
        fenced_activation = activate_inmemory_generation
        prepared = _mark_prepared_inmemory
        failed = fail_inmemory_generation
        recover = reconcile_inmemory_generations
        publication_reconcile = _reconcile_publication_inmemory
    else:
        return repo

    for name in helpers:
        method = getattr(backend, name)
        setattr(repo, name, method.__get__(repo, type(repo)))

    if is_postgres:
        # Static row mapper must remain an unbound callable when installed as an
        # instance attribute; otherwise Python would inject repo as a first arg.
        setattr(repo, "_row_to_publication_event", getattr(backend, "_row_to_publication_event"))

    for name in _GENERATION_METHODS:
        if name == "mark_knowledge_generation_prepared":
            method = prepared
        elif name == "activate_knowledge_generation":
            method = fenced_activation
        elif name == "fail_knowledge_generation":
            method = failed
        elif name == "reconcile_publication_outbox":
            method = publication_reconcile
        elif is_postgres and name == "retry_publication_outbox":
            method = _retry_publication_outbox_postgres
        else:
            method = getattr(backend, name)
        setattr(repo, name, method.__get__(repo, type(repo)))

    # Recovery is intentionally not part of the mandatory GenerationRepo
    # protocol. It is a worker-side durability hook for Ragbot's built-ins.
    setattr(repo, "reconcile_knowledge_generations", recover.__get__(repo, type(repo)))
    return repo


def supports_generation_publication(repo: Any) -> bool:
    ensure_generation_repository(repo)
    return all(callable(getattr(repo, name, None)) for name in _GENERATION_METHODS)