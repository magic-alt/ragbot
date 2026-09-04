from __future__ import annotations

from typing import Any

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


def ensure_generation_repository(repo: Any) -> Any:
    """Attach the staged-publication adapter to built-in repositories.

    Generation publication deliberately remains an additive storage capability:
    external/custom Repo implementations can continue to exist without silently
    pretending to provide atomic publication. Built-in PostgreSQL and in-memory
    repositories are upgraded in-place so existing factory/test construction
    stays source-compatible.
    """
    if callable(getattr(repo, "begin_knowledge_generation", None)):
        return repo

    is_postgres = hasattr(repo, "_pool")
    if is_postgres:
        backend = PostgresGenerationMixin
        helpers = _POSTGRES_HELPERS
    elif hasattr(repo, "_lock") and hasattr(repo, "_documents") and hasattr(repo, "_chunks"):
        backend = InMemoryGenerationMixin
        helpers = _IN_MEMORY_HELPERS
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
        if is_postgres and name == "retry_publication_outbox":
            method = _retry_publication_outbox_postgres
        else:
            method = getattr(backend, name)
        setattr(repo, name, method.__get__(repo, type(repo)))
    return repo


def supports_generation_publication(repo: Any) -> bool:
    ensure_generation_repository(repo)
    return all(callable(getattr(repo, name, None)) for name in _GENERATION_METHODS)
