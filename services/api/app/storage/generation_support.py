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

    backend = None
    if hasattr(repo, "_pool"):
        backend = PostgresGenerationMixin
    elif hasattr(repo, "_lock") and hasattr(repo, "_documents") and hasattr(repo, "_chunks"):
        backend = InMemoryGenerationMixin
    if backend is None:
        return repo

    for name in _GENERATION_METHODS:
        method = getattr(backend, name)
        setattr(repo, name, method.__get__(repo, type(repo)))
    return repo


def supports_generation_publication(repo: Any) -> bool:
    ensure_generation_repository(repo)
    return all(callable(getattr(repo, name, None)) for name in _GENERATION_METHODS)
