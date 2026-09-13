"""Compatibility import for the authoritative PostgreSQL repository.

Historically Ragbot had a three-layer inheritance chain:
``postgres_repo.PostgresRepo -> pg_repo.PostgresRepo -> ManagedPostgresRepo``.
The current implementation lives only in :mod:`services.api.app.storage.pg_repo`.
This module remains as a source-compatible import for downstream callers and
must not regain independent SQL behavior.
"""

from .pg_repo import PostgresRepo

__all__ = ["PostgresRepo"]
