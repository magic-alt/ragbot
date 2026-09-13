from __future__ import annotations

from services.api.app.storage.pg_repo import PostgresRepo
from services.api.app.storage.postgres_repo import PostgresRepo as CompatibilityPostgresRepo
from services.api.app.storage.protocol import (
    ACLRepo,
    ControlPlaneRepo,
    DebugStateRepo,
    DevelopmentTableRepo,
    IngestionRepo,
    JobQueueRepo,
    KnowledgeCatalogRepo,
    Repo,
    SourceRepo,
)
from services.api.app.storage.repo import InMemoryRepo


def test_inmemory_repo_conforms_to_focused_capabilities():
    repo = InMemoryRepo()
    assert isinstance(repo, KnowledgeCatalogRepo)
    assert isinstance(repo, ACLRepo)
    assert isinstance(repo, SourceRepo)
    assert isinstance(repo, JobQueueRepo)
    assert isinstance(repo, IngestionRepo)
    assert isinstance(repo, ControlPlaneRepo)
    assert isinstance(repo, DebugStateRepo)
    assert isinstance(repo, DevelopmentTableRepo)
    assert isinstance(repo, Repo)


def test_postgres_compatibility_import_points_to_authoritative_class():
    assert CompatibilityPostgresRepo is PostgresRepo


def test_postgres_repository_no_longer_inherits_legacy_adapter():
    assert PostgresRepo.__bases__ == (object,)
    assert PostgresRepo.__module__ == "services.api.app.storage.pg_repo"


def test_production_postgres_class_exposes_capability_methods_without_instantiation():
    required = {
        KnowledgeCatalogRepo: ["add_document", "get_document", "add_chunks", "iter_chunks"],
        ACLRepo: ["add_policy", "get_policy_hash", "list_policies"],
        SourceRepo: ["add_source", "get_source", "list_sources", "update_source", "delete_source"],
        JobQueueRepo: ["add_job", "add_job_if_absent", "claim_next_job", "heartbeat_job", "release_job_lease", "reconcile_ingestion_jobs"],
    }
    for _capability, names in required.items():
        for name in names:
            assert callable(getattr(PostgresRepo, name, None)), name
