from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from services.api.app.routes.ingest import enqueue_ingestion_job
from services.api.app.routes.quick_import import QuickSourceSpec, _run_quick_import
from services.api.app.storage.models import IngestionJob, Source
from services.api.app.storage.repo import InMemoryRepo
from services.worker import main as worker_main


@pytest.fixture(autouse=True)
def _worker_mode(monkeypatch):
    monkeypatch.setenv("RAGBOT_INGESTION_MODE", "worker")
    monkeypatch.delenv("POSTGRES_DSN", raising=False)


def _services(repo: InMemoryRepo | None = None):
    return SimpleNamespace(repo=repo or InMemoryRepo(), qdrant=object(), embedder=object())


def test_enqueue_copies_connector_config_snapshot():
    services = _services()
    source = Source(
        source_id="source-1",
        tenant_id="t1",
        source_type="repo",
        name="repo",
        config={
            "path": "https://example.com/repo.git",
            "ref": "main",
            "options": {"depth": 1},
        },
        status="active",
    )
    services.repo.add_source(source)

    job = enqueue_ingestion_job(source, services, job_id="job-snapshot")
    source.config["ref"] = "release"
    source.config["options"]["depth"] = 99

    persisted = services.repo.get_job(job.job_id)
    assert persisted is not None
    assert persisted.source_config["ref"] == "main"
    assert persisted.source_config["options"]["depth"] == 1
    assert services.repo.get_source(source.source_id).config["ref"] == "release"


def test_changed_connector_config_does_not_reuse_active_job_or_mutate_source():
    services = _services()
    first = QuickSourceSpec(
        location="https://example.com/team/repo.git",
        source_type="repo",
        config={"ref": "main"},
    )
    second = QuickSourceSpec(
        location="https://example.com/team/repo.git/",
        source_type="repo",
        config={"ref": "release"},
    )

    accepted = _run_quick_import(tenant_id="t1", spec=first, services=services)
    source_before = services.repo.get_source(accepted["source_id"])
    assert source_before is not None
    assert source_before.config["ref"] == "main"

    with pytest.raises(HTTPException) as exc_info:
        _run_quick_import(tenant_id="t1", spec=second, services=services)

    assert exc_info.value.status_code == 409
    source_after = services.repo.get_source(accepted["source_id"])
    assert source_after is not None
    assert source_after.config["ref"] == "main"
    assert len(services.repo.list_jobs(tenant_id="t1", source_id=accepted["source_id"])) == 1


def test_durable_worker_executes_job_connector_config_snapshot(monkeypatch):
    repo = InMemoryRepo()
    source = Source(
        source_id="source-1",
        tenant_id="t1",
        source_type="repo",
        name="repo",
        config={"path": "https://example.com/repo.git", "ref": "release"},
        status="active",
    )
    repo.add_source(source)
    job = IngestionJob(
        job_id="job-1",
        tenant_id="t1",
        source_id=source.source_id,
        source_type="repo",
        source_config={"path": "https://example.com/repo.git", "ref": "main"},
        status="running",
        attempts=1,
        lease_owner="worker-1",
    )
    repo.add_job(job)

    captured = {}

    def fake_pipeline(source_arg, repo_arg, qdrant_arg, job_id, embedder, existing_job):
        captured["source"] = source_arg
        captured["job_id"] = job_id
        captured["existing_job"] = existing_job
        repo_arg.update_job(job_id, status="completed", lease_owner=None, lease_expires_at=None)
        return repo_arg.get_job(job_id)

    monkeypatch.setattr(worker_main, "run_ingest_pipeline", fake_pipeline)
    services = SimpleNamespace(repo=repo, qdrant=object(), embedder=object())

    worker_main._execute_claimed_job(
        job,
        services,
        worker_id="worker-1",
        lease_seconds=3,
    )

    executed_source = captured["source"]
    assert executed_source.source_id == source.source_id
    assert executed_source.config["ref"] == "main"
    assert repo.get_source(source.source_id).config["ref"] == "release"
    assert captured["job_id"] == "job-1"
    assert captured["existing_job"] is True
