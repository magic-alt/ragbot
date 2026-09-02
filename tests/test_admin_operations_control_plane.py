from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api.app.routes.admin_ui import create_admin_ui_router
from services.api.app.routes.control_plane import build_overview, create_control_plane_router
from services.api.app.storage.models import IngestionJob, Source
from services.api.app.storage.repo import InMemoryRepo
from services.worker.scheduler import configure_source_sync, schedule_due_sources, scheduled_job_id


def _source(source_id: str = "source-1", tenant_id: str = "tenant-a") -> Source:
    return Source(
        source_id=source_id,
        tenant_id=tenant_id,
        source_type="local_fs",
        name="Manuals",
        config={"path": "/data/manuals", "nested": {"mode": "safe"}},
        status="active",
    )


def test_configure_sync_and_scheduler_are_idempotent_for_one_due_window():
    repo = InMemoryRepo()
    source = _source()
    repo.add_source(source)
    now = datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc)

    updated = configure_source_sync(
        repo,
        source,
        enabled=True,
        interval_seconds=300,
        run_immediately=True,
        now=now,
    )
    assert updated.sync_enabled is True
    assert updated.sync_next_at == now.isoformat()

    first = schedule_due_sources(repo, now=now)
    second = schedule_due_sources(repo, now=now)

    assert first["enqueued"] == 1
    assert second["enqueued"] == 0
    jobs = repo.list_jobs(tenant_id="tenant-a", source_id=source.source_id)
    assert len(jobs) == 1
    assert jobs[0].job_id == scheduled_job_id(source.source_id, now)
    assert jobs[0].stats["trigger"] == "scheduled"
    assert jobs[0].source_config == source.config
    assert jobs[0].source_config is not source.config
    assert repo.get_source(source.source_id).sync_next_at == (now + timedelta(seconds=300)).isoformat()


def test_scheduler_collapses_missed_intervals_instead_of_backfilling():
    repo = InMemoryRepo()
    source = _source()
    due = datetime(2026, 9, 2, 0, 0, tzinfo=timezone.utc)
    source.sync_enabled = True
    source.sync_interval_seconds = 3600
    source.sync_next_at = due.isoformat()
    repo.add_source(source)

    current = due + timedelta(hours=6, minutes=5)
    result = schedule_due_sources(repo, now=current)

    assert result["enqueued"] == 1
    assert len(repo.list_jobs(tenant_id="tenant-a")) == 1
    assert repo.get_source(source.source_id).sync_next_at == (due + timedelta(hours=7)).isoformat()


def test_scheduler_waits_while_a_manual_job_is_active():
    repo = InMemoryRepo()
    now = datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc)
    source = _source()
    source.sync_enabled = True
    source.sync_interval_seconds = 300
    source.sync_next_at = now.isoformat()
    repo.add_source(source)
    repo.add_job(
        IngestionJob(
            job_id="manual-job",
            tenant_id=source.tenant_id,
            source_id=source.source_id,
            source_type=source.source_type,
            source_config=dict(source.config),
            status="running",
            created_at=(now - timedelta(minutes=1)).isoformat(),
        )
    )

    result = schedule_due_sources(repo, now=now)

    assert result["blocked_active"] == 1
    assert len(repo.list_jobs(tenant_id="tenant-a")) == 1
    assert repo.get_source(source.source_id).sync_next_at == now.isoformat()


def test_overview_reports_queue_health_and_latest_knowledge_size():
    repo = InMemoryRepo()
    now = datetime.now(timezone.utc)
    source = _source()
    source.sync_enabled = True
    source.sync_interval_seconds = 300
    source.sync_next_at = (now + timedelta(minutes=5)).isoformat()
    repo.add_source(source)
    repo.add_job(
        IngestionJob(
            job_id="completed",
            tenant_id=source.tenant_id,
            source_id=source.source_id,
            source_type=source.source_type,
            source_config=dict(source.config),
            status="completed",
            doc_count=7,
            chunk_count=3,
            stats={"chunks_total": 42},
            created_at=(now - timedelta(minutes=10)).isoformat(),
            completed_at=(now - timedelta(minutes=9)).isoformat(),
        )
    )
    repo.add_job(
        IngestionJob(
            job_id="pending",
            tenant_id=source.tenant_id,
            source_id=source.source_id,
            source_type=source.source_type,
            source_config=dict(source.config),
            status="pending",
            created_at=(now - timedelta(seconds=30)).isoformat(),
        )
    )

    overview = build_overview(repo, {"tenant-a"})

    assert overview["sources"]["scheduled"] == 1
    assert overview["queue"]["pending"] == 1
    assert overview["queue"]["oldest_pending_age_seconds"] >= 29
    # Knowledge size is based on the latest completed index, not the pending run.
    assert overview["knowledge"] == {"documents": 7, "chunks": 42}


def test_catalog_api_is_tenant_scoped_and_never_returns_source_config(monkeypatch):
    repo = InMemoryRepo()
    one = _source("s1", "tenant-a")
    two = _source("s2", "tenant-b")
    repo.add_source(one)
    repo.add_source(two)
    repo.add_job(
        IngestionJob(
            job_id="j1",
            tenant_id="tenant-a",
            source_id="s1",
            source_type="local_fs",
            source_config={"path": "/secret/path", "token": "must-not-leak"},
            status="failed",
            error="boom",
        )
    )
    services = SimpleNamespace(repo=repo)
    app = FastAPI()

    async def auth_dep():
        return "tenant-a-key"

    monkeypatch.setenv(
        "RAGBOT_API_KEY_PRINCIPALS",
        '{"tenant-a-key":{"tenant_ids":["tenant-a"],"user_id":"svc-a","admin":false}}',
    )
    app.include_router(create_control_plane_router(lambda: services, auth_dep))
    client = TestClient(app)

    sources = client.get("/catalog/sources").json()["sources"]
    jobs = client.get("/catalog/jobs").json()["jobs"]

    assert [item["source_id"] for item in sources] == ["s1"]
    assert [item["job_id"] for item in jobs] == ["j1"]
    assert "source_config" not in jobs[0]
    assert "must-not-leak" not in str(jobs)


def test_admin_ui_is_zero_build_html_without_embedded_api_key():
    app = FastAPI()
    app.include_router(create_admin_ui_router())
    response = TestClient(app).get("/admin/ui")
    assert response.status_code == 200
    assert "Ragbot Control Plane" in response.text
    assert "sessionStorage" in response.text
    assert "RAGBOT_API_KEYS" not in response.text
