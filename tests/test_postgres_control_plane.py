from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.api.app.storage.managed_pg_repo import ManagedPostgresRepo
from services.api.app.storage.models import IngestionJob, Source
from services.worker.scheduler import schedule_due_sources, scheduled_job_id

pytestmark = pytest.mark.skipif(not os.getenv("POSTGRES_TEST_DSN"), reason="POSTGRES_TEST_DSN not configured")


def test_postgres_persists_source_schedule_and_atomic_scheduled_job():
    repo = ManagedPostgresRepo(os.environ["POSTGRES_TEST_DSN"], pool_min=1, pool_max=2)
    try:
        suffix = uuid.uuid4().hex
        now = datetime.now(timezone.utc).replace(microsecond=0)
        source = Source(
            source_id=f"source-control-{suffix}",
            tenant_id=f"tenant-control-{suffix}",
            source_type="local_fs",
            name="Scheduled manuals",
            config={"path": "/data/manuals"},
            sync_enabled=True,
            sync_interval_seconds=300,
            sync_next_at=(now - timedelta(seconds=1)).isoformat(),
        )
        repo.add_source(source)

        loaded = repo.get_source(source.source_id)
        assert loaded is not None
        assert loaded.sync_enabled is True
        assert loaded.sync_interval_seconds == 300
        assert loaded.sync_next_at is not None

        result = schedule_due_sources(repo, now=now)
        assert result["enqueued"] == 1
        jobs = repo.list_jobs(tenant_id=source.tenant_id, source_id=source.source_id)
        assert len(jobs) == 1
        assert jobs[0].stats["trigger"] == "scheduled"
        assert jobs[0].job_id == scheduled_job_id(source.source_id, datetime.fromisoformat(str(loaded.sync_next_at)))

        duplicate = IngestionJob(
            job_id=jobs[0].job_id,
            tenant_id=source.tenant_id,
            source_id=source.source_id,
            source_type=source.source_type,
            source_config=dict(source.config),
        )
        assert repo.add_job_if_absent(duplicate) is False
        assert len(repo.list_jobs(tenant_id=source.tenant_id, source_id=source.source_id)) == 1

        advanced = repo.get_source(source.source_id)
        assert advanced is not None
        assert datetime.fromisoformat(str(advanced.sync_next_at)) > now
    finally:
        repo.close()
