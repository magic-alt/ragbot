"""Production queue reliability integration tests using ManagedPostgresRepo."""
from __future__ import annotations

import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone

from services.api.app.storage.managed_pg_repo import ManagedPostgresRepo
from services.api.app.storage.models import IngestionJob, Source


@unittest.skipUnless(os.getenv("POSTGRES_TEST_DSN"), "POSTGRES_TEST_DSN not configured")
class ManagedPostgresReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.repo = ManagedPostgresRepo(
            dsn=os.environ["POSTGRES_TEST_DSN"],
            pool_min=1,
            pool_max=2,
        )
        self.addCleanup(self.repo.close)
        self.suffix = uuid.uuid4().hex
        self.tenant_id = f"tenant-reliability-{self.suffix}"
        self.source = Source(
            source_id=f"source-reliability-{self.suffix}",
            tenant_id=self.tenant_id,
            source_type="local_fs",
            name="reliability smoke",
            config={"path": "/tmp"},
        )
        self.repo.add_source(self.source)

    def _job(self, name: str, *, status: str, attempts: int) -> IngestionJob:
        expired = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        return IngestionJob(
            job_id=f"job-{name}-{self.suffix}",
            tenant_id=self.tenant_id,
            source_id=self.source.source_id,
            source_type=self.source.source_type,
            source_config=self.source.config,
            status=status,
            attempts=attempts,
            available_at=expired,
            lease_owner="worker-dead" if status == "running" else None,
            lease_expires_at=expired if status == "running" else None,
            error="temporary provider failure" if status == "failed" else None,
        )

    def test_reconcile_recovers_retryable_stranded_jobs(self):
        expired = self._job("expired", status="running", attempts=1)
        failed = self._job("failed", status="failed", attempts=1)
        self.repo.add_job(expired)
        self.repo.add_job(failed)

        stats = self.repo.reconcile_ingestion_jobs(max_attempts=3)

        self.assertGreaterEqual(stats["recovered_running"], 1)
        self.assertGreaterEqual(stats["recovered_failed"], 1)
        recovered_expired = self.repo.get_job(expired.job_id)
        recovered_failed = self.repo.get_job(failed.job_id)
        self.assertEqual(recovered_expired.status, "pending")
        self.assertIsNone(recovered_expired.lease_owner)
        self.assertIsNone(recovered_expired.lease_expires_at)
        self.assertEqual(recovered_failed.status, "pending")
        self.assertIsNone(recovered_failed.completed_at)

    def test_reconcile_dead_letters_exhausted_job_and_persists_metadata(self):
        exhausted = self._job("exhausted", status="running", attempts=3)
        self.repo.add_job(exhausted)

        stats = self.repo.reconcile_ingestion_jobs(max_attempts=3)
        self.assertGreaterEqual(stats["dead_lettered_running"], 1)

        stored = self.repo.get_job(exhausted.job_id)
        self.assertEqual(stored.status, "dead_lettered")
        self.assertEqual(stored.failure_class, "lease_exhausted")
        self.assertIsNotNone(stored.dead_lettered_at)
        self.assertIsNotNone(stored.completed_at)
        self.assertIsNone(stored.lease_owner)
        self.assertIn("maximum attempts", stored.error)

    def test_claim_after_reconciliation_uses_next_attempt(self):
        stranded = self._job("claim", status="failed", attempts=1)
        self.repo.add_job(stranded)
        self.repo.reconcile_ingestion_jobs(max_attempts=3)

        claimed = self.repo.claim_next_job(
            f"worker-{self.suffix}",
            lease_seconds=30,
            max_attempts=3,
        )
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.job_id, stranded.job_id)
        self.assertEqual(claimed.status, "running")
        self.assertEqual(claimed.attempts, 2)


if __name__ == "__main__":
    unittest.main()
