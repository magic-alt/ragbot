from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from services.api.app.retrieval.lexical import build_or_tsquery, contains_cjk, lexicalize
from services.api.app.runtime import validate_production_environment
from services.api.app.storage.models import IngestionJob
from services.api.app.storage.repo import InMemoryRepo


class CjkLexicalTests(unittest.TestCase):
    def test_cjk_bigram_lexicalization(self):
        terms = lexicalize("EtherCAT伺服驱动器 CiA402").split()
        self.assertIn("ethercat", terms)
        self.assertIn("cia402", terms)
        self.assertIn("伺服", terms)
        self.assertIn("服驱", terms)
        self.assertIn("驱动", terms)
        self.assertIn("动器", terms)
        self.assertTrue(contains_cjk("伺服驱动"))
        self.assertFalse(contains_cjk("servo drive"))

    def test_cjk_tsquery_is_or_based(self):
        query = build_or_tsquery("伺服驱动")
        self.assertEqual(query, "'伺服' | '服驱' | '驱动'")


class DurableInMemoryQueueTests(unittest.TestCase):
    def test_claim_heartbeat_and_expired_lease_recovery(self):
        repo = InMemoryRepo()
        job = IngestionJob(
            job_id="job-1",
            tenant_id="tenant-1",
            source_id="source-1",
            source_type="local_fs",
            source_config={"path": "/tmp"},
        )
        repo.add_job(job)

        claimed = repo.claim_next_job("worker-a", lease_seconds=30, max_attempts=3)
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.status, "running")
        self.assertEqual(claimed.attempts, 1)
        self.assertEqual(claimed.lease_owner, "worker-a")
        self.assertTrue(repo.heartbeat_job("job-1", "worker-a", lease_seconds=30))
        self.assertFalse(repo.heartbeat_job("job-1", "worker-b", lease_seconds=30))

        expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        repo.update_job("job-1", lease_expires_at=expired)
        reclaimed = repo.claim_next_job("worker-b", lease_seconds=30, max_attempts=3)
        self.assertIsNotNone(reclaimed)
        self.assertEqual(reclaimed.attempts, 2)
        self.assertEqual(reclaimed.lease_owner, "worker-b")

    def test_exhausted_expired_job_becomes_dead_lettered(self):
        repo = InMemoryRepo()
        expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        repo.add_job(
            IngestionJob(
                job_id="job-exhausted",
                tenant_id="tenant-1",
                source_id="source-1",
                source_type="local_fs",
                source_config={},
                status="running",
                attempts=3,
                lease_owner="dead-worker",
                lease_expires_at=expired,
            )
        )
        self.assertIsNone(repo.claim_next_job("worker-new", max_attempts=3))
        dead = repo.get_job("job-exhausted")
        self.assertEqual(dead.status, "dead_lettered")
        self.assertEqual(dead.failure_class, "lease_exhausted")
        self.assertIsNotNone(dead.dead_lettered_at)
        self.assertIn("maximum attempts", dead.error)


class ProductionIngestionModeTests(unittest.TestCase):
    def _valid_production_env(self) -> dict[str, str]:
        return {
            "RAGBOT_ENV": "production",
            "POSTGRES_DSN": "postgresql://example.invalid/ragbot",
            "QDRANT_URL": "https://qdrant.example.invalid",
            "EMBEDDING_MODEL": "text-embedding-3-small",
            "OPENAI_API_KEY": "test-provider-key",
            "RAGBOT_API_KEYS": "service-key",
            "RAGBOT_API_KEY_PRINCIPALS": (
                '{"service-key":{"tenant_ids":["tenant-1"],'
                '"user_id":"svc","groups":[],"roles":["reader"],"admin":false}}'
            ),
        }

    def test_production_rejects_inline_ingestion(self):
        env = self._valid_production_env()
        env["RAGBOT_INGESTION_MODE"] = "inline"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "durable worker"):
                validate_production_environment()

    def test_production_accepts_worker_ingestion(self):
        env = self._valid_production_env()
        env["RAGBOT_INGESTION_MODE"] = "worker"
        with patch.dict(os.environ, env, clear=True):
            validate_production_environment()


if __name__ == "__main__":
    unittest.main()
