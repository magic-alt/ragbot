"""PostgreSQL integration smoke tests.

Skipped in the normal unit-test matrix. CI provides POSTGRES_TEST_DSN and uses
the same migration runner as Docker/Helm before executing this file.
"""
from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from services.api.app.retrieval.embedder import HashEmbedder
from services.api.app.retrieval.pg_fts import fts_search
from services.api.app.retrieval.qdrant import InMemoryQdrant
from services.api.app.routes.quick_import import QuickSourceSpec, _run_quick_import
from services.api.app.storage.migrations import apply_migrations
from services.api.app.storage.models import ACLPolicy, IngestionJob, Source
from services.api.app.storage.pg_repo import PostgresRepo
from services.worker import main as worker_main
from services.worker.pipeline import run_ingest_pipeline


@unittest.skipUnless(os.getenv("POSTGRES_TEST_DSN"), "POSTGRES_TEST_DSN not configured")
class PostgresRepoIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.dsn = os.environ["POSTGRES_TEST_DSN"]

    def test_migration_runner_is_idempotent_after_ci_bootstrap(self):
        self.assertEqual(apply_migrations(self.dsn), [])

    def test_local_fs_ingestion_round_trip_and_native_fts(self):
        repo = PostgresRepo(self.dsn, pool_min=1, pool_max=2)
        self.addCleanup(repo.close)
        self.assertTrue(repo.healthcheck())

        now = datetime.now(timezone.utc).isoformat()
        source = Source(
            source_id="src-postgres-smoke",
            tenant_id="tenant-postgres-smoke",
            source_type="local_fs",
            name="Postgres smoke knowledge",
            config={},
            tags=["smoke", "local"],
            created_at=now,
            updated_at=now,
        )
        policy = ACLPolicy(
            acl_policy_id="acl-postgres-smoke",
            tenant_id=source.tenant_id,
            rules={"allow": ["smoke-user"]},
            policy_hash="smoke-policy-hash",
        )
        repo.add_policy(policy)
        self.assertEqual(repo.get_policy_hash(policy.acl_policy_id), policy.policy_hash)

        with tempfile.TemporaryDirectory() as directory:
            for filename in ("alpha.txt", "beta.md"):
                path = os.path.join(directory, filename)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(f"{filename} durable searchable RAG knowledge. " * 30)

            source.config = {"path": directory}
            source.acl_policy_id = policy.acl_policy_id
            repo.add_source(source)

            loaded_source = repo.get_source(source.source_id)
            self.assertIsNotNone(loaded_source)
            self.assertEqual(loaded_source.tags, ["smoke", "local"])

            qdrant = InMemoryQdrant(dim=32)
            job = run_ingest_pipeline(
                source,
                repo,
                qdrant,
                job_id="job-postgres-smoke",
                embedder=HashEmbedder(dim=32),
            )

        self.assertEqual(job.status, "completed", job.error)
        self.assertEqual(job.doc_count, 2)
        self.assertGreater(job.chunk_count, 0)
        self.assertEqual(set(job.stats["doc_ids"]), {
            "doc-src-postgres-smoke:alpha.txt",
            "doc-src-postgres-smoke:beta.md",
        })

        chunks = list(repo.iter_chunks())
        smoke_chunks = [c for c in chunks if c.tenant_id == source.tenant_id]
        self.assertTrue(smoke_chunks)
        for doc_id in job.stats["doc_ids"]:
            document = repo.get_document(doc_id)
            self.assertIsNotNone(document)
            self.assertEqual(document.tags, ["smoke", "local"])

        fts_hits = fts_search(
            repo,
            "durable searchable",
            {
                "tenant_id": source.tenant_id,
                "source_types": ["local_fs"],
                "tags": ["smoke"],
                "security_scope": [policy.policy_hash],
            },
            top_k=10,
        )
        self.assertGreaterEqual(len(fts_hits), 2)
        self.assertTrue(all(chunk.tenant_id == source.tenant_id for chunk, _ in fts_hits))

        stored_job = repo.get_job("job-postgres-smoke")
        self.assertIsNotNone(stored_job)
        self.assertEqual(stored_job.stats["chunks_ingested"], job.chunk_count)

    def test_durable_job_claim_heartbeat_and_expired_lease_recovery(self):
        repo = PostgresRepo(self.dsn, pool_min=1, pool_max=2)
        self.addCleanup(repo.close)
        suffix = uuid.uuid4().hex
        tenant_id = f"tenant-queue-{suffix}"
        source = Source(
            source_id=f"source-queue-{suffix}",
            tenant_id=tenant_id,
            source_type="local_fs",
            name="queue smoke",
            config={"path": "/tmp"},
        )
        repo.add_source(source)
        job = IngestionJob(
            job_id=f"job-queue-{suffix}",
            tenant_id=tenant_id,
            source_id=source.source_id,
            source_type=source.source_type,
            source_config=source.config,
        )
        repo.add_job(job)

        claimed = repo.claim_next_job("worker-a", lease_seconds=30, max_attempts=3)
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.job_id, job.job_id)
        self.assertEqual(claimed.status, "running")
        self.assertEqual(claimed.attempts, 1)
        self.assertEqual(claimed.lease_owner, "worker-a")
        self.assertTrue(repo.heartbeat_job(job.job_id, "worker-a", lease_seconds=30))

        expired = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        repo.update_job(job.job_id, lease_expires_at=expired)
        reclaimed = repo.claim_next_job("worker-b", lease_seconds=30, max_attempts=3)
        self.assertIsNotNone(reclaimed)
        self.assertEqual(reclaimed.job_id, job.job_id)
        self.assertEqual(reclaimed.attempts, 2)
        self.assertEqual(reclaimed.lease_owner, "worker-b")

        repo.update_job(job.job_id, attempts=3, lease_expires_at=expired)
        self.assertIsNone(repo.claim_next_job("worker-c", lease_seconds=30, max_attempts=3))
        exhausted = repo.get_job(job.job_id)
        self.assertEqual(exhausted.status, "failed")
        self.assertIn("maximum attempts", exhausted.error)

    def test_cjk_bigram_fts_retrieves_chinese_phrase(self):
        repo = PostgresRepo(self.dsn, pool_min=1, pool_max=2)
        self.addCleanup(repo.close)
        suffix = uuid.uuid4().hex
        tenant_id = f"tenant-cjk-{suffix}"
        source = Source(
            source_id=f"source-cjk-{suffix}",
            tenant_id=tenant_id,
            source_type="local_fs",
            name="Chinese retrieval smoke",
            config={},
            tags=["cjk"],
        )

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "servo.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "EtherCAT总线控制伺服驱动器需要配置CiA402状态机，"
                    "并确认同步周期和位置环控制参数。"
                )
            source.config = {"path": directory}
            repo.add_source(source)
            job = run_ingest_pipeline(
                source,
                repo,
                InMemoryQdrant(dim=32),
                job_id=f"job-cjk-{suffix}",
                embedder=HashEmbedder(dim=32),
            )

        self.assertEqual(job.status, "completed", job.error)
        hits = fts_search(
            repo,
            "伺服驱动",
            {"tenant_id": tenant_id, "source_types": ["local_fs"]},
            top_k=5,
        )
        self.assertTrue(hits)
        self.assertIn("伺服驱动器", hits[0][0].text)

    def test_quick_import_job_executes_original_postgres_config_snapshot(self):
        repo = PostgresRepo(self.dsn, pool_min=1, pool_max=2)
        self.addCleanup(repo.close)
        suffix = uuid.uuid4().hex
        tenant_id = f"tenant-quick-{suffix}"
        services = SimpleNamespace(
            repo=repo,
            qdrant=InMemoryQdrant(dim=32),
            embedder=HashEmbedder(dim=32),
        )

        with tempfile.TemporaryDirectory() as original_dir, tempfile.TemporaryDirectory() as replacement_dir:
            with open(os.path.join(original_dir, "original.txt"), "w", encoding="utf-8") as handle:
                handle.write("original durable snapshot knowledge " * 30)
            with open(os.path.join(replacement_dir, "replacement.txt"), "w", encoding="utf-8") as handle:
                handle.write("replacement mutable source knowledge " * 30)

            spec = QuickSourceSpec(location=original_dir, source_type="local_fs")
            with patch.dict(os.environ, {"RAGBOT_INGESTION_MODE": "worker"}, clear=False):
                submission = _run_quick_import(tenant_id=tenant_id, spec=spec, services=services)

            queued = repo.get_job(submission["job_id"])
            self.assertIsNotNone(queued)
            self.assertEqual(queued.status, "pending")
            self.assertEqual(queued.source_config["path"], original_dir)

            repo.update_source(submission["source_id"], config={"path": replacement_dir})
            repo.update_job(
                queued.job_id,
                status="running",
                attempts=1,
                lease_owner="worker-snapshot",
            )
            claimed = repo.get_job(queued.job_id)
            worker_main._execute_claimed_job(
                claimed,
                services,
                worker_id="worker-snapshot",
                lease_seconds=3,
            )

            finished = repo.get_job(queued.job_id)
            self.assertEqual(finished.status, "completed", finished.error)
            self.assertEqual(finished.doc_count, 1)
            self.assertTrue(any(doc_id.endswith(":original.txt") for doc_id in finished.stats["doc_ids"]))
            self.assertFalse(any(doc_id.endswith(":replacement.txt") for doc_id in finished.stats["doc_ids"]))
            current_source = repo.get_source(submission["source_id"])
            self.assertEqual(current_source.config["path"], replacement_dir)
