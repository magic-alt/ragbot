"""PostgreSQL integration smoke test.

Skipped in the normal unit-test matrix. CI provides POSTGRES_TEST_DSN in the
postgres-smoke job after applying every migration.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone

from services.api.app.retrieval.embedder import HashEmbedder
from services.api.app.retrieval.qdrant import InMemoryQdrant
from services.api.app.storage.models import ACLPolicy, Source
from services.api.app.storage.pg_repo import PostgresRepo
from services.worker.pipeline import run_ingest_pipeline


@unittest.skipUnless(os.getenv("POSTGRES_TEST_DSN"), "POSTGRES_TEST_DSN not configured")
class PostgresRepoIntegrationTests(unittest.TestCase):
    def test_local_fs_ingestion_round_trip(self):
        dsn = os.environ["POSTGRES_TEST_DSN"]
        repo = PostgresRepo(dsn, pool_min=1, pool_max=2)
        self.addCleanup(repo.close)

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
                    handle.write(f"{filename} durable RAG knowledge. " * 30)

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

        stored_job = repo.get_job("job-postgres-smoke")
        self.assertIsNotNone(stored_job)
        self.assertEqual(stored_job.stats["chunks_ingested"], job.chunk_count)
