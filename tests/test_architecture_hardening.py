"""Regression tests for architecture, lifecycle, and API contract hardening."""
from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from services.api.app.agent.callbacks import AsyncQueueCallback
from services.api.app.agent.graph import build_default_services, run_agent
from services.api.app.api import ChatRequest
from services.api.app.retrieval.qdrant import InMemoryQdrant, QdrantClientAdapter
from services.api.app.storage.models import Source
from services.worker.pipeline import purge_source_knowledge, run_ingest_pipeline


class ContractAlignmentTests(unittest.TestCase):
    def test_chat_constraints_accept_real_local_fs_source(self):
        request = ChatRequest(
            query="find this",
            tenant_id="tenant",
            user_id="user",
            constraints={"source_types": ["local_fs"]},
        )
        self.assertEqual(request.constraints.source_types, ["local_fs"])

    def test_chat_constraints_reject_removed_db_doc_source(self):
        with self.assertRaises(ValidationError):
            ChatRequest(
                query="find this",
                tenant_id="tenant",
                user_id="user",
                constraints={"source_types": ["db_doc"]},
            )


class AgentStreamLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_callback_closes_when_agent_raises(self):
        services = build_default_services()
        callback = AsyncQueueCallback()
        with patch(
            "services.api.app.agent.graph.route_node",
            new=AsyncMock(side_effect=RuntimeError("provider unavailable")),
        ):
            with self.assertRaises(RuntimeError):
                await run_agent(
                    "question",
                    "tenant",
                    "user",
                    services,
                    callback=callback,
                )

        events = []
        async for event in callback:
            events.append(event)
        self.assertTrue(callback.closed)
        self.assertEqual(events[-1].event_type, "error")
        self.assertEqual(events[-1].data["error"], "Agent execution failed")


class IngestionLifecycleTests(unittest.TestCase):
    def test_identical_files_are_not_cross_document_deduplicated(self):
        services = build_default_services()
        qdrant = InMemoryQdrant(dim=64)
        with tempfile.TemporaryDirectory() as directory:
            for filename in ("a.txt", "b.txt"):
                with open(os.path.join(directory, filename), "w", encoding="utf-8") as handle:
                    handle.write("shared evidence text")
            source = Source(
                source_id="src-duplicates",
                tenant_id="tenant-a",
                source_type="local_fs",
                name="duplicate fixture",
                config={"path": directory},
            )
            services.repo.add_source(source)
            job = run_ingest_pipeline(source, services.repo, qdrant, job_id="job-duplicates")

        self.assertEqual(job.status, "completed")
        self.assertEqual(job.doc_count, 2)
        self.assertEqual(job.stats["chunks_total"], 2)
        self.assertEqual(len(list(services.repo.iter_chunks())), 2)

    def test_reingestion_reuses_unchanged_and_prunes_removed_documents(self):
        services = build_default_services()
        qdrant = InMemoryQdrant(dim=64)
        with tempfile.TemporaryDirectory() as directory:
            keep_path = os.path.join(directory, "keep.txt")
            remove_path = os.path.join(directory, "remove.txt")
            with open(keep_path, "w", encoding="utf-8") as handle:
                handle.write("stable knowledge")
            with open(remove_path, "w", encoding="utf-8") as handle:
                handle.write("temporary knowledge")

            source = Source(
                source_id="src-reconcile",
                tenant_id="tenant-a",
                source_type="local_fs",
                name="reconcile fixture",
                config={"path": directory},
            )
            services.repo.add_source(source)
            first = run_ingest_pipeline(source, services.repo, qdrant, job_id="job-first")
            self.assertEqual(first.status, "completed")
            self.assertEqual(first.doc_count, 2)

            os.unlink(remove_path)
            second = run_ingest_pipeline(source, services.repo, qdrant, job_id="job-second")

        self.assertEqual(second.status, "completed")
        self.assertEqual(second.chunk_count, 0)
        self.assertEqual(second.stats["chunks_reused"], 1)
        self.assertEqual(second.stats["documents_removed"], 1)
        documents = services.repo.list_documents("tenant-a")
        self.assertEqual([doc.doc_id for doc in documents], ["doc-src-reconcile:keep.txt"])
        hits = qdrant.search(
            [1.0] + [0.0] * 63,
            {"tenant_id": "tenant-a"},
            top_k=10,
        )
        self.assertTrue(all(hit[2]["doc_id"] == "doc-src-reconcile:keep.txt" for hit in hits))

    def test_purge_source_removes_documents_chunks_and_vectors(self):
        services = build_default_services()
        qdrant = InMemoryQdrant(dim=64)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "knowledge.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("knowledge to delete")
            source = Source(
                source_id="src-purge",
                tenant_id="tenant-a",
                source_type="local_fs",
                name="purge fixture",
                config={"path": directory},
            )
            services.repo.add_source(source)
            job = run_ingest_pipeline(source, services.repo, qdrant, job_id="job-purge")
            self.assertEqual(job.status, "completed")
            result = purge_source_knowledge(source, services.repo, qdrant)

        self.assertEqual(result["documents"], 1)
        self.assertEqual(services.repo.list_documents("tenant-a"), [])
        self.assertEqual(list(services.repo.iter_chunks()), [])
        self.assertEqual(qdrant.search([1.0] + [0.0] * 63, {"tenant_id": "tenant-a"}, 10), [])


class QdrantConfigurationTests(unittest.TestCase):
    def test_existing_collection_dimension_mismatch_fails_fast(self):
        fake_client = SimpleNamespace(
            collection_exists=lambda _name: True,
            get_collection=lambda _name: SimpleNamespace(
                config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=768)))
            ),
        )
        with patch("qdrant_client.QdrantClient", return_value=fake_client):
            with self.assertRaisesRegex(RuntimeError, "dimension"):
                QdrantClientAdapter(
                    url="http://qdrant.invalid",
                    api_key=None,
                    collection_name="rag_chunks",
                    dim=1536,
                )


if __name__ == "__main__":
    unittest.main()
