"""Regression tests for RAG runtime/environment consistency."""
from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from unittest.mock import patch

from services.api.app.agent.graph import build_default_services
from services.api.app.retrieval.embedder import HashEmbedder, build_embedder
from services.api.app.storage.models import Source
from services.worker.pipeline import run_ingest_pipeline


class _RecordingEmbedder:
    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension
        self.calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return "recording-embedder"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0] + [0.0] * (self._dimension - 1) for _ in texts]


class BuildEmbedderConfigurationTests(unittest.TestCase):
    def test_hash_fallback_uses_vector_store_dimension(self):
        with patch.dict(os.environ, {}, clear=True):
            embedder = build_embedder(dimension=1536)
        self.assertIsInstance(embedder, HashEmbedder)
        self.assertEqual(embedder.dimension, 1536)

    def test_qdrant_dim_env_overrides_caller_dimension(self):
        with patch.dict(os.environ, {"QDRANT_DIM": "256"}, clear=True):
            embedder = build_embedder(dimension=1536)
        self.assertEqual(embedder.dimension, 256)

    def test_hash_embedder_uses_stable_digest(self):
        embedder = HashEmbedder(dim=32)
        vector = embedder.embed("stable-token")
        digest = hashlib.blake2b(b"stable-token", digest_size=8).digest()
        expected_index = int.from_bytes(digest, byteorder="big", signed=False) % 32
        self.assertEqual(vector[expected_index], 1.0)
        self.assertEqual(sum(1 for value in vector if value), 1)


class IngestionConsistencyTests(unittest.TestCase):
    def test_pipeline_uses_shared_embedder_and_stable_document_id(self):
        services = build_default_services()
        services.qdrant._dim = 8  # test-only in-memory adapter configuration
        embedder = _RecordingEmbedder(dimension=8)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "knowledge.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("Ragbot indexes local knowledge for downstream agents. " * 20)

            source = Source(
                source_id="src-config",
                tenant_id="tenant-test",
                source_type="local_fs",
                name="Configuration test source",
                config={"path": directory},
            )
            services.repo.add_source(source)
            job = run_ingest_pipeline(
                source,
                services.repo,
                services.qdrant,
                job_id="job-config",
                embedder=embedder,
            )

        self.assertEqual(job.status, "completed")
        self.assertGreater(job.chunk_count, 0)
        self.assertTrue(embedder.calls)
        self.assertEqual(job.stats["doc_id"], "doc-src-config")

        chunks = list(services.repo.iter_chunks())
        self.assertTrue(chunks)
        self.assertTrue(all(chunk.doc_id == "doc-src-config" for chunk in chunks))
        self.assertIsNotNone(services.repo.get_document("doc-src-config"))

        hits = services.qdrant.search(
            [1.0] + [0.0] * 7,
            {"tenant_id": "tenant-test"},
            top_k=10,
        )
        self.assertTrue(hits)
        self.assertEqual(hits[0][2]["embedding_model"], "recording-embedder")


if __name__ == "__main__":
    unittest.main()
