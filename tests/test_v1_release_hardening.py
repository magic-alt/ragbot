from __future__ import annotations

import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ProductionRuntimeTests(unittest.TestCase):
    def test_production_rejects_missing_durable_configuration(self):
        from services.api.app.runtime import validate_production_environment

        with patch.dict(os.environ, {"RAGBOT_ENV": "production"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Production configuration is incomplete"):
                validate_production_environment()

    def test_production_accepts_complete_scoped_configuration(self):
        from services.api.app.runtime import validate_production_environment

        principal = {
            "key-a": {
                "tenant_ids": ["tenant-a"],
                "user_id": "svc-a",
                "groups": ["engineering"],
                "roles": ["reader"],
                "admin": False,
            }
        }
        env = {
            "RAGBOT_ENV": "production",
            "POSTGRES_DSN": "postgresql://example.invalid/ragbot",
            "QDRANT_URL": "https://qdrant.example.invalid",
            "EMBEDDING_MODEL": "text-embedding-3-small",
            "EMBEDDING_API_KEY": "embedding-secret",
            "RAGBOT_API_KEYS": "key-a",
            "RAGBOT_API_KEY_PRINCIPALS": json.dumps(principal),
        }
        with patch.dict(os.environ, env, clear=True):
            validate_production_environment()


class PrincipalAuthorizationTests(unittest.TestCase):
    PRINCIPALS = {
        "tenant-key": {
            "tenant_ids": ["tenant-a"],
            "user_id": "svc-a",
            "groups": ["engineering"],
            "roles": ["reader"],
            "admin": False,
        },
        "admin-key": {
            "tenant_ids": [],
            "user_id": "admin-service",
            "groups": [],
            "roles": ["admin"],
            "admin": True,
        },
    }

    def test_tenant_and_user_are_bound_to_key(self):
        from fastapi import HTTPException
        from services.api.app.auth.principal import authorize_identity

        with patch.dict(
            os.environ,
            {"RAGBOT_API_KEY_PRINCIPALS": json.dumps(self.PRINCIPALS)},
            clear=False,
        ):
            user_id, groups, roles = authorize_identity("tenant-key", "tenant-a", "svc-a")
            self.assertEqual(user_id, "svc-a")
            self.assertIn("engineering", groups)
            self.assertIn("reader", roles)
            with self.assertRaises(HTTPException):
                authorize_identity("tenant-key", "tenant-b", "svc-a")
            with self.assertRaises(HTTPException):
                authorize_identity("tenant-key", "tenant-a", "spoofed-user")

    def test_non_admin_cannot_read_global_admin_surfaces(self):
        from fastapi import HTTPException
        from services.api.app.auth.principal import require_admin

        with patch.dict(
            os.environ,
            {"RAGBOT_API_KEY_PRINCIPALS": json.dumps(self.PRINCIPALS)},
            clear=False,
        ):
            with self.assertRaises(HTTPException):
                require_admin("tenant-key")
            require_admin("admin-key")


class EmbeddingSafetyTests(unittest.TestCase):
    def test_api_embedding_dimension_mismatch_is_rejected(self):
        from services.api.app.retrieval.embedder import APIEmbedder

        embedder = APIEmbedder(
            api_key="test",
            base_url="https://example.invalid",
            model="test-model",
            dimension=3,
        )
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            embedder._validate_dimension([0.1, 0.2])


class RerankerReliabilityTests(unittest.TestCase):
    def test_optional_reranker_failure_falls_back_to_rrf(self):
        from services.api.app.retrieval.embedder import HashEmbedder
        from services.api.app.retrieval.qdrant import InMemoryQdrant
        from services.api.app.retrieval.service import Retriever
        from services.api.app.storage.models import Chunk
        from services.api.app.storage.repo import InMemoryRepo

        class FailingReranker:
            enabled = True

            def rerank(self, query, documents, top_k=10):
                raise RuntimeError("reranker unavailable")

        repo = InMemoryRepo()
        embedder = HashEmbedder(dim=8)
        qdrant = InMemoryQdrant(dim=8)
        chunk = Chunk(
            chunk_id="c1",
            doc_id="d1",
            tenant_id="t1",
            chunk_index=0,
            text="alpha beta gamma",
            metadata={"source_type": "pdf", "acl_hash": "public"},
        )
        repo.add_chunk(chunk)
        qdrant.upsert(
            [
                (
                    chunk.chunk_id,
                    embedder.embed(chunk.text),
                    {
                        "doc_id": chunk.doc_id,
                        "tenant_id": chunk.tenant_id,
                        "source_type": "pdf",
                        "acl_hash": "public",
                        "text": chunk.text,
                    },
                )
            ]
        )
        retriever = Retriever(repo, qdrant, embedder=embedder, reranker=FailingReranker())
        results = retriever.retrieve("alpha", {"tenant_id": "t1"}, top_k=1)
        self.assertEqual([result.chunk_id for result in results], ["c1"])


class SourceBoundaryTests(unittest.TestCase):
    def test_remote_url_blocks_private_addresses(self):
        from services.worker.connectors.security import validate_remote_url

        private_dns = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 80))]
        with patch("services.worker.connectors.security.socket.getaddrinfo", return_value=private_dns):
            with self.assertRaisesRegex(ValueError, "non-public"):
                validate_remote_url("http://metadata.example/")

    def test_remote_url_honors_host_allowlist(self):
        from services.worker.connectors.security import validate_remote_url

        public_dns = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        with patch("services.worker.connectors.security.socket.getaddrinfo", return_value=public_dns):
            validate_remote_url(
                "https://docs.example.com/page",
                allowed_hosts=("*.example.com",),
            )
            with self.assertRaisesRegex(ValueError, "not allowlisted"):
                validate_remote_url(
                    "https://attacker.test/page",
                    allowed_hosts=("*.example.com",),
                )

    def test_production_local_sources_must_stay_inside_roots(self):
        from services.worker.connectors.security import validate_local_source_path

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            allowed_file = Path(root) / "doc.txt"
            allowed_file.write_text("ok", encoding="utf-8")
            outside_file = Path(outside) / "secret.txt"
            outside_file.write_text("no", encoding="utf-8")
            env = {
                "RAGBOT_ENV": "production",
                "RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS": root,
            }
            with patch.dict(os.environ, env, clear=False):
                self.assertEqual(
                    validate_local_source_path(str(allowed_file)),
                    str(allowed_file.resolve()),
                )
                with self.assertRaisesRegex(ValueError, "outside"):
                    validate_local_source_path(str(outside_file))


class ScopedApiSurfaceTests(unittest.TestCase):
    def test_source_and_search_endpoints_reject_cross_tenant_spoofing(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi TestClient not available")

        import services.api.app.api as api_mod
        from services.api.app.agent.graph import build_default_services
        from services.api.app.storage.models import Source

        principals = {
            "tenant-a-key": {
                "tenant_ids": ["tenant-a"],
                "user_id": "svc-a",
                "groups": [],
                "roles": ["reader"],
                "admin": False,
            }
        }
        old_services = api_mod._services
        old_keys = api_mod._VALID_API_KEYS
        try:
            api_mod._services = build_default_services()
            api_mod._VALID_API_KEYS = {"tenant-a-key"}
            api_mod._services.repo.add_source(
                Source(
                    source_id="tenant-b-source",
                    tenant_id="tenant-b",
                    source_type="web",
                    name="private",
                    config={"url": "https://example.com"},
                )
            )
            with patch.dict(
                os.environ,
                {"RAGBOT_API_KEY_PRINCIPALS": json.dumps(principals)},
                clear=False,
            ):
                client = TestClient(api_mod.app)
                headers = {"X-API-Key": "tenant-a-key"}
                response = client.get("/sources/tenant-b-source", headers=headers)
                self.assertEqual(response.status_code, 403)

                response = client.post(
                    "/search",
                    headers=headers,
                    json={
                        "query": "hello",
                        "tenant_id": "tenant-a",
                        "user_id": "spoofed-user",
                    },
                )
                self.assertEqual(response.status_code, 403)
        finally:
            api_mod._services = old_services
            api_mod._VALID_API_KEYS = old_keys


if __name__ == "__main__":
    unittest.main()
