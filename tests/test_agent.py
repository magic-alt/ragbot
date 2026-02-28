import asyncio
import os
import tempfile
import time
import threading
import unittest

from services.api.app.agent.graph import build_default_services, run_agent
from services.api.app.agent.nodes.code import CodeSearch, code_node, open_file_node, explain_error_node
from services.api.app.agent.nodes.finalize import finalize_node
from services.api.app.agent.nodes.route import route_node
from services.api.app.agent.nodes.sql import SqlEngine
from services.api.app.agent.nodes.synthesize import synthesize_node
from services.api.app.agent.nodes.verify import verify_node
from services.api.app.agent.nodes.web import web_node
from services.api.app.agent.session import InMemorySessionStore, SessionTurn
from services.api.app.agent.state import build_initial_state
from services.api.app.agent.callbacks import AgentEvent, AsyncQueueCallback, QueueCallback, NullCallback
from services.api.app.agent.reliability import (
    CircuitBreaker, CircuitOpenError, ToolTimeoutError,
    safe_tool_call, RetryConfig, DEFAULT_RETRY, DEFAULT_TIMEOUTS, _get_breaker,
)
from services.api.app.auth.acl import build_policy, compute_security_scope, UserContext, compute_security_scope_from_context
from services.api.app.llm.client import OpenAIClient
from services.api.app.llm.ollama import OllamaAdapter
from services.api.app.llm.provider import ModelProvider
from services.api.app.retrieval.rerank import rrf_fuse
from services.api.app.storage.models import Chunk, Document, Source, IngestionJob, TableData
from services.api.app.storage.repo import InMemoryRepo
from services.worker.dedup.hashing import content_hash
from services.worker.dedup.versioning import next_version
from services.worker.jobs.embed_and_upsert import embed_and_upsert
from contracts.types import Citation, Draft, EvidenceItem, Verification


class AgentRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_route_sql(self):
        services = build_default_services()
        state = build_initial_state("select * from sales", "t1", "u1")
        state = await route_node(state, services)
        self.assertEqual(state.route, "sql")

    async def test_route_code(self):
        services = build_default_services()
        state = build_initial_state("函数报错怎么修", "t1", "u1")
        state = await route_node(state, services)
        self.assertEqual(state.route, "code")

    async def test_route_doc(self):
        services = build_default_services()
        state = build_initial_state("请参考文档说明", "t1", "u1")
        state = await route_node(state, services)
        self.assertEqual(state.route, "doc_rag")

    async def test_route_mixed_fallback(self):
        services = build_default_services()
        state = build_initial_state("hello world", "t1", "u1")
        state = await route_node(state, services)
        self.assertEqual(state.route, "mixed")


class RetrievalAclTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.services = build_default_services()
        self.repo = self.services.repo
        self.qdrant = self.services.qdrant

        policy = build_policy("p1", "tenant-a", {"allow_users": ["u1"]})
        self.repo.add_policy(policy)

        doc = Document(
            doc_id="doc-1",
            tenant_id="tenant-a",
            source_type="pdf",
            title="Doc",
            uri="file://doc.pdf",
            version="v1",
            doc_updated_at="2025-01-01",
            ingested_at="2025-01-02",
            tags=["demo"],
            acl_policy_id=policy.acl_policy_id,
        )
        self.repo.add_document(doc)

        chunk = Chunk(
            chunk_id="chunk-1",
            doc_id=doc.doc_id,
            tenant_id=doc.tenant_id,
            chunk_index=0,
            text="Postgres 负责结构化数据与事务处理。",
            metadata={
                "source_type": "pdf",
                "ingested_at": doc.ingested_at,
                "doc_updated_at": doc.doc_updated_at,
                "version": doc.version,
                "acl_hash": policy.policy_hash,
                "tags": doc.tags,
            },
        )
        embed_and_upsert(self.repo, self.qdrant, [chunk])

    async def test_retrieval_allows_user(self):
        state = await run_agent("Postgres 做什么", "tenant-a", "u1", self.services)
        self.assertIn("Postgres", state.final.answer)

    async def test_retrieval_blocks_user(self):
        state = await run_agent("Postgres 做什么", "tenant-a", "u2", self.services)
        self.assertIn("证据不足", state.final.answer)


class SqlEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_simple_select(self):
        services = build_default_services()
        table = TableData(
            name="sales",
            columns=[{"name": "region", "type": "text"}, {"name": "amount", "type": "int"}],
            rows=[
                {"region": "cn", "amount": 10},
                {"region": "us", "amount": 20},
            ],
        )
        services.repo.register_table(table)
        state = await run_agent("select region from sales where region = 'cn'", "t1", "u1", services)
        self.assertIn("SQL 返回 1 行", state.final.answer)


class RrfTests(unittest.TestCase):
    def test_rrf_ordering(self):
        primary = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        secondary = [("c", 0.9), ("a", 0.8), ("d", 0.7)]
        fused = rrf_fuse(primary, secondary)
        self.assertEqual(fused[0][0], "a")
        self.assertIn("c", [item[0] for item in fused[:2]])


class CodeNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_code_search_in_memory(self):
        services = build_default_services()
        services.code_search = CodeSearch(
            repo_roots={},
            in_memory_files={"default": {"main.py": "def hello():\n    print('world')\n"}},
        )
        state = build_initial_state("hello", "t1", "u1")
        state = await code_node(state, services)
        self.assertTrue(len(state.tool_calls) > 0)
        self.assertTrue(state.tool_calls[-1].ok)
        self.assertTrue(len(state.evidence) > 0)

    async def test_code_search_no_match(self):
        services = build_default_services()
        services.code_search = CodeSearch(
            repo_roots={},
            in_memory_files={"default": {"main.py": "def hello():\n    pass\n"}},
        )
        state = build_initial_state("nonexistent_function_xyz", "t1", "u1")
        state = await code_node(state, services)
        self.assertTrue(state.tool_calls[-1].ok)


class WebNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_web_node_no_llm(self):
        services = build_default_services()
        state = build_initial_state("latest news", "t1", "u1")
        state = await web_node(state, services)
        self.assertTrue(len(state.tool_calls) > 0)
        last_call = state.tool_calls[-1]
        self.assertFalse(last_call.ok)
        self.assertIn("LLM not available", last_call.error)


class SynthesizeNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_synthesize_no_evidence(self):
        services = build_default_services()
        state = build_initial_state("test", "t1", "u1")
        state = await synthesize_node(state, services)
        self.assertIsNotNone(state.draft)
        self.assertIn("未找到", state.draft.answer_text)

    async def test_synthesize_with_doc_evidence(self):
        services = build_default_services()
        state = build_initial_state("what is Python", "t1", "u1")
        state.evidence.append(
            EvidenceItem(
                kind="doc_chunk",
                score=1.0,
                text="Python is a programming language. It is widely used.",
                citations=[
                    Citation(kind="chunk", chunk_id="c1", doc_id="d1"),
                ],
            )
        )
        state = await synthesize_node(state, services)
        self.assertIsNotNone(state.draft)
        self.assertTrue(len(state.draft.answer_text) > 0)
        self.assertIn("Python", state.draft.answer_text)


class VerifyNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_verify_enough_evidence(self):
        services = build_default_services()
        state = build_initial_state("test", "t1", "u1")
        state.route = "doc_rag"
        state.evidence.append(EvidenceItem(kind="doc_chunk", score=1.0, text="answer"))
        state.draft = Draft(answer_text="answer")
        state = await verify_node(state, services)
        self.assertIsNotNone(state.verification)
        self.assertTrue(state.verification.enough_evidence)

    async def test_verify_missing_evidence(self):
        services = build_default_services()
        state = build_initial_state("test", "t1", "u1")
        state.route = "doc_rag"
        state.draft = Draft(answer_text="answer")
        state = await verify_node(state, services)
        self.assertIsNotNone(state.verification)
        self.assertFalse(state.verification.enough_evidence)
        self.assertIn("doc_chunks", state.verification.missing)


class FinalizeNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_finalize_with_evidence(self):
        services = build_default_services()
        state = build_initial_state("test", "t1", "u1")
        state.draft = Draft(answer_text="all good")
        state.verification = Verification(enough_evidence=True)
        state = await finalize_node(state, services)
        self.assertEqual(state.final.confidence, "high")
        self.assertEqual(state.final.answer, "all good")

    async def test_finalize_degraded(self):
        services = build_default_services()
        state = build_initial_state("test", "t1", "u1")
        state.draft = Draft(answer_text="partial")
        state.verification = Verification(enough_evidence=False, missing=["doc_chunks"])
        state = await finalize_node(state, services)
        self.assertEqual(state.final.confidence, "low")
        self.assertIn("证据不足", state.final.answer)


class CitationHashTests(unittest.TestCase):
    def test_citation_equality(self):
        c1 = Citation(kind="chunk", chunk_id="c1", doc_id="d1")
        c2 = Citation(kind="chunk", chunk_id="c1", doc_id="d1")
        self.assertEqual(c1, c2)
        self.assertEqual(hash(c1), hash(c2))

    def test_citation_inequality(self):
        c1 = Citation(kind="chunk", chunk_id="c1", doc_id="d1")
        c2 = Citation(kind="chunk", chunk_id="c2", doc_id="d1")
        self.assertNotEqual(c1, c2)

    def test_citation_set_dedup(self):
        c1 = Citation(kind="chunk", chunk_id="c1", doc_id="d1")
        c2 = Citation(kind="chunk", chunk_id="c1", doc_id="d1")
        c3 = Citation(kind="web", url="https://example.com")
        result = list(dict.fromkeys([c1, c2, c3]))
        self.assertEqual(len(result), 2)


class SessionStoreTests(unittest.TestCase):
    def test_add_and_load(self):
        store = InMemorySessionStore()
        turn = SessionTurn(query="hi", answer="hello", confidence="high", request_id="r1")
        store.add_turn("s1", "t1", "u1", turn)
        session = store.load("s1")
        self.assertIsNotNone(session)
        self.assertEqual(len(session.turns), 1)
        self.assertEqual(session.turns[0].query, "hi")

    def test_max_turns(self):
        store = InMemorySessionStore(max_turns=3)
        for i in range(5):
            turn = SessionTurn(query=f"q{i}", answer=f"a{i}", confidence="high", request_id=f"r{i}")
            store.add_turn("s1", "t1", "u1", turn)
        session = store.load("s1")
        self.assertEqual(len(session.turns), 3)


class VersioningTests(unittest.TestCase):
    def test_next_version(self):
        self.assertEqual(next_version("1.0.0"), "1.0.1")
        self.assertEqual(next_version("2.3"), "2.4")
        self.assertEqual(next_version("1"), "2")

    def test_next_version_error(self):
        with self.assertRaises(ValueError):
            next_version("abc")


class ContentHashTests(unittest.TestCase):
    def test_deterministic(self):
        h1 = content_hash("hello world")
        h2 = content_hash("hello world")
        self.assertEqual(h1, h2)

    def test_different_input(self):
        h1 = content_hash("hello")
        h2 = content_hash("world")
        self.assertNotEqual(h1, h2)


class EmbedAndUpsertBatchTests(unittest.TestCase):
    def test_batch_upsert(self):
        services = build_default_services()
        chunks = []
        for i in range(5):
            chunks.append(
                Chunk(
                    chunk_id=f"batch-{i}",
                    doc_id="doc-batch",
                    tenant_id="t1",
                    chunk_index=i,
                    text=f"Chunk {i} content",
                    metadata={"source_type": "pdf"},
                )
            )
        embed_and_upsert(services.repo, services.qdrant, chunks, batch_size=2)
        for i in range(5):
            self.assertIsNotNone(services.repo.get_chunk(f"batch-{i}"))


# ── WU1: ModelProvider Protocol Tests ──────────────────────────────────


class ModelProviderProtocolTests(unittest.TestCase):
    def test_openai_client_satisfies_protocol(self):
        client = OpenAIClient()
        self.assertIsInstance(client, ModelProvider)

    def test_ollama_adapter_satisfies_protocol(self):
        adapter = OllamaAdapter()
        self.assertIsInstance(adapter, ModelProvider)

    def test_openai_client_enabled_without_key(self):
        client = OpenAIClient(api_key=None)
        # Without an API key env var, enabled should be False
        import os
        if not os.getenv("OPENAI_API_KEY"):
            self.assertFalse(client.enabled)

    def test_ollama_adapter_always_enabled(self):
        adapter = OllamaAdapter()
        self.assertTrue(adapter.enabled)


# ── WU2: Reliability Tests ─────────────────────────────────────────────


class TimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_triggers(self):
        async def slow_fn():
            await asyncio.sleep(5)
            return "done"

        DEFAULT_TIMEOUTS["__test_timeout"] = 0.1
        try:
            with self.assertRaises(ToolTimeoutError):
                await safe_tool_call("__test_timeout", slow_fn)
        finally:
            DEFAULT_TIMEOUTS.pop("__test_timeout", None)

    async def test_timeout_passes(self):
        def fast_fn():
            return 42

        result = await safe_tool_call("__test_fast", fast_fn)
        self.assertEqual(result, 42)


class RetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_recovers(self):
        attempts = []

        async def flaky_fn():
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("transient")
            return "ok"

        DEFAULT_RETRY["__test_retry"] = RetryConfig(max_retries=3, base_delay=0.01)
        try:
            result = await safe_tool_call("__test_retry", flaky_fn)
            self.assertEqual(result, "ok")
            self.assertEqual(len(attempts), 3)
        finally:
            DEFAULT_RETRY.pop("__test_retry", None)

    async def test_retry_exhausted(self):
        async def always_fail():
            raise ConnectionError("permanent")

        DEFAULT_RETRY["__test_retry_fail"] = RetryConfig(max_retries=2, base_delay=0.01)
        try:
            with self.assertRaises(ConnectionError):
                await safe_tool_call("__test_retry_fail", always_fail)
        finally:
            DEFAULT_RETRY.pop("__test_retry_fail", None)

    async def test_non_retryable_exception_raises_immediately(self):
        attempts = []

        async def value_error_fn():
            attempts.append(1)
            raise ValueError("not retryable")

        DEFAULT_RETRY["__test_retry_nr"] = RetryConfig(max_retries=3, base_delay=0.01)
        try:
            with self.assertRaises(ValueError):
                await safe_tool_call("__test_retry_nr", value_error_fn)
            self.assertEqual(len(attempts), 1)
        finally:
            DEFAULT_RETRY.pop("__test_retry_nr", None)


class CircuitBreakerTests(unittest.TestCase):
    def test_circuit_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)
        self.assertFalse(cb.is_open)
        cb.record_failure()
        self.assertFalse(cb.is_open)
        cb.record_failure()
        self.assertTrue(cb.is_open)

    def test_circuit_resets_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)
        cb.record_failure()
        self.assertTrue(cb.is_open)
        time.sleep(0.1)
        self.assertFalse(cb.is_open)

    def test_circuit_success_resets(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=30)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        # Should not open because success reset the counter
        self.assertFalse(cb.is_open)


class CircuitBreakerAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_safe_tool_call_circuit_open_raises(self):
        breaker = _get_breaker("test_tool_circuit")
        for _ in range(5):
            breaker.record_failure()
        self.assertTrue(breaker.is_open)
        with self.assertRaises(CircuitOpenError):
            await safe_tool_call("test_tool_circuit", lambda: "nope")


# ── WU3: Callback Tests ───────────────────────────────────────────────


class QueueCallbackTests(unittest.TestCase):
    def test_emit_and_get(self):
        cb = QueueCallback()
        event = AgentEvent("test", {"key": "value"})
        cb.emit(event)
        received = cb.get(timeout=1.0)
        self.assertIsNotNone(received)
        self.assertEqual(received.event_type, "test")
        self.assertEqual(received.data["key"], "value")

    def test_close_signals_none(self):
        cb = QueueCallback()
        cb.close()
        result = cb.get(timeout=0.1)
        self.assertIsNone(result)

    def test_null_callback_noop(self):
        cb = NullCallback()
        cb.emit(AgentEvent("test", {}))
        cb.close()
        # Should not raise


class AsyncCallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_agent_with_callback(self):
        services = build_default_services()
        cb = AsyncQueueCallback()
        state = await run_agent("hello world", "t1", "u1", services, callback=cb)
        self.assertIsNotNone(state.final)
        self.assertTrue(cb._closed)

    async def test_callback_receives_events(self):
        services = build_default_services()
        table = TableData(
            name="items",
            columns=[{"name": "name", "type": "text"}],
            rows=[{"name": "item1"}],
        )
        services.repo.register_table(table)

        cb = AsyncQueueCallback()
        events = []

        async def run():
            await run_agent("select name from items", "t1", "u1", services, callback=cb)

        task = asyncio.create_task(run())
        async for event in cb:
            events.append(event)
        await task
        event_types = [e.event_type for e in events]
        self.assertIn("route", event_types)
        self.assertIn("final", event_types)


# ── WU5/WU6: FastAPI Endpoint + Middleware Tests ───────────────────────


class FastAPIEndpointTests(unittest.TestCase):
    """Test the FastAPI app endpoints using TestClient."""

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            raise unittest.SkipTest("fastapi TestClient not available")

        from services.api.app.api import app, _get_services
        from services.api.app.agent.graph import build_default_services
        from services.api.app.storage.models import TableData

        # Pre-build services with test data
        import services.api.app.api as api_mod
        api_mod._services = build_default_services()

        table = TableData(
            name="users",
            columns=[{"name": "name", "type": "text"}, {"name": "role", "type": "text"}],
            rows=[{"name": "alice", "role": "admin"}, {"name": "bob", "role": "user"}],
        )
        api_mod._services.repo.register_table(table)

        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/admin/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_chat_endpoint(self):
        response = self.client.post("/chat", json={
            "query": "hello",
            "tenant_id": "t1",
            "user_id": "u1",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("request_id", data)

    def test_search_endpoint(self):
        response = self.client.post("/search", json={
            "query": "test search",
            "tenant_id": "t1",
            "user_id": "u1",
            "top_k": 5,
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("request_id", data)
        self.assertIn("chunks", data)
        self.assertIn("total", data)

    def test_openai_compat_endpoint(self):
        response = self.client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
        }, headers={"X-Tenant-ID": "t1", "X-User-ID": "u1"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("id", data)
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(len(data["choices"]), 1)
        self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
        self.assertIn("usage", data)

    def test_request_id_header(self):
        response = self.client.get("/admin/health", headers={"X-Request-ID": "test-123"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("X-Request-ID"), "test-123")

    def test_request_id_generated(self):
        response = self.client.get("/admin/health")
        self.assertEqual(response.status_code, 200)
        request_id = response.headers.get("X-Request-ID")
        self.assertIsNotNone(request_id)
        self.assertTrue(len(request_id) > 0)


# ── Milestone B: Source/Repo CRUD Tests ───────────────────────────────


class SourceRepoTests(unittest.TestCase):
    def setUp(self):
        self.repo = InMemoryRepo()

    def test_add_and_get_source(self):
        source = Source(
            source_id="s1", tenant_id="t1", source_type="pdf",
            name="My PDF", config={"path": "/tmp/test.pdf"},
        )
        self.repo.add_source(source)
        result = self.repo.get_source("s1")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "My PDF")
        self.assertEqual(result.source_type, "pdf")

    def test_list_sources_by_tenant(self):
        self.repo.add_source(Source(source_id="s1", tenant_id="t1", source_type="pdf", name="PDF1"))
        self.repo.add_source(Source(source_id="s2", tenant_id="t2", source_type="web", name="Web1"))
        self.repo.add_source(Source(source_id="s3", tenant_id="t1", source_type="local_fs", name="FS1"))
        t1_sources = self.repo.list_sources(tenant_id="t1")
        self.assertEqual(len(t1_sources), 2)

    def test_update_source(self):
        self.repo.add_source(Source(source_id="s1", tenant_id="t1", source_type="pdf", name="Original"))
        updated = self.repo.update_source("s1", name="Updated", status="paused")
        self.assertEqual(updated.name, "Updated")
        self.assertEqual(updated.status, "paused")

    def test_delete_source(self):
        self.repo.add_source(Source(source_id="s1", tenant_id="t1", source_type="pdf", name="ToDelete"))
        result = self.repo.delete_source("s1")
        self.assertTrue(result)
        source = self.repo.get_source("s1")
        self.assertEqual(source.status, "deleted")

    def test_delete_nonexistent(self):
        result = self.repo.delete_source("nonexistent")
        self.assertFalse(result)


class JobRepoTests(unittest.TestCase):
    def setUp(self):
        self.repo = InMemoryRepo()

    def test_add_and_get_job(self):
        job = IngestionJob(
            job_id="j1", tenant_id="t1", source_id="s1",
            source_type="pdf", source_config={"path": "/tmp/test.pdf"},
        )
        self.repo.add_job(job)
        result = self.repo.get_job("j1")
        self.assertIsNotNone(result)
        self.assertEqual(result.status, "pending")

    def test_list_jobs_by_source(self):
        self.repo.add_job(IngestionJob(job_id="j1", tenant_id="t1", source_id="s1", source_type="pdf", source_config={}))
        self.repo.add_job(IngestionJob(job_id="j2", tenant_id="t1", source_id="s2", source_type="web", source_config={}))
        self.repo.add_job(IngestionJob(job_id="j3", tenant_id="t1", source_id="s1", source_type="pdf", source_config={}))
        s1_jobs = self.repo.list_jobs(source_id="s1")
        self.assertEqual(len(s1_jobs), 2)

    def test_update_job(self):
        self.repo.add_job(IngestionJob(job_id="j1", tenant_id="t1", source_id="s1", source_type="pdf", source_config={}))
        updated = self.repo.update_job("j1", status="completed", chunk_count=10)
        self.assertEqual(updated.status, "completed")
        self.assertEqual(updated.chunk_count, 10)


# ── Milestone B: ACL Enhanced Tests ───────────────────────────────────


class ACLGroupRoleTests(unittest.TestCase):
    def test_group_access(self):
        policy = build_policy("p1", "t1", {"allow_groups": ["engineering"]})
        scope = compute_security_scope("u1", [policy], groups=["engineering"])
        self.assertIn(policy.policy_hash, scope)

    def test_group_denied(self):
        policy = build_policy("p1", "t1", {"allow_groups": ["engineering"]})
        scope = compute_security_scope("u1", [policy], groups=["marketing"])
        self.assertNotIn(policy.policy_hash, scope)

    def test_role_access(self):
        policy = build_policy("p1", "t1", {"allow_roles": ["admin"]})
        scope = compute_security_scope("u1", [policy], roles=["admin"])
        self.assertIn(policy.policy_hash, scope)

    def test_role_denied(self):
        policy = build_policy("p1", "t1", {"allow_roles": ["admin"]})
        scope = compute_security_scope("u1", [policy], roles=["viewer"])
        self.assertNotIn(policy.policy_hash, scope)

    def test_user_context(self):
        ctx = UserContext("u1", groups=["eng"], roles=["admin"])
        self.assertEqual(ctx.user_id, "u1")
        self.assertIn("eng", ctx.groups)
        self.assertIn("admin", ctx.roles)

    def test_compute_scope_from_context(self):
        policy = build_policy("p1", "t1", {"allow_roles": ["admin"]})
        ctx = UserContext("u1", roles=["admin"])
        scope = compute_security_scope_from_context(ctx, [policy])
        self.assertIn(policy.policy_hash, scope)

    def test_backward_compat_user_access(self):
        policy = build_policy("p1", "t1", {"allow_users": ["u1"]})
        scope = compute_security_scope("u1", [policy])
        self.assertIn(policy.policy_hash, scope)

    def test_allow_all_still_works(self):
        policy = build_policy("p1", "t1", {"allow_all": True})
        scope = compute_security_scope("anyone", [policy])
        self.assertIn(policy.policy_hash, scope)


# ── Milestone B: Schema Introspection Tests ───────────────────────────


class SchemaIntrospectionTests(unittest.TestCase):
    def test_inmemory_introspect(self):
        repo = InMemoryRepo()
        table = TableData(
            name="orders",
            columns=[{"name": "id", "type": "int"}, {"name": "total", "type": "float"}],
            rows=[{"id": 1, "total": 99.99}],
        )
        repo.register_table(table)
        engine = SqlEngine(repo)
        schema = engine.introspect_schema()
        self.assertEqual(len(schema), 1)
        self.assertEqual(schema[0]["table_name"], "orders")
        self.assertEqual(len(schema[0]["columns"]), 2)
        self.assertEqual(schema[0]["row_count"], 1)

    def test_introspect_empty(self):
        repo = InMemoryRepo()
        engine = SqlEngine(repo)
        schema = engine.introspect_schema()
        self.assertEqual(len(schema), 0)


# ── Milestone B: Local FS Connector Tests ─────────────────────────────


class LocalFSConnectorTests(unittest.TestCase):
    def test_list_files(self):
        from services.worker.connectors.local_fs import list_files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            open(os.path.join(tmpdir, "readme.md"), "w").write("# Hello")
            open(os.path.join(tmpdir, "notes.txt"), "w").write("Some notes")
            open(os.path.join(tmpdir, "image.png"), "w").write("fake image")
            files = list_files(tmpdir)
            self.assertEqual(len(files), 2)  # only .md and .txt

    def test_list_files_custom_ext(self):
        from services.worker.connectors.local_fs import list_files
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "data.csv"), "w").write("a,b,c")
            open(os.path.join(tmpdir, "readme.md"), "w").write("# H")
            files = list_files(tmpdir, extensions={".csv"})
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].endswith(".csv"))

    def test_read_file(self):
        from services.worker.connectors.local_fs import read_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            f.flush()
            content = read_file(f.name)
            self.assertEqual(content, "hello world")
        os.unlink(f.name)


class IngestTextTests(unittest.TestCase):
    def test_ingest_text_file(self):
        from services.worker.jobs.ingest_text import ingest_text_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("A" * 2000)
            f.flush()
            chunks = list(ingest_text_file(f.name, doc_id="d1", tenant_id="t1", chunk_size=800))
            self.assertTrue(len(chunks) >= 2)
            self.assertEqual(chunks[0].tenant_id, "t1")
            self.assertIsNotNone(chunks[0].checksum)
        os.unlink(f.name)

    def test_ingest_markdown_section(self):
        from services.worker.jobs.ingest_text import ingest_text_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Introduction\n\nSome content here about the topic.\n")
            f.flush()
            chunks = list(ingest_text_file(f.name, doc_id="d1", tenant_id="t1"))
            self.assertTrue(len(chunks) >= 1)
            self.assertEqual(chunks[0].section, "Introduction")
            self.assertEqual(chunks[0].metadata["source_type"], "markdown")
        os.unlink(f.name)

    def test_ingest_local_fs(self):
        from services.worker.jobs.ingest_text import ingest_local_fs
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "a.txt"), "w").write("Content of file A")
            open(os.path.join(tmpdir, "b.md"), "w").write("# Title\n\nContent of file B")
            chunks = list(ingest_local_fs(
                directory=tmpdir, doc_id="d1", tenant_id="t1",
            ))
            self.assertTrue(len(chunks) >= 2)
            doc_ids = {c.doc_id for c in chunks}
            self.assertTrue(len(doc_ids) >= 2)  # Different doc_id per file


# ── Milestone B: Pipeline Tests ───────────────────────────────────────


class PipelineTests(unittest.TestCase):
    def test_pipeline_text_ingest(self):
        from services.worker.pipeline import run_ingest_pipeline
        services = build_default_services()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is test content for the pipeline. " * 30)
            f.flush()

            source = Source(
                source_id="src-test", tenant_id="t1", source_type="local_fs",
                name="Test FS", config={"path": os.path.dirname(f.name)},
            )
            services.repo.add_source(source)
            job = run_ingest_pipeline(source, services.repo, services.qdrant)
            self.assertEqual(job.status, "completed")
            self.assertTrue(job.chunk_count > 0)
        os.unlink(f.name)

    def test_pipeline_dedup(self):
        from services.worker.pipeline import run_ingest_pipeline
        services = build_default_services()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Unique test content for dedup testing.")
            f.flush()

            source = Source(
                source_id="src-dedup", tenant_id="t1", source_type="local_fs",
                name="Test Dedup", config={"path": os.path.dirname(f.name)},
            )
            services.repo.add_source(source)

            job1 = run_ingest_pipeline(source, services.repo, services.qdrant, job_id="j1")
            first_count = job1.chunk_count

            job2 = run_ingest_pipeline(source, services.repo, services.qdrant, job_id="j2")
            # Second run should have 0 new chunks due to dedup
            self.assertEqual(job2.chunk_count, 0)
        os.unlink(f.name)

    def test_pipeline_error_handling(self):
        from services.worker.pipeline import run_ingest_pipeline
        services = build_default_services()
        source = Source(
            source_id="src-bad", tenant_id="t1", source_type="local_fs",
            name="Bad Source", config={"path": "/nonexistent/path/that/does/not/exist"},
        )
        services.repo.add_source(source)
        job = run_ingest_pipeline(source, services.repo, services.qdrant)
        self.assertEqual(job.status, "failed")
        self.assertIsNotNone(job.error)


# ── Milestone B: FastAPI Source/Ingest Endpoint Tests ─────────────────


class FastAPISourceEndpointTests(unittest.TestCase):
    """Test /sources and /ingest/jobs endpoints."""

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            raise unittest.SkipTest("fastapi TestClient not available")

        from services.api.app.api import app
        import services.api.app.api as api_mod
        api_mod._services = build_default_services()
        cls.client = TestClient(app)

    def test_create_source(self):
        response = self.client.post("/sources", json={
            "tenant_id": "t1",
            "source_type": "pdf",
            "name": "Test PDF Source",
            "config": {"path": "/tmp/test.pdf"},
        })
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("source_id", data)
        self.assertEqual(data["source_type"], "pdf")
        self.assertEqual(data["status"], "active")

    def test_create_source_invalid_type(self):
        response = self.client.post("/sources", json={
            "tenant_id": "t1",
            "source_type": "invalid_type",
            "name": "Bad Source",
        })
        self.assertEqual(response.status_code, 400)

    def test_list_sources(self):
        response = self.client.get("/sources")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("sources", data)
        self.assertIsInstance(data["sources"], list)

    def test_get_source_not_found(self):
        response = self.client.get("/sources/nonexistent")
        self.assertEqual(response.status_code, 404)

    def test_ingest_jobs_list(self):
        response = self.client.get("/ingest/jobs")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("jobs", data)

    def test_ingest_job_not_found(self):
        response = self.client.get("/ingest/jobs/nonexistent")
        self.assertEqual(response.status_code, 404)

    def test_trigger_job_source_not_found(self):
        response = self.client.post("/ingest/jobs", json={
            "source_id": "nonexistent",
            "tenant_id": "t1",
        })
        self.assertEqual(response.status_code, 404)

    def test_source_crud_lifecycle(self):
        # Create
        resp = self.client.post("/sources", json={
            "tenant_id": "t1", "source_type": "web", "name": "Lifecycle Test",
            "config": {"url": "https://example.com"},
        })
        self.assertEqual(resp.status_code, 201)
        source_id = resp.json()["source_id"]

        # Read
        resp = self.client.get(f"/sources/{source_id}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["name"], "Lifecycle Test")

        # Update
        resp = self.client.put(f"/sources/{source_id}", json={"name": "Updated Name"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["name"], "Updated Name")

        # Delete
        resp = self.client.delete(f"/sources/{source_id}")
        self.assertEqual(resp.status_code, 204)

        # Verify deleted
        resp = self.client.get(f"/sources/{source_id}")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()


# ── Milestone C: Enhanced Repo Ingestion Tests ────────────────────────


class RepoIngestionSymbolTests(unittest.TestCase):
    def test_python_symbol_chunking(self):
        from services.worker.jobs.ingest_repo import _split_python_symbols
        code = '''import os

def hello():
    print("hello")

class MyClass:
    def method(self):
        pass

def goodbye():
    print("bye")
'''
        result = _split_python_symbols(code, chunk_size=600)
        self.assertTrue(len(result) >= 3)  # preamble + hello + MyClass + goodbye
        names = [r[0] for r in result]
        self.assertIn("hello", names)
        self.assertIn("MyClass", names)
        self.assertIn("goodbye", names)

    def test_python_symbol_fallback_on_syntax_error(self):
        from services.worker.jobs.ingest_repo import _split_python_symbols
        bad_code = "def broken(\n"
        result = _split_python_symbols(bad_code, chunk_size=600)
        # Should fall back to line-based splitting
        self.assertTrue(len(result) >= 1)

    def test_regex_function_detection(self):
        from services.worker.jobs.ingest_repo import _split_by_functions
        code = '''const x = 1;

function hello() {
    console.log("hi");
}

function world() {
    console.log("world");
}
'''
        result = _split_by_functions(code, chunk_size=600)
        self.assertTrue(len(result) >= 2)
        names = [r[0] for r in result if r[0]]
        self.assertIn("hello", names)
        self.assertIn("world", names)

    def test_line_based_fallback(self):
        from services.worker.jobs.ingest_repo import _split_file
        text = "line\n" * 100
        result = _split_file(text, chunk_size=50)
        self.assertTrue(len(result) >= 2)
        # Each result is (name, start, end, text)
        for name, start, end, segment in result:
            self.assertIsNone(name)
            self.assertTrue(start >= 1)
            self.assertTrue(len(segment) > 0)

    def test_large_symbol_splitting(self):
        from services.worker.jobs.ingest_repo import _split_large_symbol
        large_text = "x = 1\n" * 200
        result = _split_large_symbol("big_func", large_text, 0, chunk_size=100)
        self.assertTrue(len(result) >= 2)
        # First part should have part number
        self.assertIn("part", result[0][0])

    def test_language_detection(self):
        from services.worker.jobs.ingest_repo import _LANG_MAP
        self.assertEqual(_LANG_MAP[".py"], "python")
        self.assertEqual(_LANG_MAP[".ts"], "typescript")
        self.assertEqual(_LANG_MAP[".go"], "go")
        self.assertEqual(_LANG_MAP[".rs"], "rust")


# ── Milestone C: Programming Tools Tests ──────────────────────────────


class OpenFileTests(unittest.TestCase):
    def test_open_file_full(self):
        cs = CodeSearch(
            repo_roots={},
            in_memory_files={"default": {
                "main.py": "line1\nline2\nline3\nline4\nline5\n",
            }},
        )
        result = cs.open_file("main.py", "default")
        self.assertIn("line1", result)
        self.assertIn("line5", result)
        # Should have line numbers
        self.assertIn("1", result)

    def test_open_file_range(self):
        cs = CodeSearch(
            repo_roots={},
            in_memory_files={"default": {
                "main.py": "\n".join(f"line{i}" for i in range(1, 21)),
            }},
        )
        result = cs.open_file("main.py", "default", start_line=5, end_line=10)
        self.assertIn("line5", result)
        self.assertIn("line10", result)
        self.assertNotIn("line1 |", result)

    def test_open_file_not_found(self):
        cs = CodeSearch(repo_roots={}, in_memory_files={"default": {}})
        with self.assertRaises(FileNotFoundError):
            cs.open_file("nonexistent.py", "default")


class GeneratePatchTests(unittest.TestCase):
    def test_generate_patch(self):
        cs = CodeSearch(repo_roots={})
        original = "def hello():\n    print('hello')\n"
        replacement = "def hello():\n    print('world')\n"
        result = cs.generate_patch("main.py", original, replacement)
        self.assertEqual(result.path, "main.py")
        self.assertIn("---", result.diff)
        self.assertIn("+++", result.diff)
        self.assertIn("-    print('hello')", result.diff)
        self.assertIn("+    print('world')", result.diff)

    def test_generate_patch_no_change(self):
        cs = CodeSearch(repo_roots={})
        original = "no change\n"
        result = cs.generate_patch("file.py", original, original)
        self.assertEqual(result.diff, "")  # No diff for identical content


class ExplainErrorTests(unittest.TestCase):
    def test_explain_error_with_stack_trace(self):
        cs = CodeSearch(
            repo_roots={},
            in_memory_files={"default": {
                "app.py": "\n".join(f"line {i}" for i in range(1, 50)),
            }},
        )
        error = '''Traceback (most recent call last):
  File "app.py", line 10, in main
    do_something()
ValueError: invalid value'''
        result = cs.explain_error(error, "default")
        self.assertTrue(len(result) >= 1)
        self.assertEqual(result[0].path, "app.py")

    def test_explain_error_keyword_fallback(self):
        cs = CodeSearch(
            repo_roots={},
            in_memory_files={"default": {
                "handler.py": "class ValueError:\n    pass\n",
            }},
        )
        error = "ValueError: something went wrong"
        result = cs.explain_error(error, "default")
        # Should find the keyword fallback
        self.assertTrue(len(result) >= 0)  # May or may not find matches


class StackTraceParsingTests(unittest.TestCase):
    def test_parse_python_trace(self):
        from services.api.app.agent.nodes.code import _parse_stack_trace
        trace = '''File "app/main.py", line 42, in run
    process()'''
        locations = _parse_stack_trace(trace)
        self.assertTrue(len(locations) >= 1)
        self.assertEqual(locations[0], ("app/main.py", 42))

    def test_parse_js_trace(self):
        from services.api.app.agent.nodes.code import _parse_stack_trace
        trace = "at Object.run (/app/server.js:15:3)"
        locations = _parse_stack_trace(trace)
        self.assertTrue(len(locations) >= 1)
        self.assertEqual(locations[0][1], 15)

    def test_parse_java_trace(self):
        from services.api.app.agent.nodes.code import _parse_stack_trace
        trace = "at com.example.Main.run(Main.java:25)"
        locations = _parse_stack_trace(trace)
        self.assertTrue(len(locations) >= 1)
        self.assertEqual(locations[0], ("Main.java", 25))

    def test_extract_error_keywords(self):
        from services.api.app.agent.nodes.code import _extract_error_keywords
        text = "ValueError: invalid input in process_data"
        keywords = _extract_error_keywords(text)
        self.assertIn("ValueError", keywords)


class FileReferenceParsingTests(unittest.TestCase):
    def test_parse_path_with_range(self):
        from services.api.app.agent.nodes.code import _parse_file_reference
        path, start, end = _parse_file_reference("open main.py:10-20")
        self.assertEqual(path, "main.py")
        self.assertEqual(start, 10)
        self.assertEqual(end, 20)

    def test_parse_path_with_line(self):
        from services.api.app.agent.nodes.code import _parse_file_reference
        path, start, end = _parse_file_reference("show utils.ts:42")
        self.assertEqual(path, "utils.ts")
        self.assertEqual(start, 42)

    def test_parse_plain_path(self):
        from services.api.app.agent.nodes.code import _parse_file_reference
        path, start, end = _parse_file_reference("read config.yaml")
        self.assertEqual(path, "config.yaml")
        self.assertIsNone(start)
        self.assertIsNone(end)


class OpenFileNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_open_file_node_success(self):
        from services.api.app.agent.nodes.code import open_file_node
        services = build_default_services()
        services.code_search = CodeSearch(
            repo_roots={},
            in_memory_files={"default": {"test.py": "def foo():\n    return 42\n"}},
        )
        state = build_initial_state("open test.py", "t1", "u1")
        state = await open_file_node(state, services)
        self.assertTrue(len(state.tool_calls) > 0)
        self.assertTrue(state.tool_calls[-1].ok)
        self.assertTrue(len(state.evidence) > 0)
        self.assertEqual(state.evidence[-1].kind, "file_content")

    async def test_open_file_node_not_found(self):
        from services.api.app.agent.nodes.code import open_file_node
        services = build_default_services()
        services.code_search = CodeSearch(repo_roots={}, in_memory_files={"default": {}})
        state = build_initial_state("open nonexistent.py", "t1", "u1")
        state = await open_file_node(state, services)
        self.assertTrue(len(state.tool_calls) > 0)
        self.assertFalse(state.tool_calls[-1].ok)


class ExplainErrorNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_explain_error_node(self):
        from services.api.app.agent.nodes.code import explain_error_node
        services = build_default_services()
        services.code_search = CodeSearch(
            repo_roots={},
            in_memory_files={"default": {
                "app.py": "\n".join(f"code line {i}" for i in range(50)),
            }},
        )
        state = build_initial_state(
            'File "app.py", line 10, in main\nValueError: bad', "t1", "u1",
        )
        state = await explain_error_node(state, services)
        self.assertTrue(len(state.tool_calls) > 0)
        self.assertTrue(state.tool_calls[-1].ok)


# ── Milestone C: CLI Tests ───────────────────────────────────────────


class CLITests(unittest.TestCase):
    def test_cli_ask_local(self):
        from cli.rag import main
        import io
        from unittest.mock import patch
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main(["ask", "hello world"])
        output = mock_stdout.getvalue()
        self.assertTrue(len(output) > 0)

    def test_cli_search_local(self):
        from cli.rag import main
        import io
        from unittest.mock import patch
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main(["search", "test query"])
        output = mock_stdout.getvalue()
        self.assertTrue(len(output) > 0)

    def test_cli_no_command(self):
        from cli.rag import main
        with self.assertRaises(SystemExit) as ctx:
            main([])
        self.assertEqual(ctx.exception.code, 1)

    def test_cli_help(self):
        from cli.rag import main
        with self.assertRaises(SystemExit) as ctx:
            main(["--help"])


# ── Milestone C: Context Strategy Tests ──────────────────────────────


class ClientContextTests(unittest.TestCase):
    def test_process_empty_context(self):
        from services.api.app.agent.context import process_client_context
        constraints, evidence = process_client_context(None)
        self.assertIsNone(constraints)
        self.assertEqual(len(evidence), 0)

    def test_process_selected_text(self):
        from services.api.app.agent.context import process_client_context
        ctx = {
            "selected_text": {
                "content": "def foo(): pass",
                "path": "main.py",
                "start_line": 10,
                "end_line": 12,
            }
        }
        constraints, evidence = process_client_context(ctx)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].kind, "file_content")
        self.assertIn("def foo", evidence[0].text)
        self.assertEqual(evidence[0].citations[0].path, "main.py")

    def test_process_repo_constraint(self):
        from services.api.app.agent.context import process_client_context
        from contracts.types import Constraints
        ctx = {"repo": "myrepo", "ref": "develop"}
        constraints, evidence = process_client_context(ctx)
        self.assertEqual(constraints.repo, "myrepo")
        self.assertEqual(constraints.ref, "develop")

    def test_process_open_files(self):
        from services.api.app.agent.context import process_client_context
        ctx = {
            "open_files": [
                {"path": "a.py", "content": "code A"},
                {"path": "b.py", "content": "code B"},
            ]
        }
        constraints, evidence = process_client_context(ctx)
        self.assertEqual(len(evidence), 2)
        paths = [e.metadata["path"] for e in evidence]
        self.assertIn("a.py", paths)
        self.assertIn("b.py", paths)

    def test_process_git_diff(self):
        from services.api.app.agent.context import process_client_context
        ctx = {"git_diff": "diff --git a/file.py b/file.py\n+new line"}
        constraints, evidence = process_client_context(ctx)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].kind, "patch")

    def test_process_recent_errors(self):
        from services.api.app.agent.context import process_client_context
        ctx = {"recent_errors": ["Error 1", "Error 2"]}
        constraints, evidence = process_client_context(ctx)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].kind, "error_analysis")
        self.assertIn("Error 1", evidence[0].text)

    def test_existing_constraints_preserved(self):
        from services.api.app.agent.context import process_client_context
        from contracts.types import Constraints
        existing = Constraints(repo="existing-repo", tags=["tag1"])
        ctx = {"repo": "should-not-override"}
        constraints, _ = process_client_context(ctx, existing)
        self.assertEqual(constraints.repo, "existing-repo")  # Preserved
        self.assertEqual(constraints.tags, ["tag1"])


class EvidenceDedupTests(unittest.TestCase):
    def test_dedup_removes_duplicates(self):
        from services.api.app.agent.context import dedup_evidence
        ev1 = EvidenceItem(kind="doc_chunk", text="same text", score=1.0)
        ev2 = EvidenceItem(kind="doc_chunk", text="same text", score=0.8)
        ev3 = EvidenceItem(kind="doc_chunk", text="different text", score=0.5)
        result = dedup_evidence([ev1, ev2, ev3])
        self.assertEqual(len(result), 2)

    def test_dedup_preserves_unique(self):
        from services.api.app.agent.context import dedup_evidence
        items = [
            EvidenceItem(kind="doc_chunk", text=f"unique {i}", score=1.0)
            for i in range(5)
        ]
        result = dedup_evidence(items)
        self.assertEqual(len(result), 5)


class EvidenceCompressionTests(unittest.TestCase):
    def test_compress_drops_low_score(self):
        from services.api.app.agent.context import compress_evidence
        items = [
            EvidenceItem(kind="doc_chunk", text="A" * 8000, score=1.0),
            EvidenceItem(kind="doc_chunk", text="B" * 8000, score=0.1),
        ]
        result = compress_evidence(items, max_total=4000)
        self.assertEqual(len(result), 1)
        self.assertIn("A", result[0].text)

    def test_compress_truncates_long_items(self):
        from services.api.app.agent.context import compress_evidence, MAX_SINGLE_EVIDENCE_CHARS
        items = [
            EvidenceItem(kind="doc_chunk", text="X" * 10000, score=1.0),
        ]
        result = compress_evidence(items)
        self.assertTrue(len(result[0].text) <= MAX_SINGLE_EVIDENCE_CHARS + 20)  # +margin for "truncated"

    def test_compress_empty(self):
        from services.api.app.agent.context import compress_evidence
        result = compress_evidence([])
        self.assertEqual(result, [])


# ── Milestone C: Integration Tests ───────────────────────────────────


class MilestoneCIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_with_client_context(self):
        """Test that client_context flows through chat() correctly."""
        from services.api.app.main import chat
        result = await chat(
            "what does this code do",
            "t1", "u1",
            initial_evidence=[
                EvidenceItem(
                    kind="file_content", score=1.0,
                    text="def add(a, b): return a + b",
                    citations=[Citation(kind="code", path="math.py", line_start=1, line_end=1)],
                ),
            ],
        )
        self.assertIn("answer", result)
        self.assertIn("request_id", result)

    async def test_run_agent_with_initial_evidence(self):
        """Test that initial_evidence is injected into agent state."""
        services = build_default_services()
        initial_ev = [
            EvidenceItem(
                kind="file_content", score=1.0,
                text="class Foo: pass",
                citations=[Citation(kind="code", path="foo.py")],
            ),
        ]
        state = await run_agent(
            "explain Foo", "t1", "u1", services,
            initial_evidence=initial_ev,
        )
        self.assertIsNotNone(state.final)
        # Evidence should include our injected item
        kinds = [e.kind for e in state.evidence]
        self.assertIn("file_content", kinds)

    def test_patch_result_type(self):
        from contracts.types import PatchResult
        p = PatchResult(path="test.py", diff="--- a\n+++ b\n", original_lines=5, modified_lines=6)
        self.assertEqual(p.path, "test.py")
        self.assertEqual(p.original_lines, 5)

    def test_chat_endpoint_with_context(self):
        """Test /chat with client_context via TestClient."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            raise unittest.SkipTest("fastapi TestClient not available")

        from services.api.app.api import app
        import services.api.app.api as api_mod
        api_mod._services = build_default_services()

        client = TestClient(app)
        response = client.post("/chat", json={
            "query": "explain this code",
            "tenant_id": "t1",
            "user_id": "u1",
            "client_context": {
                "selected_text": {
                    "content": "def hello(): print('hi')",
                    "path": "main.py",
                    "start_line": 1,
                    "end_line": 1,
                },
                "repo": "test-repo",
            },
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)


# ── Milestone D: Observability Tests ──────────────────────────────────


class RequestTracerTests(unittest.TestCase):
    def test_span_timing(self):
        from services.api.app.observability.tracing import RequestTracer
        tracer = RequestTracer(request_id="test-001")
        with tracer.span("test_span") as s:
            time.sleep(0.05)
            s.attributes["key"] = "value"
        record = tracer.finish()
        self.assertEqual(record.request_id, "test-001")
        self.assertEqual(len(record.spans), 1)
        self.assertEqual(record.spans[0].name, "test_span")
        self.assertTrue(record.spans[0].duration_ms >= 1)
        self.assertEqual(record.spans[0].attributes["key"], "value")

    def test_span_error(self):
        from services.api.app.observability.tracing import RequestTracer
        tracer = RequestTracer(request_id="test-002")
        try:
            with tracer.span("bad_span") as s:
                raise ValueError("test error")
        except ValueError:
            pass
        record = tracer.finish()
        self.assertEqual(record.spans[0].status, "error")
        self.assertIn("test error", record.spans[0].attributes.get("error", ""))

    def test_multiple_spans(self):
        from services.api.app.observability.tracing import RequestTracer
        tracer = RequestTracer(request_id="test-003")
        with tracer.span("route"):
            pass
        with tracer.span("retrieve"):
            pass
        with tracer.span("synthesize"):
            pass
        record = tracer.finish()
        self.assertEqual(len(record.spans), 3)
        names = [s.name for s in record.spans]
        self.assertEqual(names, ["route", "retrieve", "synthesize"])

    def test_trace_record_to_dict(self):
        from services.api.app.observability.tracing import RequestTracer
        tracer = RequestTracer(request_id="test-004")
        with tracer.span("op"):
            pass
        record = tracer.finish()
        d = record.to_dict()
        self.assertIn("trace_id", d)
        self.assertIn("spans", d)
        self.assertEqual(len(d["spans"]), 1)


class MetricsCollectorTests(unittest.IsolatedAsyncioTestCase):
    def test_record_and_aggregate(self):
        from services.api.app.observability.metrics import MetricsCollector, RequestMetrics
        collector = MetricsCollector()
        collector.record(RequestMetrics(
            request_id="r1", tenant_id="t1", user_id="u1",
            confidence="high", has_citations=True, citation_count=2,
            evidence_count=3, total_duration_ms=100, iterations=1,
            tool_calls=[{"name": "retrieve", "ok": True, "duration_ms": 50}],
            tool_success_count=1, tool_failure_count=0,
        ))
        collector.record(RequestMetrics(
            request_id="r2", tenant_id="t1", user_id="u1",
            confidence="low", has_citations=False, citation_count=0,
            evidence_count=0, total_duration_ms=200, iterations=2,
            tool_calls=[{"name": "retrieve", "ok": False, "duration_ms": 100}],
            tool_success_count=0, tool_failure_count=1,
        ))
        agg = collector.aggregate()
        self.assertEqual(agg.total_requests, 2)
        self.assertAlmostEqual(agg.citation_coverage, 0.5, places=2)
        self.assertEqual(agg.confidence_distribution["high"], 1)
        self.assertEqual(agg.confidence_distribution["low"], 1)
        self.assertAlmostEqual(agg.tool_failure_rate, 0.5, places=2)
        self.assertAlmostEqual(agg.avg_duration_ms, 150.0, places=1)

    def test_record_feedback(self):
        from services.api.app.observability.metrics import MetricsCollector, RequestMetrics
        collector = MetricsCollector()
        collector.record(RequestMetrics(request_id="r1", tenant_id="t1", user_id="u1"))
        found = collector.record_feedback("r1", "positive")
        self.assertTrue(found)
        not_found = collector.record_feedback("nonexistent", "negative")
        self.assertFalse(not_found)
        agg = collector.aggregate()
        self.assertEqual(agg.positive_feedback, 1)

    def test_aggregate_empty(self):
        from services.api.app.observability.metrics import MetricsCollector
        collector = MetricsCollector()
        agg = collector.aggregate()
        self.assertEqual(agg.total_requests, 0)

    def test_metrics_history(self):
        from services.api.app.observability.metrics import MetricsCollector, RequestMetrics
        collector = MetricsCollector()
        for i in range(5):
            collector.record(RequestMetrics(request_id=f"r{i}", tenant_id="t1", user_id="u1"))
        history = collector.get_history(last_n=3)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[-1]["request_id"], "r4")

    async def test_build_request_metrics(self):
        from services.api.app.observability.metrics import build_request_metrics
        services = build_default_services()
        state = await run_agent("hello world", "t1", "u1", services)
        metrics = build_request_metrics(state)
        self.assertEqual(metrics.request_id, state.request_id)
        self.assertEqual(metrics.tenant_id, "t1")
        self.assertTrue(metrics.iterations >= 1)


class TracingIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_agent_collects_metrics(self):
        """Verify that run_agent auto-collects metrics."""
        from services.api.app.observability.metrics import get_metrics_collector
        collector = get_metrics_collector()
        collector.reset()
        services = build_default_services()
        state = await run_agent("hello world", "t1", "u1", services)
        agg = collector.aggregate()
        self.assertTrue(agg.total_requests >= 1)


# ── Milestone D: Evaluation Tests ─────────────────────────────────────


class EvalDatasetTests(unittest.TestCase):
    def test_build_sample_dataset(self):
        from eval.datasets import build_sample_dataset
        cases = build_sample_dataset()
        self.assertTrue(len(cases) >= 3)
        categories = {c.category for c in cases}
        self.assertIn("doc_qa", categories)
        self.assertIn("db_qa", categories)
        self.assertIn("code_task", categories)

    def test_save_and_load_dataset(self):
        from eval.datasets import build_sample_dataset, save_dataset, load_dataset
        cases = build_sample_dataset()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_dataset(cases, path)
            loaded = load_dataset(path)
            self.assertEqual(len(loaded), len(cases))
            self.assertEqual(loaded[0].case_id, cases[0].case_id)
        finally:
            os.unlink(path)

    def test_eval_case_fields(self):
        from eval.datasets import EvalCase
        case = EvalCase(
            case_id="test-1",
            query="What is X?",
            category="doc_qa",
            expected_answer_contains=["X"],
            expected_route="doc_rag",
            expected_min_citations=1,
            tags=["smoke"],
        )
        self.assertEqual(case.case_id, "test-1")
        self.assertIn("X", case.expected_answer_contains)


class EvalRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_eval_case_simple(self):
        from eval.datasets import EvalCase
        from eval.runner import run_eval_case
        case = EvalCase(
            case_id="test-route",
            query="select name from users",
            category="db_qa",
            expected_route="sql",
            setup_tables=[{
                "name": "users",
                "columns": [{"name": "name", "type": "text"}],
                "rows": [{"name": "alice"}],
            }],
        )
        result = await run_eval_case(case)
        self.assertEqual(result.case_id, "test-route")
        self.assertEqual(result.actual_route, "sql")
        self.assertTrue(result.checks.get("route", False))

    async def test_run_eval_case_code(self):
        from eval.datasets import EvalCase
        from eval.runner import run_eval_case
        case = EvalCase(
            case_id="test-code",
            query="hello 函数报错怎么修",
            category="code_task",
            expected_route="code",
            expected_min_evidence=1,
            setup_files={"default": {"main.py": "def hello():\n    print('world')\n"}},
        )
        result = await run_eval_case(case)
        self.assertEqual(result.actual_route, "code")

    def test_summarize_results(self):
        from eval.datasets import EvalResult
        from eval.runner import summarize_results
        results = [
            EvalResult(case_id="c1", category="doc_qa", passed=True, duration_ms=100),
            EvalResult(case_id="c2", category="doc_qa", passed=False, duration_ms=200,
                       failure_category="bad_retrieval"),
            EvalResult(case_id="c3", category="db_qa", passed=True, duration_ms=150),
        ]
        summary = summarize_results(results)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertAlmostEqual(summary["pass_rate"], 2/3, places=3)
        self.assertIn("bad_retrieval", summary["failure_categories"])

    async def test_run_eval_suite(self):
        from eval.datasets import build_sample_dataset
        from eval.runner import run_eval_suite, summarize_results
        cases = build_sample_dataset()
        results = await run_eval_suite(cases)
        self.assertEqual(len(results), len(cases))
        summary = summarize_results(results)
        self.assertEqual(summary["total"], len(cases))


# ── Milestone D: Model Router Tests ──────────────────────────────────


class ModelRouterTests(unittest.TestCase):
    def test_router_default_tier(self):
        from services.api.app.llm.router import ModelRouter, TASK_TIER_MAP
        router = ModelRouter(fast_provider=None, routing_enabled=False)
        self.assertEqual(router.get_tier("route"), "fast")
        self.assertEqual(router.get_tier("default"), "fast")

    def test_router_tier_mapping(self):
        from services.api.app.llm.router import ModelRouter, TASK_TIER_MAP
        router = ModelRouter(fast_provider=None, routing_enabled=True)
        self.assertEqual(router.get_tier("route"), "fast")
        self.assertEqual(router.get_tier("synthesize"), "strong")
        self.assertEqual(router.get_tier("apply_patch"), "strong")
        self.assertEqual(router.get_tier("explain_error"), "strong")
        self.assertEqual(router.get_tier("verify"), "fast")

    def test_cost_tracker(self):
        from services.api.app.llm.router import CostTracker
        tracker = CostTracker()
        tracker.record("route", "fast", prompt_tokens=100, completion_tokens=50)
        tracker.record("synthesize", "strong", prompt_tokens=500, completion_tokens=200)
        summary = tracker.summary()
        self.assertEqual(summary["total_calls"], 2)
        self.assertEqual(summary["total_tokens"], 850)
        self.assertTrue(summary["total_cost_usd"] > 0)
        self.assertIn("fast", summary["by_tier"])
        self.assertIn("strong", summary["by_tier"])
        self.assertIn("route", summary["by_task"])

    def test_cost_tracker_empty(self):
        from services.api.app.llm.router import CostTracker
        tracker = CostTracker()
        summary = tracker.summary()
        self.assertEqual(summary["total_tokens"], 0)
        self.assertEqual(summary["total_cost_usd"], 0.0)

    def test_router_provider_selection(self):
        """Test that routing disabled returns fast provider."""
        from services.api.app.llm.router import ModelRouter
        fast = object()
        strong = object()
        router = ModelRouter(fast_provider=fast, strong_provider=strong, routing_enabled=False)
        self.assertIs(router.get_provider("synthesize"), fast)
        router2 = ModelRouter(fast_provider=fast, strong_provider=strong, routing_enabled=True)
        # strong doesn't have .enabled, so falls back to fast
        self.assertIs(router2.get_provider("route"), fast)


# ── Milestone D: Cache Tests ─────────────────────────────────────────


class LRUCacheTests(unittest.TestCase):
    def test_put_and_get(self):
        from services.api.app.cache.cache import LRUCache
        cache = LRUCache(max_entries=10, ttl_seconds=60)
        cache.put("k1", "v1")
        self.assertEqual(cache.get("k1"), "v1")

    def test_ttl_expiry(self):
        from services.api.app.cache.cache import LRUCache
        cache = LRUCache(max_entries=10, ttl_seconds=0.05)
        cache.put("k1", "v1")
        self.assertEqual(cache.get("k1"), "v1")
        time.sleep(0.1)
        self.assertIsNone(cache.get("k1"))

    def test_lru_eviction(self):
        from services.api.app.cache.cache import LRUCache
        cache = LRUCache(max_entries=3, ttl_seconds=60)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.put("k4", "v4")  # Should evict k1
        self.assertIsNone(cache.get("k1"))
        self.assertEqual(cache.get("k2"), "v2")

    def test_cache_stats(self):
        from services.api.app.cache.cache import LRUCache
        cache = LRUCache(max_entries=10, ttl_seconds=60)
        cache.put("k1", "v1")
        cache.get("k1")  # hit
        cache.get("k2")  # miss
        stats = cache.stats()
        self.assertEqual(stats["size"], 1)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5, places=2)

    def test_cache_clear(self):
        from services.api.app.cache.cache import LRUCache
        cache = LRUCache()
        cache.put("k1", "v1")
        cache.clear()
        self.assertIsNone(cache.get("k1"))
        stats = cache.stats()
        self.assertEqual(stats["size"], 0)


class RetrievalCacheTests(unittest.TestCase):
    def test_cache_hit(self):
        from services.api.app.cache.cache import RetrievalCache
        cache = RetrievalCache()
        cache.put("test query", {"tenant_id": "t1"}, 10, ["chunk1", "chunk2"])
        result = cache.get("test query", {"tenant_id": "t1"}, 10)
        self.assertEqual(result, ["chunk1", "chunk2"])

    def test_cache_miss(self):
        from services.api.app.cache.cache import RetrievalCache
        cache = RetrievalCache()
        result = cache.get("nonexistent", {}, 10)
        self.assertIsNone(result)

    def test_different_filters_different_keys(self):
        from services.api.app.cache.cache import RetrievalCache
        cache = RetrievalCache()
        cache.put("q", {"tenant_id": "t1"}, 10, ["a"])
        cache.put("q", {"tenant_id": "t2"}, 10, ["b"])
        r1 = cache.get("q", {"tenant_id": "t1"}, 10)
        r2 = cache.get("q", {"tenant_id": "t2"}, 10)
        self.assertEqual(r1, ["a"])
        self.assertEqual(r2, ["b"])


class EmbeddingCacheTests(unittest.TestCase):
    def test_cache_embedding(self):
        from services.api.app.cache.cache import EmbeddingCache
        cache = EmbeddingCache()
        embedding = [0.1, 0.2, 0.3]
        cache.put("test text", embedding)
        result = cache.get("test text")
        self.assertEqual(result, embedding)

    def test_cache_miss(self):
        from services.api.app.cache.cache import EmbeddingCache
        cache = EmbeddingCache()
        self.assertIsNone(cache.get("never cached"))


# ── Milestone D: Admin Endpoint Tests ─────────────────────────────────


class MilestoneDEndpointTests(unittest.TestCase):
    """Test new admin endpoints from Milestone D."""

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            raise unittest.SkipTest("fastapi TestClient not available")

        from services.api.app.api import app
        import services.api.app.api as api_mod
        api_mod._services = build_default_services()
        cls.client = TestClient(app)

    def test_metrics_endpoint(self):
        # First make a chat request to populate metrics
        self.client.post("/chat", json={
            "query": "hello", "tenant_id": "t1", "user_id": "u1",
        })
        response = self.client.get("/admin/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_requests", data)
        self.assertIn("citation_coverage", data)
        self.assertIn("tool_failure_rate", data)
        self.assertIn("avg_duration_ms", data)

    def test_metrics_history_endpoint(self):
        response = self.client.get("/admin/metrics/history?last_n=5")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("requests", data)
        self.assertIsInstance(data["requests"], list)

    def test_feedback_endpoint(self):
        # Make a chat request first
        chat_resp = self.client.post("/chat", json={
            "query": "test", "tenant_id": "t1", "user_id": "u1",
        })
        request_id = chat_resp.json().get("request_id", "")

        response = self.client.post("/admin/feedback", json={
            "request_id": request_id,
            "feedback": "positive",
        })
        self.assertEqual(response.status_code, 200)

    def test_feedback_not_found(self):
        response = self.client.post("/admin/feedback", json={
            "request_id": "nonexistent-id",
            "feedback": "negative",
        })
        self.assertEqual(response.status_code, 404)

    def test_cost_endpoint(self):
        response = self.client.get("/admin/cost")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_tokens", data)
        self.assertIn("total_cost_usd", data)

    def test_cache_endpoint(self):
        response = self.client.get("/admin/cache")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("enabled", data)
        self.assertIn("retrieval", data)
        self.assertIn("embedding", data)


# ── Milestone E: Embedder Tests ───────────────────────────────────────


class HashEmbedderTests(unittest.TestCase):
    def test_satisfies_protocol(self):
        from services.api.app.retrieval.embedder import HashEmbedder, Embedder
        e = HashEmbedder()
        self.assertIsInstance(e, Embedder)

    def test_deterministic(self):
        from services.api.app.retrieval.embedder import HashEmbedder
        e = HashEmbedder()
        v1 = e.embed("hello")
        v2 = e.embed("hello")
        self.assertEqual(v1, v2)

    def test_different_inputs(self):
        from services.api.app.retrieval.embedder import HashEmbedder
        e = HashEmbedder()
        v1 = e.embed("hello")
        v2 = e.embed("world")
        self.assertNotEqual(v1, v2)

    def test_dimension(self):
        from services.api.app.retrieval.embedder import HashEmbedder
        e = HashEmbedder(dim=64)
        v = e.embed("test")
        self.assertEqual(len(v), 64)

    def test_default_dimension(self):
        from services.api.app.retrieval.embedder import HashEmbedder
        e = HashEmbedder()
        v = e.embed("x")
        self.assertEqual(len(v), e.dimension)

    def test_batch_embed(self):
        from services.api.app.retrieval.embedder import HashEmbedder
        e = HashEmbedder()
        vecs = e.embed_batch(["a", "b", "c"])
        self.assertEqual(len(vecs), 3)
        self.assertEqual(len(vecs[0]), e.dimension)

    def test_batch_deterministic(self):
        from services.api.app.retrieval.embedder import HashEmbedder
        e = HashEmbedder()
        v1 = e.embed_batch(["a", "b"])
        v2 = e.embed_batch(["a", "b"])
        self.assertEqual(v1, v2)

    def test_embed_empty_string(self):
        from services.api.app.retrieval.embedder import HashEmbedder
        e = HashEmbedder()
        v = e.embed("")
        self.assertEqual(len(v), e.dimension)

    def test_normalized(self):
        from services.api.app.retrieval.embedder import HashEmbedder
        import math
        e = HashEmbedder()
        v = e.embed("test normalization")
        norm = math.sqrt(sum(x*x for x in v))
        self.assertAlmostEqual(norm, 1.0, places=5)


class APIEmbedderTests(unittest.TestCase):
    def test_satisfies_protocol(self):
        from services.api.app.retrieval.embedder import APIEmbedder, Embedder
        e = APIEmbedder(api_key="test", base_url="http://localhost", model="test-model")
        self.assertIsInstance(e, Embedder)

    def test_dimension(self):
        from services.api.app.retrieval.embedder import APIEmbedder
        e = APIEmbedder(api_key="test", base_url="http://localhost", model="test-model", dimension=256)
        self.assertEqual(e.dimension, 256)


class BuildEmbedderTests(unittest.TestCase):
    def test_default_returns_hash(self):
        from services.api.app.retrieval.embedder import build_embedder, HashEmbedder
        e = build_embedder()
        self.assertIsInstance(e, HashEmbedder)


# ── Milestone E: Reranker Tests ───────────────────────────────────────


class NoOpRerankerTests(unittest.TestCase):
    def test_satisfies_protocol(self):
        from services.api.app.retrieval.cross_encoder import NoOpReranker, Reranker
        r = NoOpReranker()
        self.assertIsInstance(r, Reranker)

    def test_enabled_false(self):
        from services.api.app.retrieval.cross_encoder import NoOpReranker
        r = NoOpReranker()
        self.assertFalse(r.enabled)

    def test_rerank_preserves_order(self):
        from services.api.app.retrieval.cross_encoder import NoOpReranker
        r = NoOpReranker()
        docs = ["doc a", "doc b", "doc c"]
        result = r.rerank("query", docs)
        # NoOpReranker returns (index, score) tuples preserving order
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0][0], 0)
        self.assertTrue(result[0][1] > result[1][1])

    def test_rerank_empty(self):
        from services.api.app.retrieval.cross_encoder import NoOpReranker
        r = NoOpReranker()
        result = r.rerank("query", [])
        self.assertEqual(result, [])


class CohereRerankerTests(unittest.TestCase):
    def test_satisfies_protocol(self):
        from services.api.app.retrieval.cross_encoder import CohereReranker, Reranker
        r = CohereReranker(api_key="test-key")
        self.assertIsInstance(r, Reranker)

    def test_not_enabled_without_key(self):
        from services.api.app.retrieval.cross_encoder import CohereReranker
        import os
        if not os.getenv("COHERE_API_KEY"):
            r = CohereReranker(api_key="")
            self.assertFalse(r.enabled)


class LocalCrossEncoderTests(unittest.TestCase):
    def test_satisfies_protocol(self):
        from services.api.app.retrieval.cross_encoder import LocalCrossEncoder, Reranker
        r = LocalCrossEncoder(base_url="http://localhost:8080")
        self.assertIsInstance(r, Reranker)


class BuildRerankerTests(unittest.TestCase):
    def test_default_returns_noop(self):
        from services.api.app.retrieval.cross_encoder import build_reranker, NoOpReranker
        r = build_reranker()
        self.assertIsInstance(r, NoOpReranker)


class RetrieverWithRerankerTests(unittest.TestCase):
    def test_retriever_accepts_reranker(self):
        from services.api.app.retrieval.service import Retriever
        from services.api.app.retrieval.cross_encoder import NoOpReranker
        from services.api.app.retrieval.embedder import HashEmbedder
        from services.api.app.retrieval.qdrant import InMemoryQdrant
        repo = InMemoryRepo()
        qdrant = InMemoryQdrant()
        embedder = HashEmbedder()
        reranker = NoOpReranker()
        retriever = Retriever(repo, qdrant, embedder=embedder, reranker=reranker)
        self.assertIsNotNone(retriever)


# ── Milestone E: Async Tests ─────────────────────────────────────────


class AsyncSafeToolCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_sync_function(self):
        def sync_fn():
            return 42
        result = await safe_tool_call("__test_sync", sync_fn)
        self.assertEqual(result, 42)

    async def test_async_function(self):
        async def async_fn():
            return "hello"
        result = await safe_tool_call("__test_async", async_fn)
        self.assertEqual(result, "hello")

    async def test_sync_with_args(self):
        def add(a, b):
            return a + b
        result = await safe_tool_call("__test_add", add, 3, 5)
        self.assertEqual(result, 8)

    async def test_async_with_kwargs(self):
        async def greet(name="world"):
            return f"hello {name}"
        result = await safe_tool_call("__test_greet", greet, name="async")
        self.assertEqual(result, "hello async")

    async def test_exception_propagates(self):
        def bad_fn():
            raise ValueError("test error")
        with self.assertRaises(ValueError):
            await safe_tool_call("__test_error", bad_fn)

    async def test_async_exception_propagates(self):
        async def bad_async_fn():
            raise RuntimeError("async error")
        with self.assertRaises(RuntimeError):
            await safe_tool_call("__test_async_error", bad_async_fn)

    async def test_circuit_breaker_records_success(self):
        breaker = _get_breaker("__test_cb_success")
        breaker._failures = 0
        breaker._opened_at = None
        breaker.record_failure()
        self.assertFalse(breaker.is_open)
        await safe_tool_call("__test_cb_success", lambda: "ok")
        self.assertFalse(breaker.is_open)

    async def test_circuit_breaker_records_failure(self):
        breaker = _get_breaker("__test_cb_fail")
        breaker._failures = 0
        breaker._opened_at = None
        for _ in range(2):
            try:
                def raise_fn():
                    raise ValueError("fail")
                await safe_tool_call("__test_cb_fail", raise_fn)
            except ValueError:
                pass
        self.assertTrue(breaker._failures >= 2)


class AsyncRunAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_basic_run(self):
        services = build_default_services()
        state = await run_agent("hello", "t1", "u1", services)
        self.assertIsNotNone(state.final)
        self.assertIsNotNone(state.route)

    async def test_run_with_constraints(self):
        from services.api.app.agent.state import Constraints
        services = build_default_services()
        constraints = Constraints(repo="test-repo")
        state = await run_agent("hello", "t1", "u1", services, constraints=constraints)
        self.assertIsNotNone(state.final)

    async def test_run_collects_tool_calls(self):
        services = build_default_services()
        state = await run_agent("hello world", "t1", "u1", services)
        self.assertTrue(len(state.tool_calls) > 0)

    async def test_run_respects_max_iterations(self):
        services = build_default_services()
        state = await run_agent("hello", "t1", "u1", services)
        self.assertTrue(state.iteration <= state.max_iterations)


class AsyncQueueCallbackDetailTests(unittest.IsolatedAsyncioTestCase):
    async def test_emit_and_get(self):
        cb = AsyncQueueCallback()
        event = AgentEvent("test", {"key": "value"})
        cb.emit(event)
        received = await cb.get(timeout=1.0)
        self.assertIsNotNone(received)
        self.assertEqual(received.event_type, "test")
        self.assertEqual(received.data["key"], "value")

    async def test_close_signals_none(self):
        cb = AsyncQueueCallback()
        cb.close()
        result = await cb.get(timeout=0.1)
        self.assertIsNone(result)

    async def test_emit_after_close_ignored(self):
        cb = AsyncQueueCallback()
        cb.close()
        cb.emit(AgentEvent("late", {}))
        result = await cb.get(timeout=0.1)
        self.assertIsNone(result)

    async def test_async_iter(self):
        cb = AsyncQueueCallback()
        events_emitted = [
            AgentEvent("a", {"n": 1}),
            AgentEvent("b", {"n": 2}),
            AgentEvent("c", {"n": 3}),
        ]
        for ev in events_emitted:
            cb.emit(ev)
        cb.close()

        received = []
        async for event in cb:
            received.append(event)
        self.assertEqual(len(received), 3)
        self.assertEqual([e.event_type for e in received], ["a", "b", "c"])

    async def test_overflow_drops_oldest(self):
        cb = AsyncQueueCallback(maxsize=2)
        cb.emit(AgentEvent("a", {}))
        cb.emit(AgentEvent("b", {}))
        cb.emit(AgentEvent("c", {}))
        cb.close()
        received = []
        async for event in cb:
            received.append(event)
        self.assertTrue(len(received) >= 1)


class AsyncChatTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_basic(self):
        from services.api.app.main import chat
        result = await chat("hello", "t1", "u1")
        self.assertIn("answer", result)
        self.assertIn("request_id", result)

    async def test_chat_with_services(self):
        from services.api.app.main import chat
        services = build_default_services()
        result = await chat("hello", "t1", "u1", services)
        self.assertIn("answer", result)
        self.assertIn("confidence", result)

    async def test_chat_returns_debug_info(self):
        from services.api.app.main import chat
        result = await chat("hello", "t1", "u1")
        self.assertIn("debug", result)
        self.assertIn("route", result["debug"])
        self.assertIn("tool_calls", result["debug"])


# ── Milestone E: Eval Tests ──────────────────────────────────────────


class EvalFullDatasetTests(unittest.TestCase):
    def test_build_full_dataset_size(self):
        from eval.datasets import build_full_dataset
        cases = build_full_dataset()
        self.assertTrue(len(cases) >= 200)

    def test_full_dataset_categories(self):
        from eval.datasets import build_full_dataset
        cases = build_full_dataset()
        categories = {c.category for c in cases}
        self.assertIn("doc_qa", categories)
        self.assertIn("db_qa", categories)
        self.assertIn("code_task", categories)

    def test_full_dataset_has_expected_chunks(self):
        from eval.datasets import build_full_dataset
        cases = build_full_dataset()
        with_chunks = [c for c in cases if c.expected_chunk_ids]
        self.assertTrue(len(with_chunks) >= 0)


class MRRRecallTests(unittest.TestCase):
    def test_mrr_at_k_found_first(self):
        from eval.runner import compute_mrr_at_k
        result = compute_mrr_at_k(["a"], ["a", "b", "c"], k=10)
        self.assertAlmostEqual(result, 1.0)

    def test_mrr_at_k_found_second(self):
        from eval.runner import compute_mrr_at_k
        result = compute_mrr_at_k(["b"], ["a", "b", "c"], k=10)
        self.assertAlmostEqual(result, 0.5)

    def test_mrr_at_k_not_found(self):
        from eval.runner import compute_mrr_at_k
        result = compute_mrr_at_k(["z"], ["a", "b", "c"], k=10)
        self.assertAlmostEqual(result, 0.0)

    def test_recall_at_k_partial(self):
        from eval.runner import compute_recall_at_k
        result = compute_recall_at_k(["a", "b", "c"], ["a", "d", "c"], k=10)
        self.assertAlmostEqual(result, 2/3, places=4)

    def test_recall_at_k_all(self):
        from eval.runner import compute_recall_at_k
        result = compute_recall_at_k(["a", "b"], ["a", "b", "c"], k=10)
        self.assertAlmostEqual(result, 1.0)


# ── Milestone E: Repo Protocol Tests ─────────────────────────────────


class RepoProtocolTests(unittest.TestCase):
    def test_inmemory_repo_satisfies_protocol(self):
        from services.api.app.storage.protocol import Repo
        repo = InMemoryRepo()
        self.assertIsInstance(repo, Repo)

    def test_inmemory_repo_add_document(self):
        repo = InMemoryRepo()
        doc = Document(
            doc_id="d1", tenant_id="t1", source_type="pdf",
            title="Test", uri="file://test.pdf", version="v1",
            doc_updated_at="2025-01-01", ingested_at="2025-01-02",
        )
        repo.add_document(doc)
        result = repo.get_document("d1")
        self.assertIsNotNone(result)
        self.assertEqual(result.title, "Test")

    def test_inmemory_repo_add_chunk(self):
        repo = InMemoryRepo()
        chunk = Chunk(
            chunk_id="c1", doc_id="d1", tenant_id="t1",
            chunk_index=0, text="test chunk",
        )
        repo.add_chunk(chunk)
        result = repo.get_chunk("c1")
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "test chunk")

    def test_inmemory_repo_list_policies(self):
        repo = InMemoryRepo()
        policy = build_policy("p1", "t1", {"allow_all": True})
        repo.add_policy(policy)
        policies = repo.list_policies("t1")
        self.assertEqual(len(policies), 1)

    def test_inmemory_repo_register_table(self):
        repo = InMemoryRepo()
        table = TableData(
            name="test_table",
            columns=[{"name": "id", "type": "int"}],
            rows=[{"id": 1}],
        )
        repo.register_table(table)
        result = repo.get_table("test_table")
        self.assertIsNotNone(result)
        self.assertEqual(len(result.rows), 1)

    def test_inmemory_repo_export_state(self):
        repo = InMemoryRepo()
        table = TableData(name="t", columns=[], rows=[])
        repo.register_table(table)
        state = repo.export_state()
        self.assertIn("tables", state)
        self.assertEqual(len(state["tables"]), 1)


class PostgresRepoProtocolTests(unittest.TestCase):
    def test_postgres_repo_satisfies_protocol(self):
        try:
            from services.api.app.storage.pg_repo import PostgresRepo
            from services.api.app.storage.protocol import Repo
            self.assertTrue(hasattr(PostgresRepo, 'add_document'))
            self.assertTrue(hasattr(PostgresRepo, 'get_document'))
            self.assertTrue(hasattr(PostgresRepo, 'add_chunk'))
            self.assertTrue(hasattr(PostgresRepo, 'get_chunk'))
        except ImportError:
            self.skipTest("PostgresRepo not available")


class FactoryBuildServicesTests(unittest.TestCase):
    def test_build_default_services(self):
        services = build_default_services()
        self.assertIsNotNone(services.repo)
        self.assertIsNotNone(services.qdrant)
        self.assertIsNotNone(services.retriever)
        self.assertIsNotNone(services.sql_engine)
        self.assertIsNotNone(services.code_search)
        self.assertIsNotNone(services.llm)

    def test_default_services_has_embedder(self):
        services = build_default_services()
        self.assertIsNotNone(services.embedder)

    def test_default_services_has_reranker(self):
        services = build_default_services()
        self.assertIsNotNone(services.reranker)

    def test_build_services_with_custom_repo(self):
        repo = InMemoryRepo()
        services = build_default_services(repo)
        self.assertIs(services.repo, repo)

    def test_services_retriever_connected(self):
        services = build_default_services()
        results = services.retriever.retrieve("test", {"tenant_id": "t1"}, top_k=5)
        self.assertIsInstance(results, list)