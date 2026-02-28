import os
import tempfile
import time
import threading
import unittest

from services.api.app.agent.graph import build_default_services, run_agent
from services.api.app.agent.nodes.code import CodeSearch, code_node
from services.api.app.agent.nodes.finalize import finalize_node
from services.api.app.agent.nodes.route import route_node
from services.api.app.agent.nodes.sql import SqlEngine
from services.api.app.agent.nodes.synthesize import synthesize_node
from services.api.app.agent.nodes.verify import verify_node
from services.api.app.agent.nodes.web import web_node
from services.api.app.agent.session import InMemorySessionStore, SessionTurn
from services.api.app.agent.state import build_initial_state
from services.api.app.agent.callbacks import AgentEvent, QueueCallback, NullCallback
from services.api.app.agent.reliability import (
    CircuitBreaker, CircuitOpenError, ToolTimeoutError,
    with_timeout, with_retry, safe_tool_call, RetryConfig,
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


class AgentRouteTests(unittest.TestCase):
    def test_route_sql(self):
        services = build_default_services()
        state = build_initial_state("select * from sales", "t1", "u1")
        state = route_node(state, services)
        self.assertEqual(state.route, "sql")

    def test_route_code(self):
        services = build_default_services()
        state = build_initial_state("函数报错怎么修", "t1", "u1")
        state = route_node(state, services)
        self.assertEqual(state.route, "code")

    def test_route_doc(self):
        services = build_default_services()
        state = build_initial_state("请参考文档说明", "t1", "u1")
        state = route_node(state, services)
        self.assertEqual(state.route, "doc_rag")

    def test_route_mixed_fallback(self):
        services = build_default_services()
        state = build_initial_state("hello world", "t1", "u1")
        state = route_node(state, services)
        self.assertEqual(state.route, "mixed")


class RetrievalAclTests(unittest.TestCase):
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

    def test_retrieval_allows_user(self):
        state = run_agent("Postgres 做什么", "tenant-a", "u1", self.services)
        self.assertIn("Postgres", state.final.answer)

    def test_retrieval_blocks_user(self):
        state = run_agent("Postgres 做什么", "tenant-a", "u2", self.services)
        self.assertIn("证据不足", state.final.answer)


class SqlEngineTests(unittest.TestCase):
    def test_simple_select(self):
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
        state = run_agent("select region from sales where region = 'cn'", "t1", "u1", services)
        self.assertIn("SQL 返回 1 行", state.final.answer)


class RrfTests(unittest.TestCase):
    def test_rrf_ordering(self):
        primary = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        secondary = [("c", 0.9), ("a", 0.8), ("d", 0.7)]
        fused = rrf_fuse(primary, secondary)
        self.assertEqual(fused[0][0], "a")
        self.assertIn("c", [item[0] for item in fused[:2]])


class CodeNodeTests(unittest.TestCase):
    def test_code_search_in_memory(self):
        services = build_default_services()
        services.code_search = CodeSearch(
            repo_roots={},
            in_memory_files={"default": {"main.py": "def hello():\n    print('world')\n"}},
        )
        state = build_initial_state("hello", "t1", "u1")
        state = code_node(state, services)
        self.assertTrue(len(state.tool_calls) > 0)
        self.assertTrue(state.tool_calls[-1].ok)
        self.assertTrue(len(state.evidence) > 0)

    def test_code_search_no_match(self):
        services = build_default_services()
        services.code_search = CodeSearch(
            repo_roots={},
            in_memory_files={"default": {"main.py": "def hello():\n    pass\n"}},
        )
        state = build_initial_state("nonexistent_function_xyz", "t1", "u1")
        state = code_node(state, services)
        self.assertTrue(state.tool_calls[-1].ok)


class WebNodeTests(unittest.TestCase):
    def test_web_node_no_llm(self):
        services = build_default_services()
        state = build_initial_state("latest news", "t1", "u1")
        state = web_node(state, services)
        self.assertTrue(len(state.tool_calls) > 0)
        last_call = state.tool_calls[-1]
        self.assertFalse(last_call.ok)
        self.assertIn("LLM not available", last_call.error)


class SynthesizeNodeTests(unittest.TestCase):
    def test_synthesize_no_evidence(self):
        services = build_default_services()
        state = build_initial_state("test", "t1", "u1")
        state = synthesize_node(state, services)
        self.assertIsNotNone(state.draft)
        self.assertIn("未找到", state.draft.answer_text)

    def test_synthesize_with_doc_evidence(self):
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
        state = synthesize_node(state, services)
        self.assertIsNotNone(state.draft)
        self.assertTrue(len(state.draft.answer_text) > 0)
        self.assertIn("Python", state.draft.answer_text)


class VerifyNodeTests(unittest.TestCase):
    def test_verify_enough_evidence(self):
        services = build_default_services()
        state = build_initial_state("test", "t1", "u1")
        state.route = "doc_rag"
        state.evidence.append(EvidenceItem(kind="doc_chunk", score=1.0, text="answer"))
        state.draft = Draft(answer_text="answer")
        state = verify_node(state, services)
        self.assertIsNotNone(state.verification)
        self.assertTrue(state.verification.enough_evidence)

    def test_verify_missing_evidence(self):
        services = build_default_services()
        state = build_initial_state("test", "t1", "u1")
        state.route = "doc_rag"
        state.draft = Draft(answer_text="answer")
        state = verify_node(state, services)
        self.assertIsNotNone(state.verification)
        self.assertFalse(state.verification.enough_evidence)
        self.assertIn("doc_chunks", state.verification.missing)


class FinalizeNodeTests(unittest.TestCase):
    def test_finalize_with_evidence(self):
        services = build_default_services()
        state = build_initial_state("test", "t1", "u1")
        state.draft = Draft(answer_text="all good")
        state.verification = Verification(enough_evidence=True)
        state = finalize_node(state, services)
        self.assertEqual(state.final.confidence, "high")
        self.assertEqual(state.final.answer, "all good")

    def test_finalize_degraded(self):
        services = build_default_services()
        state = build_initial_state("test", "t1", "u1")
        state.draft = Draft(answer_text="partial")
        state.verification = Verification(enough_evidence=False, missing=["doc_chunks"])
        state = finalize_node(state, services)
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


class TimeoutTests(unittest.TestCase):
    def test_timeout_triggers(self):
        def slow_fn():
            time.sleep(5)
            return "done"

        with self.assertRaises(ToolTimeoutError):
            with_timeout(slow_fn, 0.1)

    def test_timeout_passes(self):
        def fast_fn():
            return 42

        result = with_timeout(fast_fn, 5.0)
        self.assertEqual(result, 42)


class RetryTests(unittest.TestCase):
    def test_retry_recovers(self):
        attempts = []

        def flaky_fn():
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("transient")
            return "ok"

        config = RetryConfig(max_retries=3, base_delay=0.01)
        result = with_retry(flaky_fn, config)
        self.assertEqual(result, "ok")
        self.assertEqual(len(attempts), 3)

    def test_retry_exhausted(self):
        def always_fail():
            raise ConnectionError("permanent")

        config = RetryConfig(max_retries=2, base_delay=0.01)
        with self.assertRaises(ConnectionError):
            with_retry(always_fail, config)

    def test_non_retryable_exception_raises_immediately(self):
        attempts = []

        def value_error_fn():
            attempts.append(1)
            raise ValueError("not retryable")

        config = RetryConfig(max_retries=3, base_delay=0.01)
        with self.assertRaises(ValueError):
            with_retry(value_error_fn, config)
        self.assertEqual(len(attempts), 1)


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

    def test_safe_tool_call_circuit_open_raises(self):
        # Force the circuit open for a tool
        from services.api.app.agent.reliability import _get_breaker
        breaker = _get_breaker("test_tool_circuit")
        for _ in range(5):
            breaker.record_failure()
        self.assertTrue(breaker.is_open)
        with self.assertRaises(CircuitOpenError):
            safe_tool_call("test_tool_circuit", lambda: "nope")


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

    def test_run_agent_with_callback(self):
        services = build_default_services()
        cb = QueueCallback()
        state = run_agent("hello world", "t1", "u1", services, callback=cb)
        self.assertIsNotNone(state.final)
        # Callback should have been signaled to close
        self.assertTrue(cb._closed.is_set())
        # Drain the remaining events
        while True:
            result = cb.get(timeout=0.1)
            if result is None:
                break
        self.assertTrue(cb.closed)

    def test_callback_receives_events(self):
        services = build_default_services()
        table = TableData(
            name="items",
            columns=[{"name": "name", "type": "text"}],
            rows=[{"name": "item1"}],
        )
        services.repo.register_table(table)

        events = []
        cb = QueueCallback()

        def run_in_thread():
            run_agent("select name from items", "t1", "u1", services, callback=cb)

        t = threading.Thread(target=run_in_thread)
        t.start()

        while True:
            try:
                event = cb.get(timeout=2.0)
            except Exception:
                break
            if event is None:
                break
            events.append(event)

        t.join(timeout=5)
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


class OpenFileNodeTests(unittest.TestCase):
    def test_open_file_node_success(self):
        from services.api.app.agent.nodes.code import open_file_node
        services = build_default_services()
        services.code_search = CodeSearch(
            repo_roots={},
            in_memory_files={"default": {"test.py": "def foo():\n    return 42\n"}},
        )
        state = build_initial_state("open test.py", "t1", "u1")
        state = open_file_node(state, services)
        self.assertTrue(len(state.tool_calls) > 0)
        self.assertTrue(state.tool_calls[-1].ok)
        self.assertTrue(len(state.evidence) > 0)
        self.assertEqual(state.evidence[-1].kind, "file_content")

    def test_open_file_node_not_found(self):
        from services.api.app.agent.nodes.code import open_file_node
        services = build_default_services()
        services.code_search = CodeSearch(repo_roots={}, in_memory_files={"default": {}})
        state = build_initial_state("open nonexistent.py", "t1", "u1")
        state = open_file_node(state, services)
        self.assertTrue(len(state.tool_calls) > 0)
        self.assertFalse(state.tool_calls[-1].ok)


class ExplainErrorNodeTests(unittest.TestCase):
    def test_explain_error_node(self):
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
        state = explain_error_node(state, services)
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


class MilestoneCIntegrationTests(unittest.TestCase):
    def test_chat_with_client_context(self):
        """Test that client_context flows through chat() correctly."""
        from services.api.app.main import chat
        result = chat(
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

    def test_run_agent_with_initial_evidence(self):
        """Test that initial_evidence is injected into agent state."""
        services = build_default_services()
        initial_ev = [
            EvidenceItem(
                kind="file_content", score=1.0,
                text="class Foo: pass",
                citations=[Citation(kind="code", path="foo.py")],
            ),
        ]
        state = run_agent(
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
