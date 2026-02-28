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
