import unittest

from services.api.app.agent.graph import build_default_services, run_agent
from services.api.app.agent.nodes.code import CodeSearch, code_node
from services.api.app.agent.nodes.finalize import finalize_node
from services.api.app.agent.nodes.route import route_node
from services.api.app.agent.nodes.synthesize import synthesize_node
from services.api.app.agent.nodes.verify import verify_node
from services.api.app.agent.nodes.web import web_node
from services.api.app.agent.session import InMemorySessionStore, SessionTurn
from services.api.app.agent.state import build_initial_state
from services.api.app.auth.acl import build_policy
from services.api.app.retrieval.rerank import rrf_fuse
from services.api.app.storage.models import Chunk, Document, TableData
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


if __name__ == "__main__":
    unittest.main()
