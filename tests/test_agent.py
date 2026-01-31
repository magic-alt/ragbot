import unittest

from services.api.app.agent.graph import build_default_services, run_agent
from services.api.app.agent.nodes.route import route_node
from services.api.app.agent.state import build_initial_state
from services.api.app.auth.acl import build_policy
from services.api.app.retrieval.rerank import rrf_fuse
from services.api.app.storage.models import Chunk, Document, TableData
from services.worker.jobs.embed_and_upsert import embed_and_upsert


class AgentRouteTests(unittest.TestCase):
    def test_route_sql(self):
        state = build_initial_state("select * from sales", "t1", "u1")
        state = route_node(state)
        self.assertEqual(state.route, "sql")

    def test_route_code(self):
        state = build_initial_state("函数报错怎么修", "t1", "u1")
        state = route_node(state)
        self.assertEqual(state.route, "code")


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
        self.assertIn("Postgres", state.final)

    def test_retrieval_blocks_user(self):
        state = run_agent("Postgres 做什么", "tenant-a", "u2", self.services)
        self.assertIn("证据不足", state.final)


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
        self.assertIn("SQL 返回 1 行", state.final)


class RrfTests(unittest.TestCase):
    def test_rrf_ordering(self):
        primary = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        secondary = [("c", 0.9), ("a", 0.8), ("d", 0.7)]
        fused = rrf_fuse(primary, secondary)
        self.assertEqual(fused[0][0], "a")
        self.assertIn("c", [item[0] for item in fused[:2]])


if __name__ == "__main__":
    unittest.main()
