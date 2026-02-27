from __future__ import annotations

from services.api.app.agent.graph import build_default_services
from services.api.app.auth.acl import build_policy
from services.api.app.main import chat
from services.api.app.storage.models import Chunk, Document, TableData
from services.worker.jobs.embed_and_upsert import embed_and_upsert


def main() -> None:
    services = build_default_services()
    repo = services.repo
    qdrant = services.qdrant

    policy = build_policy("p1", "tenant-a", {"allow_users": ["u1"]})
    repo.add_policy(policy)

    doc = Document(
        doc_id="doc-1",
        tenant_id="tenant-a",
        source_type="pdf",
        title="Demo",
        uri="file://OpenVLA_AnOpen-Source Vision-Language-Action_Model.pdf",
        version="v1",
        doc_updated_at="2025-01-01",
        ingested_at="2025-01-02",
        tags=["demo"],
        acl_policy_id=policy.acl_policy_id,
    )
    repo.add_document(doc)

    chunk = Chunk(
        chunk_id="chunk-1",
        doc_id=doc.doc_id,
        tenant_id=doc.tenant_id,
        chunk_index=0,
        text="Postgres 负责结构化数据查询与事务处理，Qdrant 用于向量检索。",
        metadata={
            "source_type": "pdf",
            "ingested_at": doc.ingested_at,
            "doc_updated_at": doc.doc_updated_at,
            "version": doc.version,
            "acl_hash": policy.policy_hash,
            "tags": doc.tags,
        },
    )
    embed_and_upsert(repo, qdrant, [chunk])

    table = TableData(
        name="sales",
        columns=[{"name": "region", "type": "text"}, {"name": "amount", "type": "int"}],
        rows=[
            {"region": "cn", "amount": 10},
            {"region": "us", "amount": 20},
        ],
    )
    repo.register_table(table)

    print("Doc RAG:")
    print(chat("Postgres 负责什么", "tenant-a", "u1", services))

    print("\nSQL:")
    print(chat("select region from sales where region = 'cn'", "tenant-a", "u1", services))

    print("\nCode:")
    print(chat("class SqlEngine", "tenant-a", "u1", services))


if __name__ == "__main__":
    main()
