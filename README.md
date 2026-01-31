# ragbot

这是基于 `PROJECT.md` 的 **最小可运行 Agentic RAG** 实现，重点覆盖：

- Postgres/Qdrant 的接口与过滤字段设计（以 in-memory 适配器实现）
- Agent 状态机（route → retrieve/sql/code → synthesize → verify → finalize）
- 跨语言工具契约（`contracts/tools.schema.json`）
- 单元测试覆盖路由、检索、SQL、安全过滤、融合排序

> 说明：本仓库实现的是 **可运行的参考骨架**。默认使用内存适配器，不依赖真实 Postgres/Qdrant。

## 目录结构

- `contracts/`：OpenAPI + 工具 Schema + TS/Python 类型
- `services/api/app/`：核心 Agent 与检索逻辑
- `services/worker/`：摄取/嵌入/去重（最小实现）
- `packages/node-client/`：Node 客户端示例
- `tests/`：单元测试

## 快速开始

运行单元测试：

```bash
python -m unittest discover -s tests -p "test_*.py"
```

运行示例聊天（Python）：

```python
from services.api.app.main import chat
from services.api.app.agent.graph import build_default_services
from services.api.app.storage.models import Document, Chunk
from services.api.app.auth.acl import build_policy
from services.worker.jobs.embed_and_upsert import embed_and_upsert

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
    uri="file://demo.pdf",
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
    text="这是一个示例文档片段，介绍 Postgres 和 Qdrant。",
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

response = chat("请说明 Postgres 在系统中的作用", "tenant-a", "u1", services)
print(response)
```

## 设计说明（与 PROJECT.md 对齐）

- **检索融合**：`Qdrant TopK` + `FTS TopK` → `RRF` 融合 → TopK 输出
- **安全过滤**：检索前按 `tenant_id + acl_hash` 过滤
- **工具约束**：SQL 仅支持简单 `SELECT`，内置限制行数
- **引用强制**：每条证据生成 `cite` 字段，最终回答附带引用

## 后续扩展建议

- 替换 `InMemoryQdrant` 为真实 Qdrant SDK
- 替换 `SqlEngine` 为真实 Postgres 只读连接
- 增强 `web_node` 外部检索
- 增加 Rerank Cross-Encoder 模块

