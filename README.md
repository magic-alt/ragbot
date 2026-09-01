# Ragbot

Ragbot 是一个面向本地/企业知识库的 **Agentic RAG knowledge service**：把 PDF、网页、Git 仓库和本地文本目录摄取为可检索知识，通过 Qdrant + PostgreSQL 混合检索为自身 Agent 或其他 Agent/应用提供底层知识支撑。

> 当前 `main` 已包含 Milestone E（真实 Embedder、Reranker、异步 Agent 路径、评测与 PostgresRepo）。项目包版本仍为 `0.5.0`；版本号升级应在正式 release 中单独处理。

## 能力范围

- **知识摄取**：PDF / Web / Git / local filesystem → chunk → dedup → embedding → Qdrant
- **混合检索**：Qdrant vector search + PostgreSQL FTS + RRF，可选 cross-encoder rerank
- **Agentic RAG**：route → retrieve/sql/code/web → synthesize → verify → finalize
- **多租户与 ACL**：tenant/user scope、ACL hash 前置过滤、API Key
- **LLM**：OpenAI-compatible provider 与 Ollama adapter
- **服务接口**：`/search`、`/chat`、OpenAI-compatible `/v1/chat/completions`、Source/Ingest API
- **工程能力**：SSE、可靠性封装、评测、指标/Tracing、Docker Compose、Helm

## 架构

```text
Other Agents / CLI / SDK / IDE
            │ REST / SSE / OpenAI-compatible API
            ▼
┌──────────────────────── Ragbot API ────────────────────────┐
│ Source API → Ingestion Pipeline                            │
│   local/PDF/Web/Git → chunk → dedup → shared Embedder     │
│                                      │                     │
│                                      ▼                     │
│                                 Qdrant vectors             │
│                                                           │
│ Query → route → retrieve ──┬─ Qdrant vector search        │
│                            ├─ PostgreSQL FTS               │
│                            └─ RRF / optional rerank        │
│          → synthesize → verify → final + citations         │
└───────────────────────┬───────────────────────────────────┘
                        │
               PostgreSQL metadata / ACL / jobs
```

一个关键不变量是：**摄取与查询必须使用同一个 embedding 模型和相同向量维度**。如果修改 `EMBEDDING_MODEL` 或 `QDRANT_DIM`，应创建兼容 collection 并重新索引，而不是继续查询旧向量。

## 快速开始

### 1. Python 本地开发

要求 Python 3.10+。

```bash
git clone https://github.com/magic-alt/ragbot.git
cd ragbot
python -m venv .venv

# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev,postgres,qdrant,worker,observability]"
cp .env.example .env   # Windows 可手动复制
python -m pytest -q
uvicorn services.api.app.api:app --reload --host 0.0.0.0 --port 8000
```

`requirements.txt` 是 Docker/完整运行时依赖集合；开发时优先使用 editable install + extras，以便同时验证包元数据。

### 2. Docker Compose

```bash
cp .env.example .env
mkdir -p data
# 编辑 .env，至少配置真实语义 RAG 所需的 OPENAI_API_KEY；
# EMBEDDING_API_KEY 为空时会回退到 OPENAI_API_KEY。
docker compose up -d
```

默认启动 API + PostgreSQL + Qdrant。附加服务：

```bash
docker compose --profile ollama up -d
docker compose --profile observability up -d
```

宿主机 `./data` 默认只读挂载到容器 `/data`，可用 `RAGBOT_DATA_DIR` 指向其他本地知识目录。

### 3. 建立本地知识库

先创建 Source：

```bash
curl -X POST http://localhost:8000/sources \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo",
    "source_type": "local_fs",
    "name": "local-docs",
    "config": {"path": "/data"},
    "tags": ["local"]
  }'
```

从响应复制 `source_id`，触发摄取：

```bash
curl -X POST http://localhost:8000/ingest/jobs \
  -H "Content-Type: application/json" \
  -d '{"source_id":"<SOURCE_ID>","tenant_id":"demo"}'
```

检索：

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query":"这个知识库的主要内容是什么？",
    "tenant_id":"demo",
    "user_id":"local-user",
    "top_k":5
  }'
```

如果配置了 `RAGBOT_API_KEYS`，以上请求同时传入 `X-API-Key`。

## Embedding 配置

推荐生产环境显式设置：

```bash
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=                 # 空时使用 OPENAI_API_KEY
EMBEDDING_BASE_URL=https://api.openai.com
QDRANT_DIM=1536
QDRANT_COLLECTION=rag_chunks
```

未配置可用的 embedding API 时，Ragbot 会使用**确定性的 HashEmbedder** 作为开发/测试回退。它可以验证完整管线，但不等价于语义 embedding，不建议作为生产知识库质量方案。

完整配置说明、Docker 与生产检查项见 [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)。

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/chat` | POST | Agentic RAG，支持 SSE |
| `/search` | POST | 纯检索，适合其他 Agent 作为 knowledge tool 调用 |
| `/v1/chat/completions` | POST | OpenAI-compatible adapter |
| `/sources` | CRUD | Source 管理 |
| `/ingest/jobs` | POST/GET | 摄取任务触发与状态 |
| `/admin/health` | GET | 健康检查 |
| `/admin/metrics` | GET | 质量/运行指标 |
| `/admin/cache` | GET | 缓存统计 |

共享数据契约位于 `contracts/`；架构演进与尚未完成项见 [`ROADMAP.md`](ROADMAP.md)。

## 测试与 CI

```bash
python -m pip check
python -m compileall -q contracts services cli eval
python -m pytest -q
```

GitHub Actions 对 Python 3.10 与 3.12 执行上述检查。Milestone E 的基线提交记录为 228 tests passing；本仓库应以当前 CI 结果而不是 README 中的固定测试数字作为合并依据。

## 仓库状态与 License

仓库当前未包含开源许可证文件。公开可见不等于自动授予开源使用许可；在正式选择并加入 LICENSE 前，应按“未授予额外许可”理解。
