# Ragbot

Ragbot 是一个面向本地与企业知识库的 **Agentic RAG product/service**：把 PDF、网页、Git 仓库和本地文档目录快速构建为可检索知识库，通过 Qdrant + PostgreSQL 混合检索，为自身 Agent、其他 Agent、IDE 或业务应用提供知识底座。

> 当前 package / FastAPI / Helm metadata 仍为 `0.5.0`。代码已进入 v1.0 release-gate 阶段，但只有真实 provider staging、生产运维 gate 与 exact release commit 全部通过后才会发布 `1.0.0`。见 [`docs/V1_RELEASE_READINESS.md`](docs/V1_RELEASE_READINESS.md)。

## 你可以用 Ragbot 做什么

- **快速建库**：一个 `rag ingest ... --wait` 命令即可创建/复用 Source、提交任务并等待知识可用；也可用 manifest 批量导入。
- **多源摄取**：PDF / Web / Git / local filesystem → chunk → dedup → embedding → PostgreSQL + Qdrant。
- **可恢复摄取**：PostgreSQL durable queue、worker claim、lease、heartbeat、crash recovery、bounded retry。
- **幂等与重复部署友好**：同 tenant/type/location 默认复用稳定 Source；pending/running Job 默认去重；支持显式 idempotency key。
- **混合检索**：Qdrant vector search + PostgreSQL FTS/CJK bigram + RRF，可选 cross-encoder rerank。
- **Agentic RAG**：route → retrieve/sql/code/web → synthesize → verify → finalize，输出 citation。
- **多租户与 ACL**：API-key principal 绑定 tenant/user/groups/roles，检索前置 ACL scope。
- **兼容现有 AI 工具链**：`/search`、`/chat`、OpenAI-compatible `/v1/chat/completions`、CLI、Node SDK。
- **生产部署**：Docker Compose、Helm、health/readiness、metrics/tracing、真实 provider staging workflow。

## 架构

```text
CLI / SDK / Other Agents / IDE / Applications
                    │
          REST / SSE / OpenAI-compatible API
                    ▼
┌────────────────────────── Ragbot API ──────────────────────────┐
│ API Key → trusted principal → tenant/user/groups/roles        │
│                                                               │
│ Quick Import ─┬─ stable Source upsert                         │
│               ├─ active-job dedupe / idempotency             │
│               └─ PostgreSQL durable ingestion queue           │
│                                                               │
│ Query → retrieve ─┬─ Qdrant vector search                     │
│                   ├─ PostgreSQL FTS / CJK bigrams             │
│                   └─ RRF / optional rerank                    │
│       → agent synthesize → verify → answer + citations        │
└─────────────────────────────┬─────────────────────────────────┘
                              │
                       PostgreSQL / Qdrant
                              ▲
                              │ claim + lease + heartbeat
┌──────────────────── Ragbot Ingestion Worker ──────────────────┐
│ PDF/Web/Git/local → chunk → dedup → shared Embedder           │
│                                  ├─ PostgreSQL metadata/FTS    │
│                                  └─ Qdrant vectors             │
└───────────────────────────────────────────────────────────────┘
```

关键不变量：

1. **摄取和查询使用同一 embedding contract。** 模型或向量维度变化必须重新索引；Ragbot 不会静默 truncate/zero-pad。
2. **生产身份不能由请求自行扩大。** API key principal 决定可信 tenant/user/groups/roles。
3. **生产摄取不依赖 API 进程生命周期。** production 禁止 inline ingestion，任务必须由 durable worker 执行。

## 60 秒建立 RAG 数据库

完整教程见 [`docs/QUICKSTART.md`](docs/QUICKSTART.md)。

### 1. 启动

```bash
cp .env.example .env
mkdir -p data
# 将本地资料放到 ./data；在 .env 中配置 LLM / embedding provider
docker compose up -d --build
```

默认 Compose 启动 API、ingestion worker、migration、PostgreSQL 和 Qdrant。

### 2. 检查部署

```bash
python -m pip install -e ".[postgres,qdrant,worker]"
rag --server http://localhost:8000 doctor
```

预期：

```text
ragbot doctor: READY
```

### 3. 一条命令建库

本地目录：

```bash
rag --server http://localhost:8000 \
  --tenant engineering \
  ingest /data/manuals \
  --name "Engineering manuals" \
  --tag manuals \
  --wait
```

远程 PDF：

```bash
rag --server http://localhost:8000 \
  --tenant engineering \
  ingest https://example.com/product/guide.pdf \
  --wait
```

Git 仓库：

```bash
rag --server http://localhost:8000 \
  --tenant engineering \
  ingest https://github.com/magic-alt/ragbot \
  --ref main \
  --wait
```

默认会自动判断 `local_fs` / `pdf` / `repo` / `web`。`--wait` 会持续读取 durable Job，直到 `completed` 或 `failed`，并输出最终 docs/chunks 数量。

### 4. 用 manifest 批量建立知识库

仓库提供 [`examples/ragbot-manifest.json`](examples/ragbot-manifest.json)：

```bash
rag --server http://localhost:8000 \
  import examples/ragbot-manifest.json \
  --wait
```

manifest 可以同时声明本地目录、PDF、Git 和 Web Source。HTTP batch API 单次最多接收 100 个 Source。

### 5. 查询

```bash
rag --server http://localhost:8000 \
  --tenant engineering \
  search "How does ingestion crash recovery work?" \
  --top-k 5

rag --server http://localhost:8000 \
  --tenant engineering \
  ask "Summarize the ingestion architecture and cite the sources"
```

## Quick Import：产品级摄取入口

高级用户仍可使用低层 `/sources` + `/ingest/jobs` 两阶段 API；普通建库建议使用 Quick Import。

### 单 Source

```bash
curl -X POST http://localhost:8000/ingest/quick \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant_id": "engineering",
    "location": "/data/manuals",
    "name": "Engineering manuals",
    "tags": ["manuals"]
  }'
```

默认行为：

- Source ID 由 `tenant + source type + normalized location` 稳定派生；
- 已存在 Source 默认复用，并同步本次提供的 config/name/tags；
- 同一 Source 已有 `pending/running` Job 时默认返回该 Job，不重复排队；
- `idempotency_key` 可让重复请求返回**完全相同的 Job**；
- `idempotency_key` 要求稳定 Source identity，因此不能与 `reuse_source=false` 同时使用。

### 批量 Source

```bash
curl -X POST http://localhost:8000/ingest/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant_id": "engineering",
    "sources": [
      {"location": "/data/manuals"},
      {"location": "https://example.com/guide.pdf"},
      {"location": "https://github.com/magic-alt/ragbot", "config": {"ref": "main"}}
    ]
  }'
```

每个 Source 独立返回 submission 结果；某一项配置错误不会掩盖其他项状态。

## Python 本地开发

要求 Python 3.10+。

```bash
git clone https://github.com/magic-alt/ragbot.git
cd ragbot
python -m venv .venv

# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev,postgres,qdrant,worker,observability]"
cp .env.example .env
python -m pytest -q
uvicorn services.api.app.api:app --reload --host 0.0.0.0 --port 8000
```

开发模式允许 InMemoryRepo / InMemoryQdrant / HashEmbedder 与 inline ingestion，用于无外部依赖的快速验证；它们不等价于 production。

## Production mode

```bash
RAGBOT_ENV=production
RAGBOT_INGESTION_MODE=worker
```

production 会 fail fast，至少要求 PostgreSQL、Qdrant、semantic embedding、API keys 与 principal mapping。不会静默降级到 InMemory storage、HashEmbedder 或 API-process inline ingestion。

典型 principal：

```bash
RAGBOT_API_KEYS=tenant-a-key
RAGBOT_API_KEY_PRINCIPALS='{
  "tenant-a-key": {
    "tenant_ids": ["tenant-a"],
    "user_id": "svc-knowledge-a",
    "groups": ["engineering"],
    "roles": ["reader"],
    "admin": false
  }
}'
```

生产 Source 安全边界还包括：remote Web/PDF/Git SSRF 防护与 hostname allowlist、本地 source root allowlist、bounded download、HTTPS remote Git。高安全环境仍应在网络层配置 egress policy。

## Embedding 与检索

典型配置：

```bash
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=https://api.openai.com
QDRANT_DIM=1536
QDRANT_COLLECTION=rag_chunks
```

Hybrid retrieval 使用 Qdrant vector search + PostgreSQL FTS + RRF。连续 CJK 文本额外生成 bigram lexemes，例如：

```text
伺服驱动器 → 伺服 / 服驱 / 驱动 / 动器
```

仓库还包含真实 PostgreSQL/Qdrant 的 1000-PDF offline integration/capacity benchmark；详细方法、指标和限制见 [`docs/BENCHMARK_1000_PDF.md`](docs/BENCHMARK_1000_PDF.md)。该基准不是任意生产语料的 semantic quality 声明。

## Durable worker

```bash
RAGBOT_WORKER_POLL_SECONDS=1
RAGBOT_WORKER_LEASE_SECONDS=120
RAGBOT_WORKER_MAX_ATTEMPTS=3
python -m services.worker.main
```

Job 和 lease 持久化于 PostgreSQL。API restart 不会丢失 pending Job；worker 崩溃后 lease 到期可被其他 worker reclaim。

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/ingest/quick` | POST | 一步创建/复用 Source 并提交 ingestion |
| `/ingest/batch` | POST | 批量 Quick Import，单次最多 100 Source |
| `/ingest/jobs` | POST/GET | 低层 durable job enqueue / query |
| `/ingest/jobs/{job_id}/retry` | POST | 重试 failed Job |
| `/sources` | CRUD | 低层 Source 生命周期管理 |
| `/search` | POST | hybrid retrieval，适合作为 Agent knowledge tool |
| `/chat` | POST | Agentic RAG，支持 SSE |
| `/v1/chat/completions` | POST | OpenAI-compatible adapter |
| `/admin/health` | GET | process liveness |
| `/admin/ready` | GET | storage dependency readiness |
| `/admin/metrics` | GET | 质量/运行指标 |
| `/admin/cache` | GET | cache 统计 |

HTTP contract 的 source of truth 是 FastAPI `/openapi.json`。详细说明见 [`docs/API.md`](docs/API.md)。

## 部署与运维

- **Docker Compose**：API + worker + migration + PostgreSQL + Qdrant；可选 Ollama/Jaeger。
- **Helm**：API/worker Deployments、migration initContainer、readiness/liveness、Ingress、HPA、source mounts；production render 强制 durable worker。
- **Staging Smoke**：真实 provider credential + PostgreSQL + Qdrant，覆盖四类 Source、hybrid `/search`、Agent `/chat` 和 ACL negative isolation。

生产资料：

- [`docs/QUICKSTART.md`](docs/QUICKSTART.md)
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)
- [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md)
- [`docs/V1_RELEASE_READINESS.md`](docs/V1_RELEASE_READINESS.md)

## 测试与 CI

```bash
python -m pip check
python -m compileall -q contracts services cli eval
python -m pytest -q
```

GitHub Actions 覆盖 Python 3.10/3.12、PostgreSQL migrations/queue/FTS/CJK retrieval、Node SDK、Docker Compose、Helm、production/security regression 和 bundled example。正式 release 仍以 **exact release commit** 对应的 CI 与 staging evidence 为准。

## License

Ragbot 使用 **Apache License 2.0**。见 [`LICENSE`](LICENSE)。
