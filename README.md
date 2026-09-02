# Ragbot

Ragbot 是一个面向本地与企业知识库的 **Agentic RAG product/service**。它可以把 PDF、网页、Git、本地目录、S3/MinIO、Google Drive、Notion 和 Confluence 快速构建为可检索知识库，并通过 Qdrant + PostgreSQL 混合检索，为 Agent、IDE、内部应用和自动化流程提供知识底座。

> 当前 package / FastAPI / Helm metadata 仍为 `0.5.0`。项目已经进入 v1.0 release-gate 阶段，但只有 exact-head CI、真实 provider staging、生产运维/灾备 gate 全部通过后才会通过独立 release-only PR 发布 `1.0.0`。见 [`docs/V1_RELEASE_READINESS.md`](docs/V1_RELEASE_READINESS.md)。

## 核心能力

- **快速建库**：`rag ingest ... --wait` 一步创建/复用 Source、提交 durable ingestion 并等待可用；支持 manifest 批量导入。
- **多源摄取**：PDF / Web / Git / local filesystem / S3-MinIO / Google Drive / Notion / Confluence。
- **可恢复摄取**：PostgreSQL durable queue、worker claim、lease、heartbeat、crash recovery、durable retry/backoff、dead-letter queue、reconciliation。
- **周期同步**：Source-level scheduled sync，多 worker 下 deterministic Job ID + atomic insert-if-absent。
- **SaaS 增量复用**：Drive / Notion / Confluence 使用 metadata-first refresh，未变化文档跳过正文下载和 embedding。
- **混合检索**：Qdrant vector search + PostgreSQL FTS/CJK bigram + RRF，可选 reranker。
- **Agentic RAG**：route → retrieve/sql/code/web → synthesize → verify → finalize，输出 citation。
- **多租户与 ACL**：API-key principal 绑定 tenant/user/groups/roles，trusted identity 决定检索与写入范围。
- **RBAC**：reader / operator / owner / global admin。
- **管理控制面**：内置 `/admin/ui`、Source Catalog、Job progress、Retry、DLQ Requeue、scheduled sync、queue health。
- **生产部署**：Docker Compose、Helm、KEDA worker backlog autoscaling、health/readiness、metrics/tracing、backup/restore smoke。
- **兼容 AI 工具链**：`/search`、`/chat`、OpenAI-compatible `/v1/chat/completions`、CLI、Node SDK。

## 架构

```text
CLI / Admin UI / SDK / Agent / IDE / Application
                    │
          REST / SSE / OpenAI-compatible API
                    ▼
┌────────────────────────── Ragbot API ──────────────────────────┐
│ API key → trusted principal → tenant/user/groups/roles        │
│                                                               │
│ Quick Import → stable Source → PostgreSQL durable queue       │
│ Catalog / schedule / Retry / Requeue / reconcile              │
│                                                               │
│ Query → Qdrant vector ─┐                                     │
│         PostgreSQL FTS ├─ RRF / rerank → Agent → citations   │
│         CJK bigrams ───┘                                     │
└─────────────────────────────┬─────────────────────────────────┘
                              │
                       PostgreSQL / Qdrant
                              ▲
                              │ claim + lease + heartbeat
┌──────────────────── Ragbot Ingestion Worker ──────────────────┐
│ immutable Job connector snapshot                              │
│   → connector/provider short retry                            │
│   → parse/chunk/dedup/embed                                    │
│   → PostgreSQL metadata/FTS + Qdrant vectors                  │
│   → durable retry/backoff → DLQ when exhausted/permanent      │
│ Scheduler + queue reconciliation                              │
└───────────────────────────────────────────────────────────────┘
```

关键生产不变量：

1. 摄取和查询必须使用同一 embedding contract；维度不匹配直接失败，不静默 truncate/zero-pad。
2. API principal 决定可信 tenant/user/groups/roles；请求不能扩大授权范围。
3. production 摄取必须使用独立 durable worker，不依赖 API 进程生命周期。
4. Job 的 connector config 是 immutable submission snapshot。
5. SaaS secret 不进入 Source/Job 配置，只保存 `credential_ref=env:VARIABLE`。
6. PostgreSQL 与 Qdrant 都属于 durable knowledge state；生产恢复必须同时考虑两者。

## 60 秒建立 RAG 数据库

完整教程见 [`docs/QUICKSTART.md`](docs/QUICKSTART.md)。

### 1. 启动

```bash
cp .env.example .env
mkdir -p data
docker compose up -d --build
```

默认启动 API、worker、migration、PostgreSQL 16 和 Qdrant v1.19.0。

### 2. 检查部署

```bash
python -m pip install -e ".[postgres,qdrant,worker,s3,saas]"
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

Git：

```bash
rag --server http://localhost:8000 \
  --tenant engineering \
  ingest https://github.com/magic-alt/ragbot \
  --ref main \
  --wait
```

Google Drive：

```bash
cp .env.worker.example .env.worker
# 编辑 .env.worker，例如 RAGBOT_DRIVE_TOKEN=...
RAGBOT_WORKER_ENV_FILE="$PWD/.env.worker" docker compose up -d --build

rag --server http://localhost:8000 \
  --tenant engineering \
  ingest gdrive://1AbCdEfFolder \
  --credential-ref env:RAGBOT_DRIVE_TOKEN \
  --wait
```

SaaS token 只进入 worker；CLI/API 只提交 secret reference。

### 4. 批量导入

```bash
rag --server http://localhost:8000 \
  import examples/ragbot-manifest.json \
  --wait
```

HTTP batch API 单次最多 100 个 Source。

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

## Admin Control Plane

访问：

```text
http://localhost:8000/admin/ui
```

无需额外前端 build。UI 支持：

- Quick Import；
- Source Catalog；
- ingestion progress；
- failed Job Retry；
- **Dead Lettered** count / failure class；
- DLQ **Requeue snapshot**；
- Source pause/resume；
- scheduled sync；
- queue health；
- 当前 reader/operator/owner/admin 能力提示。

API key 仅保存在浏览器 `sessionStorage`。前端禁用写按钮只是体验层；真正 RBAC 始终由后端执行。

## Production mode

```bash
RAGBOT_ENV=production
RAGBOT_INGESTION_MODE=worker
```

production 至少要求 PostgreSQL、Qdrant、semantic embedding、API keys 和 principal mappings，不会静默降级到 InMemory storage、HashEmbedder 或 inline ingestion。

### Reader key：只查询

```bash
RAGBOT_API_KEYS=tenant-a-reader
RAGBOT_API_KEY_PRINCIPALS='{
  "tenant-a-reader": {
    "tenant_ids": ["tenant-a"],
    "user_id": "svc-reader-a",
    "groups": ["engineering"],
    "roles": ["reader"],
    "admin": false
  }
}'
```

适合 `/search`、`/chat`、catalog 和 Job read。**不能**用于 Quick Import、Source CRUD、schedule、Retry/Requeue。

### Operator key：建立/维护知识库

```bash
RAGBOT_API_KEYS=tenant-a-operator
RAGBOT_API_KEY_PRINCIPALS='{
  "tenant-a-operator": {
    "tenant_ids": ["tenant-a"],
    "user_id": "svc-knowledge-operator-a",
    "groups": ["engineering"],
    "roles": ["operator"],
    "admin": false
  }
}'
```

适合建库自动化与 Admin UI 的 tenant-scoped 写操作。`owner` 是 tenant operator 的 superset；`admin=true` 仅用于全局运维/reconcile 等管理面，不应给普通客户端。

如果一个部署同时需要 reader/operator/admin，多 key 应全部出现在 `RAGBOT_API_KEYS`，并在 `RAGBOT_API_KEY_PRINCIPALS` 中分别映射。

## Durable retry / DLQ / reconciliation

推荐 worker 参数：

```bash
RAGBOT_WORKER_POLL_SECONDS=1
RAGBOT_WORKER_LEASE_SECONDS=120
RAGBOT_WORKER_MAX_ATTEMPTS=3
RAGBOT_WORKER_RETRY_BASE_SECONDS=5
RAGBOT_WORKER_RETRY_MAX_SECONDS=300
RAGBOT_RECONCILE_SECONDS=30
RAGBOT_SCHEDULER_SCAN_SECONDS=30
RAGBOT_PROVIDER_MAX_ATTEMPTS=4
RAGBOT_PROVIDER_BACKOFF_BASE_SECONDS=0.5
RAGBOT_PROVIDER_BACKOFF_MAX_SECONDS=30
```

处理模型：

```text
provider transient error
  → short exponential retry / Retry-After
  → whole ingestion retry in durable queue
  → dead_lettered when permanent or exhausted
```

DLQ 默认使用失败 Job 的 immutable connector snapshot 重新入队。只有明确希望使用已修改 Source 配置时才选择 current Source config。

详细运维见 [`docs/ADMIN_OPERATIONS.md`](docs/ADMIN_OPERATIONS.md)。

## Cloud / SaaS Connectors

Drive、Notion、Confluence 当前采用 metadata-first synchronization：每次枚举远端 metadata，但只下载/embedding 新增或版本变化的内容；远端删除在 replacement snapshot 成功后清理。

这**不是** provider delta/change-feed API。真正的 Drive Changes API / persistent cursor/token 将作为独立优化里程碑实现，不会把现有 metadata-first 枚举错误宣传成 delta feed。

完整配置见 [`docs/CLOUD_CONNECTORS.md`](docs/CLOUD_CONNECTORS.md)。

## Embedding 与检索

```bash
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=...
EMBEDDING_BASE_URL=https://api.openai.com
QDRANT_DIM=1536
QDRANT_COLLECTION=rag_chunks
```

Hybrid retrieval 使用 Qdrant vector search + PostgreSQL FTS/CJK bigrams + RRF。仓库包含真实 PostgreSQL/Qdrant 的 deterministic 1000-PDF integration/capacity benchmark，见 [`docs/BENCHMARK_1000_PDF.md`](docs/BENCHMARK_1000_PDF.md)。

## Backup / Restore

Ragbot 提供：

```bash
bash scripts/backup_ragbot.sh ./backups/<name>
bash scripts/restore_ragbot.sh ./backups/<name>
```

它们覆盖 PostgreSQL custom-format dump/restore 与 Qdrant collection snapshot，并使用 SHA-256 manifest 校验。CI 中的 `Backup + restore smoke` 会真实执行 seed → backup → delete → restore → verify。

生产恢复前必须阅读 [`docs/DISASTER_RECOVERY.md`](docs/DISASTER_RECOVERY.md)。

## API 概览

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/ingest/quick` | POST | 一步创建/复用 Source 并提交 ingestion |
| `/ingest/batch` | POST | 批量 Quick Import |
| `/ingest/jobs` | POST/GET | durable Job trigger/list |
| `/ingest/jobs/{id}/retry` | POST | retry failed Job using current Source config |
| `/ingest/jobs/{id}/requeue` | POST | requeue dead-lettered Job |
| `/sources` | CRUD | low-level Source lifecycle |
| `/sources/{id}/sync` | PUT | scheduled sync |
| `/catalog/session` | GET | current principal capability summary |
| `/catalog/overview` | GET | tenant-scoped Source/queue/knowledge summary |
| `/catalog/sources` | GET | redacted Source Catalog |
| `/catalog/jobs` | GET | redacted Job history/failure class |
| `/search` | POST | direct hybrid retrieval |
| `/chat` | POST | Agentic RAG |
| `/v1/chat/completions` | POST | OpenAI-compatible adapter |
| `/admin/queue/metrics` | GET | global backlog/DLQ metrics |
| `/admin/queue/reconcile` | POST | global queue reconciliation |
| `/admin/ui` | GET | built-in operations console |
| `/admin/health` | GET | liveness |
| `/admin/ready` | GET | storage/vector readiness |

完整语义见 [`docs/API.md`](docs/API.md)。

## Python 本地开发

要求 Python 3.10+。

```bash
git clone https://github.com/magic-alt/ragbot.git
cd ragbot
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,postgres,qdrant,worker,s3,saas,observability]"
python -m pytest -q
uvicorn services.api.app.api:app --reload --host 0.0.0.0 --port 8000
```

## Deployment / Release documents

- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — 快速建库
- [`docs/API.md`](docs/API.md) — HTTP/RBAC/Job contracts
- [`docs/CLOUD_CONNECTORS.md`](docs/CLOUD_CONNECTORS.md) — S3/Drive/Notion/Confluence
- [`docs/ADMIN_OPERATIONS.md`](docs/ADMIN_OPERATIONS.md) — queue/DLQ/scheduler/operator runbook
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — Compose/Helm/KEDA/production deployment
- [`docs/DISASTER_RECOVERY.md`](docs/DISASTER_RECOVERY.md) — PostgreSQL + Qdrant backup/restore
- [`docs/BENCHMARK_1000_PDF.md`](docs/BENCHMARK_1000_PDF.md) — 1000-PDF integration/capacity baseline
- [`docs/V1_RELEASE_READINESS.md`](docs/V1_RELEASE_READINESS.md) — v1 release gates

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
