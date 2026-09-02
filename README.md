# Ragbot

Ragbot 是一个面向本地与企业知识库的 **Agentic RAG product/service**：把 PDF、网页、Git、本地目录、S3/MinIO、Google Drive、Notion 和 Confluence 快速构建为可检索知识库，通过 Qdrant + PostgreSQL 混合检索，为自身 Agent、其他 Agent、IDE 或业务应用提供知识底座。

> 当前 package / FastAPI / Helm metadata 仍为 `0.5.0`。代码已进入 v1.0 release-gate 阶段，但只有真实 provider staging、生产运维 gate 与 exact release commit 全部通过后才会发布 `1.0.0`。见 [`docs/V1_RELEASE_READINESS.md`](docs/V1_RELEASE_READINESS.md)。

## 你可以用 Ragbot 做什么

- **快速建库**：一个 `rag ingest ... --wait` 命令即可创建/复用 Source、提交任务并等待知识可用；也可用 manifest 批量导入。
- **多源摄取**：PDF / Web / Git / local filesystem / S3-MinIO / Google Drive / Notion / Confluence → chunk → dedup → embedding → PostgreSQL + Qdrant。
- **云知识增量同步**：Drive / Notion / Confluence 先比较远端 metadata/version，未变化文档直接复用已有 chunks，只下载和重新 embedding 新增/修改内容。
- **可恢复摄取**：PostgreSQL durable queue、worker claim、lease、heartbeat、crash recovery、bounded retry、scheduled sync。
- **管理控制面**：内置 `/admin/ui`、Source Catalog、Job progress/retry、pause/resume、周期同步、queue health。
- **幂等与重复部署友好**：同 tenant/type/location 默认复用稳定 Source；同配置 active Job 做便利性去重；跨 replica/并发严格幂等使用 deterministic `idempotency_key`。
- **混合检索**：Qdrant vector search + PostgreSQL FTS/CJK bigram + RRF，可选 cross-encoder rerank。
- **Agentic RAG**：route → retrieve/sql/code/web → synthesize → verify → finalize，输出 citation。
- **多租户与 ACL**：API-key principal 绑定 tenant/user/groups/roles，检索前置 ACL scope。
- **兼容现有 AI 工具链**：`/search`、`/chat`、OpenAI-compatible `/v1/chat/completions`、CLI、Node SDK。
- **生产部署**：Docker Compose、Helm、worker backlog autoscaling、health/readiness、metrics/tracing、真实 provider staging workflow。

## 架构

```text
CLI / Admin UI / SDK / Other Agents / IDE / Applications
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
│ Catalog / schedule / retry / queue operations                 │
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
│ Job connector snapshot                                       │
│   → PDF/Web/Git/local/S3/Drive/Notion/Confluence             │
│   → metadata-first incremental reuse → chunk/dedup            │
│   → shared Embedder                                           │
│        ├─ PostgreSQL metadata/FTS                             │
│        └─ Qdrant vectors                                      │
│ Scheduler scan → deterministic scheduled Job → durable queue  │
└───────────────────────────────────────────────────────────────┘
```

关键不变量：

1. **摄取和查询使用同一 embedding contract。** 模型或向量维度变化必须重新索引；Ragbot 不会静默 truncate/zero-pad。
2. **生产身份不能由请求自行扩大。** API key principal 决定可信 tenant/user/groups/roles。
3. **生产摄取不依赖 API 进程生命周期。** production 禁止 inline ingestion，任务必须由 durable worker 执行。
4. **排队 Job 的 connector 配置是 immutable snapshot。** Source 后续修改影响未来 Job，不会把已排队 Job 静默重定向。
5. **SaaS 密钥不进入 Source/Job 配置。** Source 只保存 `credential_ref=env:VARIABLE`；worker 在执行时解析真实 secret。

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

Git 仓库：

```bash
rag --server http://localhost:8000 \
  --tenant engineering \
  ingest https://github.com/magic-alt/ragbot \
  --ref main \
  --wait
```

Google Drive：

```bash
# token 只存在于 worker 环境；CLI 只传 secret reference
export RAGBOT_DRIVE_TOKEN='...'
rag --server http://localhost:8000 \
  --tenant engineering \
  ingest gdrive://1AbCdEfFolder \
  --credential-ref env:RAGBOT_DRIVE_TOKEN \
  --wait
```

默认可自动判断 `local_fs` / `pdf` / `repo` / `web` / `s3` / `gdrive` / `notion` / `confluence`。云端连接、安全 credential reference 和增量同步详见 [`docs/CLOUD_CONNECTORS.md`](docs/CLOUD_CONNECTORS.md)。

### 4. 用 manifest 批量建立知识库

仓库提供 [`examples/ragbot-manifest.json`](examples/ragbot-manifest.json)：

```bash
rag --server http://localhost:8000 \
  import examples/ragbot-manifest.json \
  --wait
```

manifest 可以混合声明本地、Web、Git、S3 与 SaaS Source。HTTP batch API 单次最多接收 100 个 Source。

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

部署完成后直接访问：

```text
http://localhost:8000/admin/ui
```

无需额外前端 build。控制台支持 Quick Import、Source Catalog、ingestion progress/retry、pause/resume、scheduled sync 和 queue health。API key 仅保存于浏览器 `sessionStorage`。Cloud Quick Import 只接受 `credential_ref`，不会要求用户把 token/private key 粘贴到 Source config。

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
- 已存在 Source 默认复用，并在可安全提交新 run 时同步本次提供的 config/name/tags；
- 同一 Source 已有**相同 connector config** 的 `pending/running` Job 时默认返回该 Job；
- 若 active Job 与新请求的 connector config 不同，返回 `409`；
- `idempotency_key` 可让重复请求返回**完全相同的 Job**，适合多 API replica/并发自动化；
- `idempotency_key` 要求稳定 Source identity，因此不能与 `reuse_source=false` 同时使用。

### 批量 Source

```bash
curl -X POST http://localhost:8000/ingest/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant_id": "engineering",
    "sources": [
      {"location": "/data/manuals"},
      {"location": "s3://engineering-manuals/servo/"},
      {
        "location": "notion://0123456789abcdef0123456789abcdef",
        "config": {"credential_ref": "env:RAGBOT_NOTION_TOKEN"}
      }
    ]
  }'
```

每个 Source 独立返回 submission 结果；某一项配置错误不会掩盖其他项状态。

## Cloud / SaaS Connectors

Google Drive、Notion 与 Confluence 都使用 metadata-first incremental sync：每次仍枚举远端 metadata 形成完整 Source snapshot，但只有远端版本变化的 document 才下载正文和重新 embedding；远端删除由 replacement pruning 清理。

统一安全合同：

```json
{
  "credential_ref": "env:RAGBOT_NOTION_TOKEN"
}
```

Ragbot API 只校验 reference，不解析真实 secret；worker 执行时才读取环境变量。生产 Kubernetes 推荐用 `worker.extraEnv` 或 `worker.extraEnvFrom` 从 Secret / ExternalSecret 注入这些变量，而不是把 secret 暴露给 API Pod。完整配置见 [`docs/CLOUD_CONNECTORS.md`](docs/CLOUD_CONNECTORS.md)。

## Python 本地开发

要求 Python 3.10+。

```bash
git clone https://github.com/magic-alt/ragbot.git
cd ragbot
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,postgres,qdrant,worker,s3,saas,observability]"
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

生产 Source 安全边界包括 remote Web/PDF/Git SSRF 防护、本地 source root allowlist、S3/MinIO endpoint allowlist、Confluence hostname allowlist、bounded download 和 HTTPS remote Git。高安全环境仍应配置网络层 egress policy。

## Embedding 与检索

```bash
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=https://api.openai.com
QDRANT_DIM=1536
QDRANT_COLLECTION=rag_chunks
```

Hybrid retrieval 使用 Qdrant vector search + PostgreSQL FTS + RRF。连续 CJK 文本额外生成 bigram lexemes。仓库还包含真实 PostgreSQL/Qdrant 的 1000-PDF offline integration/capacity benchmark，见 [`docs/BENCHMARK_1000_PDF.md`](docs/BENCHMARK_1000_PDF.md)。

## Durable worker 与 scheduled sync

```bash
RAGBOT_WORKER_POLL_SECONDS=1
RAGBOT_WORKER_LEASE_SECONDS=120
RAGBOT_WORKER_MAX_ATTEMPTS=3
RAGBOT_SCHEDULER_SCAN_SECONDS=30
python -m services.worker.main
```

Job 和 lease 持久化于 PostgreSQL。API restart 不会丢失 pending Job；worker 崩溃后 lease 到期可被其他 worker reclaim。Job enqueue 时复制 connector snapshot；Source 后续改动不会改变已排队任务。

Scheduled sync 使用 deterministic scheduled Job ID + atomic insert-if-absent，因此多个 worker 可以安全扫描同一批 due Sources；服务停机错过多个周期时只 collapse 为当前一次 refresh，不制造恢复期 ingestion storm。Drive/Notion/Confluence 的 scheduled refresh 会进一步复用远端版本未变化的 document chunks。

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/ingest/quick` | POST | 一步创建/复用 Source 并提交 ingestion |
| `/ingest/batch` | POST | 批量 Quick Import，单次最多 100 Source |
| `/ingest/jobs` | POST/GET | durable job enqueue / query |
| `/ingest/jobs/{job_id}/retry` | POST | 重试 failed Job |
| `/sources` | CRUD | Source 生命周期管理 |
| `/sources/{source_id}/sync` | PUT | 启用/关闭 scheduled sync |
| `/catalog/overview` | GET | tenant-scoped knowledge/queue summary |
| `/catalog/sources` | GET | Source Catalog，connector config 已脱敏 |
| `/catalog/jobs` | GET | ingestion progress/history |
| `/search` | POST | hybrid retrieval |
| `/chat` | POST | Agentic RAG，支持 SSE |
| `/v1/chat/completions` | POST | OpenAI-compatible adapter |
| `/admin/ui` | GET | 内置管理控制台 |
| `/admin/queue/metrics` | GET | admin queue/backlog metrics |
| `/admin/health` | GET | process liveness |
| `/admin/ready` | GET | storage dependency readiness |

HTTP contract 的 source of truth 是 FastAPI `/openapi.json`。详细说明见 [`docs/API.md`](docs/API.md)。

## 部署与运维

- **Docker Compose**：API + worker + migration + PostgreSQL + Qdrant；可选 Ollama/Jaeger。
- **Helm**：API/worker Deployments、migration initContainer、readiness/liveness、Ingress、HPA、source mounts、worker-only connector secret injection；production render 强制 durable worker。
- **KEDA**：可选基于 PostgreSQL durable queue backlog 的 worker autoscaling，而不是用 CPU 间接估计 backlog。
- **Staging Smoke**：真实 provider credential + PostgreSQL + Qdrant；SaaS connector 使用独立可选 staging credentials。

生产资料：

- [`docs/QUICKSTART.md`](docs/QUICKSTART.md)
- [`docs/CLOUD_CONNECTORS.md`](docs/CLOUD_CONNECTORS.md)
- [`docs/ADMIN_OPERATIONS.md`](docs/ADMIN_OPERATIONS.md)
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

GitHub Actions 覆盖 Python 3.10/3.12、PostgreSQL migrations/queue/FTS/CJK retrieval、SaaS incremental protocol regressions、Node SDK、Docker Compose、Helm、production/security regression 和 bundled example。正式 release 仍以 **exact release commit** 对应的 CI 与 staging evidence 为准。

## License

Ragbot 使用 **Apache License 2.0**。见 [`LICENSE`](LICENSE)。
