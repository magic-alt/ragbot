# Ragbot

> 面向本地与企业知识库的 Agentic RAG 服务：把 PDF、网页、Git、本地目录和 SaaS 数据源快速变成可检索、可引用、可供 Agent 使用的知识库。

Ragbot 提供 ingestion、hybrid retrieval、Agentic RAG、多租户 ACL、管理控制面、PostgreSQL durable worker、Qdrant 向量检索与生产部署能力，同时保留无需 Docker 的轻量本地开发路径。

> 当前 package / FastAPI / Helm metadata 为 `0.5.0`。v1 release gate 见 [`docs/V1_RELEASE_READINESS.md`](docs/V1_RELEASE_READINESS.md)。

## 60 秒上手

只要求 **Python 3.10+**：

```bash
git clone https://github.com/magic-alt/ragbot.git
cd ragbot
python scripts/ragbot.py up --mode auto
```

`--mode auto` 会自动选择：

- Docker 可用：启动 API + worker + PostgreSQL + Qdrant + migrations；
- Docker 不可用：启动纯 Python local 模式，使用 inline ingestion + in-memory stores。

启动完成后：

```text
API       http://127.0.0.1:8000
Admin UI  http://127.0.0.1:8000/admin/ui
```

检查：

```bash
python scripts/ragbot.py status
python scripts/ragbot.py doctor
```

Windows PowerShell：

```powershell
python .\scripts\ragbot.py up --mode auto
# 或
.\scripts\ragbot.ps1 up --mode auto
```

Linux/macOS：

```bash
bash scripts/ragbot.sh up --mode auto
```

完整部署与 CLI 手册：[`docs/CLI_DEPLOYMENT.md`](docs/CLI_DEPLOYMENT.md)。

## 从文档到问答

把本地资料放到 `data/`：

```text
data/
├─ manuals/
│  ├─ architecture.md
│  └─ notes.txt
└─ pdf/
   └─ product_manual.pdf
```

导入：

```bash
python scripts/ragbot.py ingest data/manuals --tenant engineering --tag manuals
python scripts/ragbot.py ingest data/pdf/product_manual.pdf --tenant engineering --type pdf
```

检索与问答：

```bash
python scripts/ragbot.py search "How does ingestion recovery work?" \
  --tenant engineering --top-k 5

python scripts/ragbot.py ask \
  "Summarize the ingestion architecture and cite the sources" \
  --tenant engineering
```

日常运维：

```bash
python scripts/ragbot.py logs -f
python scripts/ragbot.py restart --mode local
python scripts/ragbot.py down
```

## 两种本地部署模式

| 模式 | 命令 | Storage | Ingestion | 持久化 | 适合 |
| --- | --- | --- | --- | --- | --- |
| Local | `python scripts/ragbot.py up --mode local` | InMemoryRepo + InMemoryQdrant | inline | API 进程生命周期 | 快速开发、功能验证 |
| Docker | `python scripts/ragbot.py up --mode docker` | PostgreSQL + Qdrant | durable worker | Docker volumes | 长期本地知识库、接近生产拓扑 |

Bootstrap helper 会创建本地环境、安装依赖、启动服务、等待 readiness，并为 Docker Source 自动处理 `./data/...` → `/data/...` 的路径映射。

## Semantic Embedding / LLM

真实 semantic retrieval 至少配置：

```dotenv
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=<your-key>
EMBEDDING_BASE_URL=https://api.openai.com
QDRANT_DIM=1536
```

使用 `ask` 再配置 LLM：

```dotenv
RAGBOT_LLM_PROVIDER=openai
OPENAI_API_KEY=<your-key>
OPENAI_BASE_URL=https://api.openai.com
OPENAI_MODEL=gpt-4o-mini
```

development 环境在 embedding key 缺失时允许 HashEmbedder fallback。它只适合 pipeline smoke，不应用于评估真实语义检索质量。

## 核心能力

- **快速建库**：`ingest` 创建/复用 Source、提交 ingestion，并等待知识可用；`dead_lettered` 是明确终态，CLI 会立即失败而不是等待超时。
- **多源摄取**：PDF / Web / Git / local filesystem / S3-MinIO / Google Drive / Notion / Confluence。
- **可恢复摄取**：PostgreSQL durable queue、worker claim、lease、heartbeat、retry/backoff、DLQ、reconciliation。
- **Source generation fencing**：Job 提交时快照 Source 生命周期代次；Source 更新/删除会 fence 旧 Job，删除采用 tombstone-first → purge，降低运行中 worker 回写已删除知识的风险。
- **周期同步**：Source-level scheduled sync，多 worker deterministic Job ID + atomic insert-if-absent。
- **SaaS 增量复用**：Drive / Notion / Confluence metadata-first refresh，未变化内容跳过正文下载和 embedding。
- **混合检索**：Qdrant vector + PostgreSQL FTS/CJK bigram + RRF，可选 reranker。
- **Agentic RAG**：route → retrieve/sql/code/web → synthesize → verify → finalize，输出 citation。
- **多租户与 ACL**：API-key principal 绑定 tenant/user/groups/roles；RAG 检索在证据进入 synthesis 前执行 tenant/ACL pre-filter。
- **RBAC**：tenant-scoped read；operator/owner mutation；global admin 管理面。
- **SQL Tool Security Boundary**：Agent SQL 默认关闭；生产启用时必须使用独立 `RAGBOT_SQL_DSN`、专用只读数据库角色和 schema allowlist，禁止复用 Ragbot 内部 `POSTGRES_DSN`。
- **管理控制面**：内置 `/admin/ui`、Source Catalog、Job progress、Retry/Requeue、scheduled sync、queue health。
- **生产部署**：Docker Compose、Helm、KEDA、health/readiness、OpenTelemetry tracing、Prometheus `/metrics`、backup/restore。
- **AI 工具链兼容**：`/search`、`/chat`、OpenAI-shaped `/v1/chat/completions`、CLI、Node SDK；chat-completions 会保留 system 与多轮 user/assistant 上下文，并传递 `temperature` / `max_tokens`。

## 架构

```text
CLI / Admin UI / SDK / Agent / IDE / Application
                    │
          REST / SSE / OpenAI-shaped API
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
│                                                               │
│ Agent SQL: disabled by default → isolated read-only SQL DSN   │
└─────────────────────────────┬─────────────────────────────────┘
                              │
                       PostgreSQL / Qdrant
                              ▲
                              │ claim + lease + heartbeat
┌──────────────────── Ragbot Ingestion Worker ──────────────────┐
│ immutable Job Source snapshot + Source generation fence      │
│ connector → parse/chunk/dedup/embed                           │
│ → PostgreSQL metadata/FTS + Qdrant vectors                    │
│ → durable retry/backoff → DLQ                                 │
│ scheduler + queue reconciliation                              │
└───────────────────────────────────────────────────────────────┘
```

## SQL Tool 安全边界

Ragbot 的内部 PostgreSQL 是 control-plane / queue / document metadata / FTS 的权威存储，**不是 Agent SQL 数据源**。默认：

```dotenv
RAGBOT_SQL_TOOL_ENABLED=false
```

如确实需要 Agent 查询结构化业务数据，生产环境必须显式配置隔离的数据面：

```dotenv
RAGBOT_SQL_TOOL_ENABLED=true
RAGBOT_SQL_DSN=postgresql://ragbot_reader:***@analytics-db:5432/analytics
RAGBOT_SQL_ALLOWED_SCHEMAS=rag_views,analytics
RAGBOT_SQL_LIMIT=200
RAGBOT_SQL_TIMEOUT_MS=3000
```

生产启动会拒绝 `RAGBOT_SQL_DSN == POSTGRES_DSN`。数据库侧仍应使用专用 read-only role、allowlisted views，并在多租户业务库中采用 RLS/tenant-safe views；应用层 SELECT 校验不是数据库权限边界的替代品。

## OpenAI-compatible API 边界

`POST /v1/chat/completions` 支持：

- system message；
- 多轮 user/assistant context；
- 最后一个 user turn 作为当前 retrieval query；
- `temperature`；
- `max_tokens`；
- non-stream / SSE stream transport。

当前 token usage 仍为估算值；SSE 输出当前是在 Agent final answer 产生后按 chunk 发送，并非底层 provider 的原生 token-by-token streaming。因此这里强调 **OpenAI-shaped transport compatibility**，不宣称与 OpenAI API 的全部行为/字段完全等价。

## Production Metrics

`/admin/metrics` 保留进程内的诊断 JSON；生产监控使用管理员保护的 Prometheus endpoint：

```text
GET /metrics
X-API-Key: <global-admin-key>
```

主要指标包括 HTTP request/latency、ingestion queue status、oldest pending age、stale leases、Source counts，以及 Agent citation/retrieval/tool-failure/latency 指标。跨副本聚合由 Prometheus 完成。

## 原生 `rag` CLI

```bash
rag --server http://localhost:8000 doctor

rag --server http://localhost:8000 \
  --tenant engineering \
  ingest /data/manuals \
  --wait

rag --server http://localhost:8000 \
  --tenant engineering \
  search "query" --top-k 5

rag --server http://localhost:8000 \
  --tenant engineering \
  ask "question"
```

主要命令：

```text
rag ingest
rag import
rag doctor
rag search
rag ask
rag patch
```

## Local Source 注意事项

`local_fs` 默认扫描：

```text
.txt  .md  .markdown  .rst  .csv  .log
```

PDF 使用独立 `pdf` connector。扫描型 PDF 需要先 OCR。Docker 模式建议本地 Source 均放在 `./data`，容器内对应 `/data`。

## Production 不变量

生产模式至少要求：

- `RAGBOT_ENV=production`；
- durable worker；
- PostgreSQL control plane；
- Qdrant；
- semantic embedding；
- scoped API keys / principal mappings；
- TLS / ingress / rate limit / egress policy；
- PostgreSQL + Qdrant backup / restore；
- real-provider staging smoke。

如果启用 Agent SQL，还额外要求独立 `RAGBOT_SQL_DSN` 和 schema allowlist。Production 不会静默降级到 InMemory storage、HashEmbedder 或 inline ingestion。

## Admin Control Plane

```text
http://localhost:8000/admin/ui
```

支持 Quick Import、Source Catalog、ingestion progress、Retry、DLQ Requeue、pause/resume、scheduled sync、queue health 和权限提示，无需额外前端 build。

## 文档导航

- [`docs/CLI_DEPLOYMENT.md`](docs/CLI_DEPLOYMENT.md) — 一键部署、Windows/Linux/macOS、日常 CLI 运维
- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — 从 Source 到 queryable knowledge base
- [`docs/API.md`](docs/API.md) — HTTP / RBAC / Job contracts
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) — 环境变量和 provider 配置
- [`docs/CLOUD_CONNECTORS.md`](docs/CLOUD_CONNECTORS.md) — S3 / Drive / Notion / Confluence
- [`docs/ADMIN_OPERATIONS.md`](docs/ADMIN_OPERATIONS.md) — queue / DLQ / scheduler / operator runbook
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — Compose / Helm / KEDA / production
- [`docs/DISASTER_RECOVERY.md`](docs/DISASTER_RECOVERY.md) — PostgreSQL + Qdrant backup / restore
- [`docs/BENCHMARK_1000_PDF.md`](docs/BENCHMARK_1000_PDF.md) — 1000-PDF integration/capacity baseline
- [`docs/V1_RELEASE_READINESS.md`](docs/V1_RELEASE_READINESS.md) — v1 release gates

## Backup / Restore

```bash
bash scripts/backup_ragbot.sh ./backups/<name>
bash scripts/restore_ragbot.sh ./backups/<name>
```

生产恢复前请阅读 [`docs/DISASTER_RECOVERY.md`](docs/DISASTER_RECOVERY.md)。

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
