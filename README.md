# Ragbot

> 面向本地与企业知识库的 Agentic RAG 服务：把 PDF、网页、Git、本地目录和 SaaS 数据源快速变成可检索、可引用、可供 Agent 使用的知识库。

Ragbot 提供 ingestion、hybrid retrieval、Agentic RAG、多租户 ACL、正式 RBAC capability matrix、PostgreSQL durable worker、Qdrant 向量检索、Prometheus/OpenTelemetry telemetry 与生产部署能力，同时保留无需 Docker 的轻量本地开发路径。

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

把资料放到 `data/`：

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

Bootstrap controller 会创建本地环境、安装依赖、启动服务、等待 readiness，并为 Docker Source 自动处理 `./data/...` → `/data/...` 的路径映射。

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

development 环境在 embedding key 缺失时允许 HashEmbedder fallback。它只适合 pipeline smoke，不应用于真实语义检索质量评估。

## 核心能力

- **快速建库**：`ingest` 创建/复用 Source、提交 ingestion，并等待知识可用；`dead_lettered` 是明确终态，CLI 会立即失败而不是等到 timeout。
- **多源摄取**：PDF / Web / Git / local filesystem / S3-MinIO / Google Drive / Notion / Confluence。
- **可恢复摄取**：唯一 queue contract 是 PostgreSQL durable Jobs + worker claim/lease/heartbeat/retry/backoff/DLQ/reconciliation；旧 `worker/queue.py` 已删除。
- **Source generation fencing**：Job 快照 Source 生命周期代次；Source 更新/删除会 fence 旧 Job，删除采用 tombstone-first → purge。
- **周期同步**：Source-level scheduled sync，多 worker deterministic Job ID + atomic insert-if-absent。
- **SaaS 增量复用**：Drive / Notion / Confluence metadata-first refresh，未变化内容跳过正文下载和 embedding。
- **混合检索**：Qdrant vector + PostgreSQL FTS/CJK bigram + RRF，可选 reranker。
- **Agentic RAG**：route → retrieve/sql/code/web → synthesize → verify → finalize，输出 citation。
- **多租户与 ACL**：API-key principal 绑定 tenant/user/groups/roles；证据进入 synthesis 前执行 tenant/ACL pre-filter。
- **正式 RBAC**：reader → operator → owner capability hierarchy；owner 独占 destructive `source.delete`；`admin=true` 只用于全局运维。
- **SQL Tool Security Boundary**：Agent SQL 默认关闭；生产启用必须使用独立 `RAGBOT_SQL_DSN`、只读 DB identity 和 schema allowlist，禁止复用内部 `POSTGRES_DSN`。
- **管理控制面**：内置 `/admin/ui`、Source Catalog、Job progress、Retry/Requeue、scheduled sync、queue health。
- **生产 telemetry**：Prometheus Counter/Histogram + durable queue/source gauges；可选 OpenTelemetry/OTLP metrics + tracing。
- **单一 CLI**：`cli/rag.py` 是唯一产品 CLI implementation；`scripts/ragbot.py` 只负责 bootstrap/deployment 并委托产品命令。
- **AI 工具链兼容**：`/search`、`/chat`、OpenAI-shaped `/v1/chat/completions`、CLI、Node SDK；chat-completions 保留 system/多轮上下文并传递 `temperature` / `max_tokens`。

## 架构

```text
CLI / Admin UI / SDK / Agent / IDE / Application
                    │
          REST / SSE / OpenAI-shaped API
                    ▼
┌────────────────────────── Ragbot API ──────────────────────────┐
│ API key → trusted principal → RBAC capability → ACL scope     │
│                                                               │
│ Quick Import → stable Source → PostgreSQL durable queue       │
│ Catalog / schedule / Retry / Requeue / reconcile              │
│                                                               │
│ Query → Qdrant vector ─┐                                     │
│         PostgreSQL FTS ├─ RRF / rerank → Agent → citations   │
│         CJK bigrams ───┘                                     │
│                                                               │
│ Agent SQL: disabled by default → isolated read-only SQL DSN   │
│ Metrics: Prometheus + optional OTLP                           │
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

## RBAC capability matrix

| Capability | reader | operator | owner | global admin |
|---|:---:|:---:|:---:|:---:|
| `knowledge.query` | ✓ | ✓ | ✓ | ✓ |
| `catalog.read` | ✓ | ✓ | ✓ | ✓ |
| `feedback.write` | ✓ | ✓ | ✓ | ✓ |
| `source.create/update/sync` |  | ✓ | ✓ | ✓ |
| `ingestion.run/retry` |  | ✓ | ✓ | ✓ |
| `source.delete` |  |  | ✓ | ✓ |
| global metrics/reconcile/admin |  |  |  | ✓ |

业务 ACL role 可以和平台 role 共存，但任意自定义 role 不会自动获得平台 capability。Production 非 admin principal 必须显式包含 `reader`、`operator` 或 `owner` 之一。

## SQL Tool 安全边界

内部 PostgreSQL 是 control-plane / queue / document metadata / FTS 的权威存储，**不是 Agent SQL 数据源**。默认：

```dotenv
RAGBOT_SQL_TOOL_ENABLED=false
```

生产启用示例：

```dotenv
RAGBOT_SQL_TOOL_ENABLED=true
RAGBOT_SQL_DSN=postgresql://ragbot_reader:***@analytics-db:5432/analytics
RAGBOT_SQL_ALLOWED_SCHEMAS=rag_views,analytics
RAGBOT_SQL_LIMIT=200
RAGBOT_SQL_TIMEOUT_MS=3000
```

生产启动会拒绝 `RAGBOT_SQL_DSN == POSTGRES_DSN`。数据库侧仍应使用专用 read-only role、allowlisted views，并在多租户业务库中采用 RLS/tenant-safe views。

## Retrieval cache 边界

Ragbot 当前**没有 production RetrievalCache**。原先的 process-local cache 从未真正接入 retrieval，而且 API replicas 与 workers 之间没有一致 invalidation，因此相关 `RAGBOT_CACHE_*` 配置和 `/admin/cache` claim 已删除。

`services/api/app/cache/` 仅保留 local primitives 供单测/实验使用。未来只有在存在 shared / generation-aware invalidation contract，并且 benchmark 证明收益后，才应重新把缓存放到 retrieval path。

## OpenAI-compatible API 边界

`POST /v1/chat/completions` 支持 system、多轮 user/assistant context、最后 user turn retrieval、`temperature`、`max_tokens` 和 non-stream/SSE transport。

当前 token usage 是估算值；SSE 在 Agent final answer 生成后按 chunk 发送，不宣称 provider-native token streaming 或 OpenAI API 全字段等价。

## Production Metrics

`GET /metrics` 需要 global-admin principal。Agent request/tool/latency/feedback 在事件发生时直接进入 Prometheus Counter/Histogram；跨副本由 Prometheus 聚合。Queue/source gauges 从 shared repository 在 scrape 时刷新。

主要指标：

```text
ragbot_agent_requests_total
ragbot_agent_request_duration_seconds
ragbot_agent_retrieval_duration_seconds
ragbot_agent_tool_calls_total
ragbot_agent_tool_duration_seconds
ragbot_agent_feedback_total
ragbot_http_requests_total
ragbot_ingestion_jobs
ragbot_ingestion_oldest_pending_age_seconds
ragbot_ingestion_stale_running_leases
```

可选 OTLP metrics：

```dotenv
RAGBOT_OTEL_METRICS_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
```

`/admin/metrics` 和 `/admin/metrics/history` 只保留当前 API process 的 bounded diagnostics/request-history，不是生产 metrics backend。

## 单一 CLI ownership

产品 CLI：

```bash
rag --server http://localhost:8000 doctor
python -m cli.rag --server http://localhost:8000 --tenant engineering search "query"
```

唯一 implementation 是 `cli/rag.py`。`scripts/ragbot.py` 是 bootstrap/deployment controller，负责 setup/up/down/restart/status/logs/path mapping，ask/search/ingest/import/doctor 最终委托给 `cli.rag`。旧 `cli/rag_impl.py` 与 `scripts/ragbot_impl.py` 已删除。

## Local Source 注意事项

`local_fs` 默认扫描：

```text
.txt  .md  .markdown  .rst  .csv  .log
```

PDF 使用独立 `pdf` connector。扫描型 PDF 需要先 OCR。Docker 模式建议本地 Source 放在 `./data`，容器内对应 `/data`。

## Production 不变量

生产模式至少要求：

- `RAGBOT_ENV=production`；
- durable PostgreSQL worker；
- PostgreSQL control plane + Qdrant；
- semantic embedding；
- scoped API principals + explicit RBAC role；
- TLS / ingress / rate limit / egress policy；
- PostgreSQL + Qdrant backup / restore；
- Prometheus 或 OTLP telemetry；
- real-provider staging smoke。

如果启用 Agent SQL，还要求独立 `RAGBOT_SQL_DSN` 和 schema allowlist。Production 不会静默降级到 InMemory storage、HashEmbedder 或 inline ingestion。

## 文档导航

- [`docs/CLI_DEPLOYMENT.md`](docs/CLI_DEPLOYMENT.md) — 部署与 CLI 运维
- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — Source → queryable knowledge base
- [`docs/API.md`](docs/API.md) — HTTP / RBAC / Job contracts
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) — 环境变量与 provider 配置
- [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md) — authoritative architecture
- [`docs/CLOUD_CONNECTORS.md`](docs/CLOUD_CONNECTORS.md) — S3 / Drive / Notion / Confluence
- [`docs/ADMIN_OPERATIONS.md`](docs/ADMIN_OPERATIONS.md) — queue / DLQ / scheduler runbook
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — Compose / Helm / KEDA / production
- [`docs/DISASTER_RECOVERY.md`](docs/DISASTER_RECOVERY.md) — PostgreSQL + Qdrant backup / restore
- [`docs/V1_RELEASE_READINESS.md`](docs/V1_RELEASE_READINESS.md) — v1 release gates

## Backup / Restore

```bash
bash scripts/backup_ragbot.sh ./backups/<name>
bash scripts/restore_ragbot.sh ./backups/<name>
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
