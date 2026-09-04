# Ragbot Configuration Guide

本文档定义 Ragbot 当前代码实际读取的主要运行配置。模板见根目录 `.env.example`。

## 1. 配置原则

1. `.env` 只用于本机配置，禁止提交；secrets 由部署平台注入。
2. **Embedding model、dimension、Qdrant collection 是一个索引契约。** 修改任一项后应重新索引。
3. API 与 ingestion worker 使用同一 embedding 配置和同一 PostgreSQL/Qdrant。
4. `HashEmbedder`、InMemoryRepo/InMemoryQdrant、inline ingestion 仅用于开发/测试。
5. `RAGBOT_ENV=production` 会拒绝上述非持久化回退。
6. Production API key 必须绑定 principal，不能让请求自行扩大 tenant/user scope。
7. **`POSTGRES_DSN` 只属于 Ragbot control plane。** Agent SQL 默认关闭；启用时使用独立 `RAGBOT_SQL_DSN`。

## 2. Runtime / durable ingestion

| Variable | Default | Meaning |
|---|---|---|
| `RAGBOT_ENV` | `development` | `production/prod` 开启生产 fail-fast |
| `RAGBOT_INGESTION_MODE` | `auto` | `auto` / `inline` / `worker`；production 禁止 inline |
| `RAGBOT_WORKER_ID` | hostname:pid | worker lease identity |
| `RAGBOT_WORKER_POLL_SECONDS` | `1` | 无任务时轮询间隔 |
| `RAGBOT_WORKER_LEASE_SECONDS` | `120` | job lease 时长 |
| `RAGBOT_WORKER_MAX_ATTEMPTS` | `3` | 最大 durable execution attempts |
| `RAGBOT_WORKER_RETRY_BASE_SECONDS` | `5` | job-level retry 初始 backoff |
| `RAGBOT_WORKER_RETRY_MAX_SECONDS` | `300` | job-level retry 最大 backoff |
| `RAGBOT_RECONCILE_SECONDS` | `30` | lease/failed-job reconciliation 扫描间隔 |
| `RAGBOT_SCHEDULER_SCAN_SECONDS` | `30` | Source scheduled-sync 扫描间隔 |
| `RAGBOT_PROVIDER_MAX_ATTEMPTS` | `4` | provider request 内层最大重试次数 |
| `RAGBOT_PROVIDER_BACKOFF_BASE_SECONDS` | `0.5` | provider retry 初始 backoff |
| `RAGBOT_PROVIDER_BACKOFF_MAX_SECONDS` | `30` | provider retry 最大 backoff |

`auto` 在存在 `POSTGRES_DSN` 时选择 durable worker，否则使用 inline development path。Docker Compose 显式设置 `worker` 并启动独立 worker；Helm production render 要求 `worker.enabled=true`。

Worker 入口：

```bash
python -m services.worker.main
```

任务 claim 使用 PostgreSQL `FOR UPDATE SKIP LOCKED`。Worker 会 heartbeat 续租；进程异常结束后，过期 lease 可被其他 worker reclaim。达到最大 attempts、永久配置错误或 Source generation mismatch 后任务进入 `dead_lettered`。`rag ingest --wait` 会把 `dead_lettered` 当作终态立即返回失败。

每个 Job 还会在 `stats.source_generation` 中记录 Source 生命周期代次。Source 更新/删除后，旧 Job 会被 fence；删除采用 tombstone-first，再 purge PostgreSQL/Qdrant knowledge。

## 3. LLM

| Variable | Default | Meaning |
|---|---|---|
| `RAGBOT_LLM_PROVIDER` | `openai` | `openai` 或 `ollama` |
| `OPENAI_API_KEY` | empty | OpenAI-compatible API key |
| `OPENAI_BASE_URL` | `https://api.openai.com` | OpenAI-compatible base URL |
| `OPENAI_MODEL` | `gpt-4o-mini` | 主模型 |
| `OPENAI_WEB_MODEL` | `OPENAI_MODEL` | Web search 模型 |
| `OPENAI_ORGANIZATION` | empty | optional organization |
| `OPENAI_PROJECT` | empty | optional project |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3` | Ollama model |

`/v1/chat/completions` 会保留 system message 和历史 user/assistant context，最后一个非空 user turn 作为当前 retrieval query；`temperature` 与 `max_tokens` 会进入 synthesis LLM 调用。当前 usage 为估算值，SSE 仍是 final-answer chunk streaming，不是 provider-native token streaming。

## 4. Embedding / Qdrant

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_MODEL` | empty | OpenAI-compatible embedding model；development 为空时 HashEmbedder |
| `EMBEDDING_API_KEY` | `OPENAI_API_KEY` | embedding API key |
| `EMBEDDING_BASE_URL` | `OPENAI_BASE_URL` | embedding endpoint |
| `QDRANT_URL` | empty | development 为空时 InMemoryQdrant |
| `QDRANT_API_KEY` | empty | Qdrant API key |
| `QDRANT_COLLECTION` | `rag_chunks` | collection name |
| `QDRANT_DIM` | 64(in-memory) / 1536(Qdrant) | vector dimension |

生产示例：

```bash
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=https://api.openai.com
QDRANT_DIM=1536
QDRANT_COLLECTION=rag_chunks
```

Embedding API 返回维度必须严格等于 `QDRANT_DIM`；不会 truncate 或 zero-pad。

### Embedding 模型升级

1. 确认新模型输出维度；
2. 创建新 collection，例如 `rag_chunks_v2`；
3. 设置对应 `QDRANT_DIM`；
4. 重新摄取所有 Source；
5. 验证 Recall/MRR/ACL/citation；
6. 切换查询流量；
7. 保留旧 collection 作为 rollback，再择机清理。

## 5. PostgreSQL control plane / lexical retrieval

| Variable | Default | Meaning |
|---|---|---|
| `POSTGRES_DSN` | empty | Ragbot production metadata/job/queue/FTS store |

`POSTGRES_DSN` 是 Ragbot 内部权威数据库，保存 Sources、Jobs、schedules、documents/chunks、ACL、queue leases 与 lexical state。**它不应直接暴露给 Agent SQL。**

Migrations 位于 `infra/migrations/`，由 `python -m services.api.app.storage.migrations` 按顺序、advisory lock 保护执行。

英文/数字使用 PostgreSQL `simple` FTS；CJK 文本额外生成 overlapping bigrams。`lexical_version` 属于 chunk reuse contract，因此 representation 升级后 re-ingest 会重写旧 lexical representation，后续相同内容仍可复用。

回归基准：

```bash
POSTGRES_TEST_DSN='postgresql://...' python -m eval.cjk_retrieval
```

当前 release floor：Recall@5 ≥ 0.90、MRR ≥ 0.80。

## 6. Optional Agent SQL

| Variable | Default | Meaning |
|---|---|---|
| `RAGBOT_SQL_TOOL_ENABLED` | `false` | 是否允许 Agent SQL route/tool |
| `RAGBOT_SQL_DSN` | empty | 独立结构化查询数据库 DSN |
| `RAGBOT_SQL_ALLOWED_SCHEMAS` | empty | Agent SQL schema allowlist；production 启用时必填 |
| `RAGBOT_SQL_LIMIT` | `200` | SQL result row limit |
| `RAGBOT_SQL_TIMEOUT_MS` | `3000` | statement timeout |

默认配置是 fail-closed：

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

Production startup 会拒绝 `RAGBOT_SQL_DSN == POSTGRES_DSN`，也会拒绝空 schema allowlist。数据库本身仍必须使用 dedicated read-only role、最小 schema/view grant；如果业务数据是多租户，使用 PostgreSQL RLS 或 tenant-safe views。应用层单条 SELECT / READ ONLY transaction / timeout / limit 只是 defense-in-depth。

## 7. Reranker

| Variable | Default | Meaning |
|---|---|---|
| `RAGBOT_RERANK_ENABLED` | `false` | 是否启用 rerank |
| `RAGBOT_RERANK_PROVIDER` | `cohere` | `cohere` 或 `local` |
| `RAGBOT_RERANK_MODEL` | provider default | model |
| `RAGBOT_RERANK_API_KEY` | empty | Cohere key |
| `RAGBOT_RERANK_BASE_URL` | empty | local rerank endpoint |
| `RAGBOT_RERANK_TOP_K` | `10` | rerank top-k |

Reranker 是 optional quality layer。Provider failure 时 Retriever 回退到 RRF，不中断整体检索。

## 8. API identity / ACL

| Variable | Default | Meaning |
|---|---|---|
| `RAGBOT_API_KEYS` | empty | comma-separated service credentials |
| `RAGBOT_API_KEY_PRINCIPALS` | empty | JSON API-key → tenant/user/groups/roles/admin mapping；production required |
| `RAGBOT_CORS_ORIGINS` | empty | comma-separated CORS origins |
| `RAGBOT_API_PORT` | `8000` | Compose exposed port |

示例：

```bash
RAGBOT_API_KEYS=tenant-a-key
RAGBOT_API_KEY_PRINCIPALS='{"tenant-a-key":{"tenant_ids":["tenant-a"],"user_id":"svc-a","groups":["engineering"],"roles":["reader"],"admin":false}}'
```

Production 每个 API key 都必须有 stable `user_id` 和 tenant scope 或 `admin=true`。

## 9. Source boundaries

| Variable | Default | Meaning |
|---|---|---|
| `RAGBOT_DATA_DIR` | `./data` | Compose host knowledge directory |
| `RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS` | empty | production local_fs/local PDF/local Git allow roots |
| `RAGBOT_WEB_ALLOWED_HOSTS` | empty | Web hostname allowlist |
| `RAGBOT_PDF_ALLOWED_HOSTS` | empty | remote PDF hostname allowlist |
| `RAGBOT_GIT_ALLOWED_HOSTS` | empty | remote Git hostname allowlist |
| `RAGBOT_ALLOW_PRIVATE_SOURCE_NETWORKS` | `false` | whether private/loopback/link-local destinations are allowed |
| `RAGBOT_WEB_MAX_REDIRECTS` | `5` | Web redirect limit |
| `RAGBOT_PDF_MAX_REDIRECTS` | `5` | PDF redirect limit |
| `RAGBOT_PDF_MAX_BYTES` | `26214400` | PDF download hard limit |
| `CODE_REPO_ROOT` | `.` | server-owned code-search root |

`RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS` 使用操作系统 path separator 表示多个根目录。生产中建议只读挂载 Source data。应用层 URL 验证不是 egress firewall 的替代品；敏感环境应同时限制网络出口。

## 10. Cache / tracing / metrics

| Variable | Default | Meaning |
|---|---|---|
| `RAGBOT_CACHE_ENABLED` | `true` | cache feature flag |
| `RAGBOT_CACHE_TTL_SECONDS` | `300` | cache TTL configuration |
| `RAGBOT_CACHE_MAX_ENTRIES` | `1000` | cache capacity |
| `RAGBOT_TRACING_ENABLED` | `false` | OpenTelemetry export |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | empty | OTLP endpoint |

Prometheus endpoint：

```text
GET /metrics
X-API-Key: <global-admin-key>
```

它输出 HTTP request/latency、queue status、oldest pending age、stale leases、Source counts，以及 Agent quality/latency gauges。`/admin/metrics` 仍保留用于诊断的 JSON 聚合。

## 11. Staging smoke

`.github/workflows/staging-smoke.yml` 是手动 release gate。GitHub `staging` environment 至少配置：

- secret `STAGING_OPENAI_API_KEY`；
- 可选 variable `STAGING_OPENAI_BASE_URL`；
- 可选 `STAGING_OPENAI_MODEL`；
- 可选 `STAGING_EMBEDDING_MODEL` / `STAGING_EMBEDDING_BASE_URL` / `STAGING_QDRANT_DIM`。

Workflow 以 production mode 启动真实 PostgreSQL/Qdrant/API/worker，运行 `eval/staging_smoke.py`，覆盖 local_fs、Web、PDF、Git、hybrid search、Agent `/chat` 和 ACL negative isolation。

## 12. 安装

```bash
# minimal
pip install -e .

# full development
pip install -e ".[dev,postgres,qdrant,worker,observability]"

# Docker/full runtime dependency set
pip install -r requirements.txt
```

`pyproject.toml` 是 package metadata source of truth；`requirements.txt` 是 Docker 完整 runtime 集合。

## 13. Production checklist

- `RAGBOT_ENV=production`；
- external PostgreSQL + Qdrant；
- semantic embedding model + correct dimension；
- `RAGBOT_INGESTION_MODE=worker` + 至少一个 durable worker；
- scoped API-key principals；
- source roots/egress allowlist；
- Agent SQL 默认关闭；若启用则 isolated read-only DSN + schema allowlist + DB-native tenancy；
- TLS、rate limit、secret management；
- Prometheus/OTLP 监控接入；
- PostgreSQL backup/restore 与 Qdrant snapshot/restore 实测；
- embedding reindex、migration、rollback runbook；
- CI + manual staging smoke 均通过后才发布 `v1.0.0`。
