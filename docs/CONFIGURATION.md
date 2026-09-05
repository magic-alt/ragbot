# Ragbot Configuration Guide

本文档只描述当前代码实际读取、且属于受支持运行契约的配置。模板见根目录 `.env.example`。

## 1. 配置原则

1. `.env` 只用于本机配置；secrets 由部署平台注入。
2. **Embedding model、dimension、Qdrant collection 是一个索引契约。** 修改任一项后应重新索引。
3. API 与 ingestion worker 使用同一 PostgreSQL/Qdrant 与 embedding contract。
4. `HashEmbedder`、InMemoryRepo/InMemoryQdrant、inline ingestion 仅用于开发/测试。
5. `RAGBOT_ENV=production` 会拒绝非持久化回退。
6. Production API key 必须绑定 scoped principal；请求不能自行扩大 tenant/user scope。
7. **`POSTGRES_DSN` 只属于 Ragbot control plane。** Agent SQL 默认关闭；启用时使用独立 `RAGBOT_SQL_DSN`。
8. Ragbot 当前**没有受支持的 runtime RetrievalCache**。仓库中的 local cache primitives 仅用于实验/测试，不进入 retrieval path，也没有 `RAGBOT_CACHE_*` 生产配置。
9. Retrieval 调参必须通过 vector/lexical/hybrid ablation 与 Recall@K/MRR 验证，避免只凭少量 Top-K 结果修改权重。

## 2. Runtime / durable ingestion

| Variable | Default | Meaning |
|---|---|---|
| `RAGBOT_ENV` | `development` | `production/prod` 开启生产 fail-fast |
| `RAGBOT_INGESTION_MODE` | `auto` | `auto` / `inline` / `worker`；production 禁止 inline |
| `RAGBOT_WORKER_ID` | hostname:pid | worker lease identity |
| `RAGBOT_WORKER_POLL_SECONDS` | `1` | 无任务时轮询间隔 |
| `RAGBOT_WORKER_LEASE_SECONDS` | `120` | job lease 时长 |
| `RAGBOT_WORKER_MAX_ATTEMPTS` | `3` | 最大 durable execution attempts |
| `RAGBOT_WORKER_RETRY_BASE_SECONDS` | `5` | job retry 初始 backoff |
| `RAGBOT_WORKER_RETRY_MAX_SECONDS` | `300` | job retry 最大 backoff |
| `RAGBOT_RECONCILE_SECONDS` | `30` | lease / dead-job reconciliation 间隔 |
| `RAGBOT_SCHEDULER_SCAN_SECONDS` | `30` | Source scheduled-sync 扫描间隔 |
| `RAGBOT_PROVIDER_MAX_ATTEMPTS` | `4` | provider 内层最大重试次数 |
| `RAGBOT_PROVIDER_BACKOFF_BASE_SECONDS` | `0.5` | provider retry 初始 backoff |
| `RAGBOT_PROVIDER_BACKOFF_MAX_SECONDS` | `30` | provider retry 最大 backoff |

`auto` 在存在 `POSTGRES_DSN` 时选择 durable worker，否则走 inline development path。Production/Compose/Helm 使用独立 `python -m services.worker.main` worker。

Durable queue 唯一实现位于 repository/PostgreSQL lease contract；旧 `services/worker/queue.py` abstraction 已删除。Worker 使用 `FOR UPDATE SKIP LOCKED` claim、heartbeat、expired lease reclaim、bounded retry 和 DLQ。`rag ingest --wait` 将 `dead_lettered` 视为立即终态失败。

每个 Job 在 `stats.source_generation` 中冻结 Source 生命周期代次。Source 更新/删除后旧 Job 会被 fence；删除采用 tombstone-first，再 purge PostgreSQL/Qdrant knowledge。

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

`/v1/chat/completions` 保留 system message 和历史 user/assistant context，最后一个非空 user turn 用于当前 retrieval；`temperature` / `max_tokens` 会进入 synthesis。当前 usage 是估算值，SSE 是 final-answer chunk streaming，不是 provider-native token streaming。

## 4. Embedding / Qdrant

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_MODEL` | empty | OpenAI-compatible embedding model；未形成 semantic endpoint 时 development 使用 HashEmbedder |
| `EMBEDDING_API_KEY` | `OPENAI_API_KEY` | embedding API key |
| `EMBEDDING_BASE_URL` | `OPENAI_BASE_URL` | embedding endpoint |
| `EMBEDDING_ALLOW_ANONYMOUS` | `false` | 显式允许无认证的远端 embedding endpoint；localhost/loopback/`host.docker.internal` 自动允许 |
| `EMBEDDING_TIMEOUT_SECONDS` | `30` | 每批 embedding HTTP 请求超时（正整数秒）；本机大模型可设为 `300` |
| `EMBEDDING_BATCH_SIZE` | `100` | 每次 embedding 请求的文本数（正整数）；本机大模型可设为 `8` |
| `EMBEDDING_QUERY_INSTRUCTION` | model default | query-side retrieval instruction；Qwen3 Embedding 自动提供默认 instruction |
| `QDRANT_URL` | empty | development 为空时 InMemoryQdrant |
| `QDRANT_API_KEY` | empty | Qdrant API key |
| `QDRANT_COLLECTION` | `rag_chunks` | collection name |
| `QDRANT_DIM` | model-known dimension / 64 fallback | vector dimension；显式值优先 |

Embedding 返回维度必须严格等于 `QDRANT_DIM`。已知 Qwen3 Embedding 维度可从模型名推断，包括常见 Ollama quantization suffix：

| Model family | Native dimension |
|---|---:|
| `qwen3-embedding:0.6b*` | 1024 |
| `qwen3-embedding:4b*` | 2560 |
| `qwen3-embedding:8b*` | 4096 |

本地 Ollama 示例：

```dotenv
EMBEDDING_MODEL=qwen3-embedding:0.6b
EMBEDDING_BASE_URL=http://127.0.0.1:11434
QDRANT_DIM=1024
```

无需伪造 API key。远端无认证 endpoint 必须显式设置 `EMBEDDING_ALLOW_ANONYMOUS=true`；否则 development 回退到 HashEmbedder，production 则因非 semantic fallback fail-fast。

Qwen3 query 使用 `Instruct: ...\nQuery:...` 形态，document embedding 保持原始正文。这样不会把 query instruction 污染进 corpus vector。

模型升级应使用 compatible collection → re-ingest → retrieval/ACL/citation gate → traffic cutover → rollback window。Chunk reuse identity 现在包含 `embedding_model` 与 `embedding_dimension`；切换模型/维度后，即使正文 checksum 没变化，也会强制重新 embedding，避免 benchmark 静默复用旧向量。

更完整的模型比较、ablation 与本地 Qwen 指南见 [`RETRIEVAL_QUALITY.md`](RETRIEVAL_QUALITY.md)。

## 5. PostgreSQL control plane / lexical retrieval

| Variable | Default | Meaning |
|---|---|---|
| `POSTGRES_DSN` | empty | production metadata/job/queue/FTS store |

PostgreSQL 保存 Sources、Jobs、schedules、documents/chunks、ACL、queue leases 与 lexical state。Migrations 位于 `infra/migrations/`，由 `python -m services.api.app.storage.migrations` 在 advisory lock 下执行。

英文/数字使用 PostgreSQL `simple` FTS；CJK 额外生成 overlapping bigrams。回归基准：

```bash
POSTGRES_TEST_DSN='postgresql://...' python -m eval.cjk_retrieval
```

Release floor：Recall@5 ≥ 0.90、MRR ≥ 0.80。

## 6. Retrieval fusion / quality controls

| Variable / request field | Default | Meaning |
|---|---|---|
| `RAGBOT_RETRIEVAL_CANDIDATE_POOL` | `max(40, top_k*4)` when unset | vector/lexical pre-rerank recall budget；最大 200 |
| `/search.mode` | `hybrid` | `vector` / `lexical` / `hybrid` ablation mode |
| `/search.candidate_pool` | env/default | request-level candidate budget override |
| `/search.rerank` | `true` | 是否应用已配置 reranker；ablation 可显式关闭 |
| `/search.explain` | `false` | 标记 diagnostic request；retrieval trace 仍保持结构化兼容字段 |

Hybrid 使用 adaptive RRF：强 lexical evidence 保持 50/50；弱 lexical evidence 提升 vector 权重；CJK query 对英文 corpus 仅靠少量 ASCII token 命中时，lexical 权重降至 0.1，避免例如单个 `GPU` token 获得与完整 lexical match 相同的融合权威。

CLI：

```bash
python -m cli.rag search "query" --mode vector --no-rerank --explain
python -m cli.rag search "query" --mode lexical --no-rerank --explain
python -m cli.rag search "query" --mode hybrid --candidate-pool 50 --no-rerank --explain
```

DeepSeek in Action benchmark：

```bash
python scripts/retrieval_ablation.py eval/datasets/deepseek_in_action_retrieval.json \
  --tenant engineering --candidate-pool 50
```

默认关闭 reranker，以隔离 first-stage retrieval/fusion；再加 `--with-reranker` 测 cross-encoder 的增量收益。报告包含整体及 exact/paraphrase/cross-lingual 分类的 Recall@1/3/5/10 与 MRR@10。

## 7. Optional Agent SQL

| Variable | Default | Meaning |
|---|---|---|
| `RAGBOT_SQL_TOOL_ENABLED` | `false` | 是否允许 Agent SQL |
| `RAGBOT_SQL_DSN` | empty | 独立结构化查询 DSN |
| `RAGBOT_SQL_ALLOWED_SCHEMAS` | empty | production 启用时必填的 schema allowlist |
| `RAGBOT_SQL_LIMIT` | `200` | row limit |
| `RAGBOT_SQL_TIMEOUT_MS` | `3000` | statement timeout |

Production 会拒绝 `RAGBOT_SQL_DSN == POSTGRES_DSN`，也会拒绝空 allowlist。数据库仍应使用 dedicated read-only role、最小 grants，以及多租户场景下的 RLS 或 tenant-safe views。

## 8. Reranker

| Variable | Default | Meaning |
|---|---|---|
| `RAGBOT_RERANK_ENABLED` | `false` | 是否启用 rerank |
| `RAGBOT_RERANK_PROVIDER` | `cohere` | `cohere` 或 `local` |
| `RAGBOT_RERANK_MODEL` | provider default | model |
| `RAGBOT_RERANK_API_KEY` | empty | provider key |
| `RAGBOT_RERANK_BASE_URL` | empty | local rerank endpoint |
| `RAGBOT_RERANK_TOP_K` | `10` | provider factory default top-k；请求最终 Top-K 仍由 search `top_k` 控制 |

Provider failure 时 Retriever 回退到 pre-rerank ranking。`candidate_pool` 控制 cross-encoder 最多看到多少候选，`top_k` 控制最终输出；两者不再绑定为固定 `top_k * 2`。

## 9. API identity / RBAC / ACL

| Variable | Default | Meaning |
|---|---|---|
| `RAGBOT_API_KEYS` | empty | comma-separated service credentials |
| `RAGBOT_API_KEY_PRINCIPALS` | empty | JSON API-key → tenant/user/groups/roles/admin mapping；production required |
| `RAGBOT_CORS_ORIGINS` | empty | comma-separated CORS origins |
| `RAGBOT_API_PORT` | `8000` | exposed API port |

Production 每个非 admin principal 都必须有 stable `user_id`、tenant scope，以及至少一个平台 RBAC role。

### Platform capability matrix

| Capability | reader | operator | owner | global admin |
|---|:---:|:---:|:---:|:---:|
| `knowledge.query` | ✓ | ✓ | ✓ | ✓ |
| `catalog.read` | ✓ | ✓ | ✓ | ✓ |
| `feedback.write` | ✓ | ✓ | ✓ | ✓ |
| `source.create` |  | ✓ | ✓ | ✓ |
| `source.update` |  | ✓ | ✓ | ✓ |
| `source.sync` |  | ✓ | ✓ | ✓ |
| `ingestion.run` |  | ✓ | ✓ | ✓ |
| `ingestion.retry` |  | ✓ | ✓ | ✓ |
| `source.delete` |  |  | ✓ | ✓ |
| global admin/metrics/reconcile |  |  |  | ✓ |

`admin=true` 是全局运维身份，不是 tenant owner 的别名。`roles` 中还可以包含 ACL 业务角色（例如 `finance-approver`）；这些角色参与 document ACL scope，但**不会自动获得平台 capabilities**。

示例：

```bash
RAGBOT_API_KEYS=tenant-a-key
RAGBOT_API_KEY_PRINCIPALS='{"tenant-a-key":{"tenant_ids":["tenant-a"],"user_id":"svc-a","groups":["engineering"],"roles":["operator","finance-approver"],"admin":false}}'
```

`GET /catalog/session` 返回 legacy UI summary 以及完整 `effective_capabilities` / `role_capability_matrix`。

## 10. Source boundaries

| Variable | Default | Meaning |
|---|---|---|
| `RAGBOT_DATA_DIR` | `./data` | host knowledge directory |
| `RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS` | empty | production local source allow roots |
| `RAGBOT_WEB_ALLOWED_HOSTS` | empty | Web hostname allowlist |
| `RAGBOT_PDF_ALLOWED_HOSTS` | empty | remote PDF allowlist |
| `RAGBOT_GIT_ALLOWED_HOSTS` | empty | remote Git allowlist |
| `RAGBOT_ALLOW_PRIVATE_SOURCE_NETWORKS` | `false` | permit private/loopback/link-local targets |
| `RAGBOT_WEB_MAX_REDIRECTS` | `5` | Web redirect limit |
| `RAGBOT_PDF_MAX_REDIRECTS` | `5` | PDF redirect limit |
| `RAGBOT_PDF_MAX_BYTES` | `26214400` | PDF hard download limit |
| `CODE_REPO_ROOT` | `.` | server-owned code-search root |

应用层 URL 验证不是 egress firewall 的替代品。

## 11. Observability / metrics

| Variable | Default | Meaning |
|---|---|---|
| `RAGBOT_TRACING_ENABLED` | `false` | enable OpenTelemetry tracing |
| `RAGBOT_OTEL_METRICS_ENABLED` | `false` | enable OTLP metrics export |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | empty | OTLP collector endpoint |

Prometheus：

```text
GET /metrics
X-API-Key: <global-admin-key>
```

Agent request/tool/latency/feedback 在事件发生时直接写入 Prometheus Counter/Histogram，因此 Prometheus 可以跨 API replicas 聚合；queue/source gauges 从 shared repository 在 scrape 时刷新。主要指标包括：

- `ragbot_agent_requests_total`
- `ragbot_agent_request_duration_seconds`
- `ragbot_agent_retrieval_duration_seconds`
- `ragbot_agent_tool_calls_total`
- `ragbot_agent_tool_duration_seconds`
- `ragbot_agent_feedback_total`
- `ragbot_http_requests_total`
- `ragbot_ingestion_jobs`
- `ragbot_ingestion_oldest_pending_age_seconds`
- `ragbot_ingestion_stale_running_leases`

`/admin/metrics` 与 `/admin/metrics/history` 仍是**当前 API process 的 bounded diagnostic history**，只用于 request inspection / feedback lookup，不是生产监控 backend。

启用 OTLP metrics：

```dotenv
RAGBOT_OTEL_METRICS_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
```

## 12. CLI ownership

唯一产品 CLI implementation 是 `cli/rag.py`，同时服务：

```bash
rag ...
python -m cli.rag ...
```

`scripts/ragbot.py` 只负责 setup/up/down/restart/status/logs 和部署路径映射；ask/search/ingest/import/doctor 最终委托给 `cli.rag`。旧 `cli/rag_impl.py` 与 `scripts/ragbot_impl.py` 已删除。

## 13. Staging / install / production gate

Full development：

```bash
pip install -e ".[dev,postgres,qdrant,worker,observability]"
```

Production checklist：external PostgreSQL + Qdrant、semantic embeddings、durable worker、explicit RBAC principal、Source egress/root policy、SQL fail-closed、Prometheus/OTLP integration、backup/restore drill、migration/reindex/rollback runbook，以及 CI + staging smoke 全通过。
