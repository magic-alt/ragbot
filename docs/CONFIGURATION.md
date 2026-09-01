# Ragbot Configuration Guide

本文档定义 Ragbot 当前代码实际读取的主要运行配置，并给出本地开发、Docker Compose 与生产部署的建议。配置模板见根目录 `.env.example`。

## 1. 配置原则

1. `.env` 只用于本机配置，禁止提交；`.env.example` 只保留非敏感默认值。
2. **Embedding model、embedding dimension、Qdrant collection 是一个不可拆分的索引契约。** 修改其中任意一项后应重新索引。
3. 摄取与查询使用同一个 `AgentServices.embedder`；不要为 worker 单独配置另一个 embedding 空间。
4. `HashEmbedder` 仅是开发/测试回退，不是生产语义检索方案。
5. `RAGBOT_API_KEYS` 为空表示不校验 API Key，只适用于可信本地环境。

## 2. LLM

| Variable | Default | Meaning |
|---|---|---|
| `RAGBOT_LLM_PROVIDER` | `openai` | `openai` 或 `ollama` |
| `OPENAI_API_KEY` | empty | OpenAI-compatible API key |
| `OPENAI_BASE_URL` | `https://api.openai.com` | OpenAI-compatible base URL |
| `OPENAI_MODEL` | `gpt-4o-mini` | 主模型 |
| `OPENAI_WEB_MODEL` | `OPENAI_MODEL` | Web search 模型 |
| `OPENAI_ORGANIZATION` | empty | 可选 organization |
| `OPENAI_PROJECT` | empty | 可选 project |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3` | Ollama model |

`services/api/app/llm/router.py` 已有 fast/strong model-router primitive，但当前 `factory.py` 仍直接构建单一 provider，因此不要把 `RAGBOT_MODEL_FAST/STRONG` 当作当前生产配置契约；其完整接线应作为独立功能 PR 完成。

## 3. Embedding 与 Qdrant

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_MODEL` | empty | OpenAI-compatible embedding model；为空时使用 HashEmbedder |
| `EMBEDDING_API_KEY` | `OPENAI_API_KEY` | embedding API key |
| `EMBEDDING_BASE_URL` | `OPENAI_BASE_URL` | embedding endpoint |
| `QDRANT_URL` | empty | 为空使用 InMemoryQdrant；Docker 固定连接 `qdrant:6333` |
| `QDRANT_API_KEY` | empty | Qdrant API key |
| `QDRANT_COLLECTION` | `rag_chunks` | collection name |
| `QDRANT_DIM` | 64(in-memory) / 1536(Qdrant) | vector dimension |

典型 OpenAI 配置：

```bash
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=https://api.openai.com
QDRANT_DIM=1536
```

### 变更 embedding 模型时

不要直接让新模型查询旧 collection。建议流程：

1. 选择新模型并确认输出维度；
2. 创建新的 collection name（例如 `rag_chunks_v2`）；
3. 设置对应 `QDRANT_DIM`；
4. 重新摄取所有 Source；
5. 验证 Recall/MRR、ACL 与 citation；
6. 切换流量后再清理旧 collection。

## 4. Reranker

| Variable | Default | Meaning |
|---|---|---|
| `RAGBOT_RERANK_ENABLED` | `false` | 是否启用 rerank |
| `RAGBOT_RERANK_PROVIDER` | `cohere` | `cohere` 或 `local` |
| `RAGBOT_RERANK_MODEL` | provider default | 模型名 |
| `RAGBOT_RERANK_API_KEY` | empty | Cohere key |
| `RAGBOT_RERANK_BASE_URL` | empty | local rerank endpoint |
| `RAGBOT_RERANK_TOP_K` | `10` | rerank top-k |

## 5. PostgreSQL

| Variable | Default | Meaning |
|---|---|---|
| `POSTGRES_DSN` | empty | 为空使用 InMemoryRepo；设置后使用 PostgresRepo/SQL engine |
| `POSTGRES_ALLOWED_SCHEMAS` | empty | SQL 可访问 schema allowlist |
| `POSTGRES_SQL_LIMIT` | `200` | SQL result limit |
| `POSTGRES_SQL_TIMEOUT_MS` | `3000` | SQL timeout |

Docker Compose 内部使用 `postgres` 服务名，不使用宿主机 `.env` 中的 `localhost` DSN。数据库初始化脚本位于 `infra/migrations/`。

## 6. API / ACL / 本地文件

| Variable | Default | Meaning |
|---|---|---|
| `RAGBOT_API_KEYS` | empty | 逗号分隔 API keys；空值关闭校验 |
| `RAGBOT_CORS_ORIGINS` | empty | 逗号分隔 CORS origins |
| `CODE_REPO_ROOT` | `.` | code-search 根目录 |
| `RAGBOT_DATA_DIR` | `./data` | Docker 宿主机知识目录，挂载到 `/data` |
| `RAGBOT_API_PORT` | `8000` | Docker 暴露端口 |

生产部署应使用非空 API key，并在反向代理/API gateway 层增加 TLS、速率限制与更强身份认证。不要把 Qdrant/Postgres 管理端口直接暴露到公网。

## 7. Cache / Tracing

| Variable | Default | Meaning |
|---|---|---|
| `RAGBOT_CACHE_ENABLED` | `true` | cache feature flag |
| `RAGBOT_CACHE_TTL_SECONDS` | `300` | retrieval cache TTL |
| `RAGBOT_CACHE_MAX_ENTRIES` | `1000` | cache capacity |
| `RAGBOT_TRACING_ENABLED` | `false` | enable OpenTelemetry export |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | empty | OTLP endpoint |

Compose 中 Jaeger 位于 `observability` profile。只有同时启用 profile 与 `RAGBOT_TRACING_ENABLED=true` 才能获得完整本地 OTLP 链路。

## 8. 安装方式

最小包：

```bash
pip install -e .
```

完整开发环境：

```bash
pip install -e ".[dev,postgres,qdrant,worker,observability]"
```

Docker/full-runtime requirements：

```bash
pip install -r requirements.txt
```

`pyproject.toml` 是 Python package metadata 的权威来源；`requirements.txt` 是 Docker 使用的完整 runtime 集合。新增直接 import 的基础依赖时应同步判断它属于 base dependency 还是 optional extra，避免 editable install 与 Docker 环境行为不同。

## 9. Production checklist

- 配置真实 semantic embedding，并记录 model + dimension + collection version。
- `RAGBOT_API_KEYS` 非空；secrets 由部署平台注入，不写入镜像或仓库。
- Postgres/Qdrant 使用持久卷、备份、网络隔离与最小权限凭据。
- 固定并定期升级基础镜像/服务镜像版本；生产不要长期依赖 `latest`。
- 对重新摄取、失败重试、embedding 模型迁移建立 runbook。
- 使用 CI 的完整测试结果作为合并门禁，并在 GitHub repository settings 中把 CI 设为 `main` 的 required check。
