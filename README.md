# Ragbot

Ragbot 是一个面向本地/企业知识库的 **Agentic RAG knowledge service**：把 PDF、网页、Git 仓库和本地文本目录摄取为可检索知识，通过 Qdrant + PostgreSQL 混合检索，为自身 Agent 或其他 Agent / 应用提供底层知识支撑。

> 当前代码已经具备 README 下述核心能力，并进入 v1.0 release-gate 阶段。Python package / FastAPI / Helm metadata 仍为 `0.5.0`；`1.0.0` 版本号、tag 与 GitHub Release 只会在真实 staging gate 通过后通过独立 release PR 处理。详细状态见 [`docs/V1_RELEASE_READINESS.md`](docs/V1_RELEASE_READINESS.md)。

## 能力范围

- **知识摄取**：PDF / Web / Git / local filesystem → chunk → dedup → embedding → Qdrant
- **持久化摄取队列**：PostgreSQL pending jobs → worker lease / heartbeat / crash recovery / bounded retry
- **混合检索**：Qdrant vector search + PostgreSQL FTS + RRF，可选 cross-encoder rerank
- **中文 lexical baseline**：PostgreSQL `simple` FTS + CJK bigram lexemes；CI 固定 Recall@5 / MRR regression floor
- **Agentic RAG**：route → retrieve/sql/code/web → synthesize → verify → finalize
- **多租户与 ACL**：tenant/user scope、ACL hash 前置过滤、API-key principal
- **LLM**：OpenAI-compatible provider 与 Ollama adapter
- **服务接口**：`/search`、`/chat`、OpenAI-compatible `/v1/chat/completions`、Source/Ingest API
- **工程能力**：SSE、可靠性封装、评测、指标/Tracing、Docker Compose、Helm、真实 provider staging workflow

这些不是仅有接口定义的占位能力：当前实现中存在完整 ingestion pipeline、durable worker、Qdrant/PostgreSQL retrieval、Agent graph、Source/Job persistence、SSE transport、迁移机制和部署配置。能力边界和 release gate 统一记录在 v1 readiness 文档。

## 架构

```text
Other Agents / CLI / SDK / IDE
            │ REST / SSE / OpenAI-compatible API
            ▼
┌──────────────────────── Ragbot API ───────────────────────────┐
│ API Key → trusted principal → tenant/user/groups/roles       │
│                                                              │
│ Source/Ingest API → PostgreSQL pending job queue              │
│                                                              │
│ Query → route → retrieve ──┬─ Qdrant vector search           │
│                            ├─ PostgreSQL FTS / CJK bigrams   │
│                            └─ RRF / optional rerank          │
│          → synthesize → verify → final + citations           │
└──────────────────────────┬───────────────────────────────────┘
                           │
                    PostgreSQL / Qdrant
                           ▲
                           │ claim + lease + heartbeat
┌────────────────── Ragbot Ingestion Worker ───────────────────┐
│ local/PDF/Web/Git → chunk → dedup → shared Embedder         │
│                              ├─ PostgreSQL metadata/chunks    │
│                              └─ Qdrant vectors                │
└──────────────────────────────────────────────────────────────┘
```

三个关键不变量：

1. **摄取与查询必须使用同一个 embedding 模型和向量维度。** API 返回错误维度时 Ragbot 直接失败，不会静默截断或补零。修改 `EMBEDDING_MODEL` / `QDRANT_DIM` 后应创建兼容 collection 并重新索引。
2. **生产身份不能由请求自行扩大。** `RAGBOT_API_KEY_PRINCIPALS` 将 API key 绑定到 tenant/user/groups/roles。
3. **生产摄取不能依赖 API 进程生命周期。** `RAGBOT_ENV=production` 禁止 inline ingestion；任务必须经 PostgreSQL durable queue 由独立 worker 执行。

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

开发环境默认 `RAGBOT_ENV=development`，允许 InMemoryRepo / InMemoryQdrant / HashEmbedder 和 inline ingestion 作为快速验证回退。这些模式不等价于生产部署。

### 2. Docker Compose

```bash
cp .env.example .env
mkdir -p data
# 编辑 .env，配置 LLM / embedding credentials
docker compose up -d --build
```

Compose 默认启动：

- API
- independent ingestion worker
- migration service
- PostgreSQL
- Qdrant

可选服务：

```bash
docker compose --profile ollama up -d
docker compose --profile observability up -d
```

宿主机 `./data` 默认只读挂载到 API 和 worker 的 `/data`，所以 `local_fs` / local PDF / local Git Source 使用相同容器路径。

### 3. 建立知识库

创建 Source：

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

触发摄取：

```bash
curl -X POST http://localhost:8000/ingest/jobs \
  -H "Content-Type: application/json" \
  -d '{"source_id":"<SOURCE_ID>","tenant_id":"demo"}'
```

PostgreSQL 部署中该请求只负责持久化 `pending` job；worker 使用 `FOR UPDATE SKIP LOCKED` claim，并通过 lease/heartbeat 保证异常退出后任务可恢复。

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

开发模式可以不启用 API key；生产模式必须启用 scoped principal。

## Production mode

设置：

```bash
RAGBOT_ENV=production
RAGBOT_INGESTION_MODE=worker
```

Ragbot 会 fail fast，至少要求：

- `POSTGRES_DSN`
- `QDRANT_URL`
- `EMBEDDING_MODEL`
- `EMBEDDING_API_KEY` 或 `OPENAI_API_KEY`
- `RAGBOT_API_KEYS`
- `RAGBOT_API_KEY_PRINCIPALS`
- durable ingestion worker

生产模式不会静默使用 `InMemoryRepo`、`InMemoryQdrant`、`HashEmbedder` 或 API-process inline ingestion。

### API-key principal

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

请求中的 `tenant_id/user_id` 必须与 API-key principal 一致。Source/Ingest CRUD、`/search`、`/chat` 和 OpenAI-compatible adapter 使用同一授权边界；全局 metrics/cost/cache 在 principal 模式下要求 `admin=true`。

这是一套 service-to-service v1 身份模型；OIDC/OAuth2/SAML/企业目录集成属于后续 IAM 演进。

## Source 安全边界

生产环境：

- Web / remote PDF / remote Git 默认拒绝 loopback、private、link-local、reserved 等目的地址；
- redirect 每跳重新验证；
- 可用 `RAGBOT_WEB_ALLOWED_HOSTS`、`RAGBOT_PDF_ALLOWED_HOSTS`、`RAGBOT_GIT_ALLOWED_HOSTS` 建立 hostname allowlist；
- `local_fs`、local PDF、local Git 必须位于 `RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS`；
- Web/PDF 下载有 hard size limit；
- remote Git production source 使用 HTTPS。

高安全部署仍应在 VPC/firewall/service-mesh 层实施 egress policy；应用层 URL 校验不是网络隔离的替代品。

## Embedding 与混合检索

典型配置：

```bash
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=                 # 空时使用 OPENAI_API_KEY
EMBEDDING_BASE_URL=https://api.openai.com
QDRANT_DIM=1536
QDRANT_COLLECTION=rag_chunks
```

Hybrid retrieval 使用 Qdrant vector search + PostgreSQL FTS + RRF。启用 optional reranker 后，如果 provider 临时不可用，会回退到 RRF 而不是让检索整体失败。

### 中文 lexical retrieval

英文/数字仍由 PostgreSQL `simple` FTS 处理；连续 CJK 文本会额外生成 overlapping bigram lexemes，例如：

```text
伺服驱动器 → 伺服 / 服驱 / 驱动 / 动器
```

`eval/cjk_retrieval.py` 会直接对生产 PostgreSQL FTS 路径计算 Recall@5 和 MRR。当前固定 regression corpus 的 CI 结果为 **Recall@5=1.000、MRR=1.000**，release floor 分别为 0.90 / 0.80。

这只是防回归基线。是否引入 PGroonga / pg_jieba / external lexical index，应在更大的真实企业中文 corpus 上比较 Recall/MRR/NDCG/latency 后决定。

## Durable worker 配置

主要参数：

```bash
RAGBOT_INGESTION_MODE=auto        # development; production 使用 worker
RAGBOT_WORKER_POLL_SECONDS=1
RAGBOT_WORKER_LEASE_SECONDS=120
RAGBOT_WORKER_MAX_ATTEMPTS=3
```

Worker 入口：

```bash
python -m services.worker.main
```

任务状态和 lease 都保存在 PostgreSQL，因此 API restart 不会丢失 `pending` job；running worker 崩溃后，lease 到期可由其他 worker reclaim。

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/chat` | POST | Agentic RAG，支持 SSE |
| `/search` | POST | pure retrieval，适合作为其他 Agent 的 knowledge tool |
| `/v1/chat/completions` | POST | OpenAI-compatible adapter |
| `/sources` | CRUD | Source 管理 |
| `/ingest/jobs` | POST/GET | durable job enqueue / query / retry |
| `/admin/health` | GET | liveness |
| `/admin/ready` | GET | dependency readiness |
| `/admin/metrics` | GET | 质量/运行指标 |
| `/admin/cache` | GET | 缓存统计 |

HTTP OpenAPI 的 source of truth 是 FastAPI `/openapi.json`。共享非 HTTP contract 位于 `contracts/`。详细接口见 [`docs/API.md`](docs/API.md)。

## 部署

- **Docker Compose**：API + worker + migration + PostgreSQL + Qdrant；可选 Ollama/Jaeger。
- **Helm**：API/worker Deployments、migration initContainer、readiness/liveness、Ingress、HPA、source volume mounts；production render 强制 durable worker 和 shared PostgreSQL/Qdrant。
- **Staging Smoke**：`.github/workflows/staging-smoke.yml` 使用真实 provider credential，执行四类 Source、hybrid `/search`、Agent `/chat` 和 ACL negative isolation。

完整生产配置与升级流程：

- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)
- [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md)

## 测试与 CI

```bash
python -m pip check
python -m compileall -q contracts services cli eval
python -m pytest -q
```

GitHub Actions 覆盖：

- Python 3.10 / 3.12；
- PostgreSQL migration + durable queue recovery + FTS integration；
- CJK Recall@5/MRR regression benchmark；
- Node SDK TypeScript typecheck；
- 两套 Docker Compose config；
- Helm lint/default/production render；
- production/security regression tests。

PR #5 首轮验证：Python 3.12 **255 passed / 4 skipped**；PostgreSQL integration **4 passed**；CJK benchmark **Recall@5=1.000 / MRR=1.000**。正式 release 仍必须以 exact release commit 对应的 CI 和 staging 结果为准。

## v1.0 Release 状态

代码侧 release blockers 已进一步收敛：License、durable ingestion 和 CJK lexical baseline 已实现。当前仍 **不会提前把 `0.5.0` 改成 `1.0.0`**。

发布前必须至少：

1. 在 GitHub `staging` environment 配置真实 provider credential；
2. 成功运行 `Staging Smoke` workflow；
3. 完成生产 backup/restore、egress/TLS/secrets 等运维 gate；
4. 再创建仅包含版本/metadata/changelog 的 `release/v1.0.0` PR；
5. 从该 exact CI-validated commit 创建 tag 和 GitHub Release。

完整 checklist：[`docs/V1_RELEASE_READINESS.md`](docs/V1_RELEASE_READINESS.md)。变更历史：[`CHANGELOG.md`](CHANGELOG.md)。

## License

Ragbot 使用 **Apache License 2.0**。见 [`LICENSE`](LICENSE)。
