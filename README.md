# Ragbot

Ragbot 是一个面向本地/企业知识库的 **Agentic RAG knowledge service**：把 PDF、网页、Git 仓库和本地文本目录摄取为可检索知识，通过 Qdrant + PostgreSQL 混合检索，为自身 Agent 或其他 Agent / 应用提供底层知识支撑。

> 当前代码已经具备 README 下述核心能力，并正在进行 v1.0 release hardening。Python package / Helm metadata 仍为 `0.5.0`；`1.0.0` 版本号、tag 与 GitHub Release 应在正式 release PR 中单独处理。详细能力验收与发布门槛见 [`docs/V1_RELEASE_READINESS.md`](docs/V1_RELEASE_READINESS.md)。

## 能力范围

- **知识摄取**：PDF / Web / Git / local filesystem → chunk → dedup → embedding → Qdrant
- **混合检索**：Qdrant vector search + PostgreSQL FTS + RRF，可选 cross-encoder rerank
- **Agentic RAG**：route → retrieve/sql/code/web → synthesize → verify → finalize
- **多租户与 ACL**：tenant/user scope、ACL hash 前置过滤、API-key principal
- **LLM**：OpenAI-compatible provider 与 Ollama adapter
- **服务接口**：`/search`、`/chat`、OpenAI-compatible `/v1/chat/completions`、Source/Ingest API
- **工程能力**：SSE、可靠性封装、评测、指标/Tracing、Docker Compose、Helm

这些不是仅有接口定义的占位能力：当前实现中存在完整 ingestion pipeline、Qdrant/PostgreSQL retrieval、Agent graph、Source/Job persistence、SSE transport、迁移机制和部署配置。能力边界、已验证路径以及仍保留的限制不在 README 中隐藏，统一记录在 v1 readiness 文档。

## 架构

```text
Other Agents / CLI / SDK / IDE
            │ REST / SSE / OpenAI-compatible API
            ▼
┌──────────────────────── Ragbot API ───────────────────────────┐
│ API Key → trusted principal → tenant/user/groups/roles       │
│                                                              │
│ Source API → Ingestion Pipeline                              │
│   local/PDF/Web/Git → chunk → dedup → shared Embedder       │
│                                      │                       │
│                                      ▼                       │
│                                 Qdrant vectors               │
│                                                              │
│ Query → route → retrieve ──┬─ Qdrant vector search          │
│                            ├─ PostgreSQL FTS                 │
│                            └─ RRF / optional rerank          │
│          → synthesize → verify → final + citations           │
└───────────────────────┬─────────────────────────────────────┘
                        │
               PostgreSQL metadata / ACL / jobs
```

两个关键不变量：

1. **摄取与查询必须使用同一个 embedding 模型和相同向量维度。** API 返回错误维度时 Ragbot 会拒绝写入/查询，不再静默截断或补零。修改 `EMBEDDING_MODEL` 或 `QDRANT_DIM` 后应创建兼容 collection 并重新索引。
2. **生产身份不能由请求自行声明。** `RAGBOT_API_KEY_PRINCIPALS` 将 API key 绑定到 tenant/user/groups/roles；HTTP payload/header 不能扩大该授权范围。

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

开发环境默认 `RAGBOT_ENV=development`，允许 InMemoryRepo / InMemoryQdrant / HashEmbedder 作为快速验证回退。这些回退不等价于生产语义知识库。

### 2. Docker Compose

```bash
cp .env.example .env
mkdir -p data
# 编辑 .env，配置 LLM / embedding credentials
docker compose up -d --build
```

默认启动 API + PostgreSQL + Qdrant；数据库 schema 由 one-shot migration service 在 API 启动前执行。附加服务：

```bash
docker compose --profile ollama up -d
docker compose --profile observability up -d
```

宿主机 `./data` 默认只读挂载到容器 `/data`，`local_fs` / local PDF / local Git Source 应使用容器可见路径。

### 3. 建立本地知识库

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

开发模式下可以不启用 API key。生产模式必须启用 scoped principal。

## Production mode

设置：

```bash
RAGBOT_ENV=production
```

后，Ragbot 会 fail fast，要求至少存在：

- `POSTGRES_DSN`
- `QDRANT_URL`
- `EMBEDDING_MODEL`
- `EMBEDDING_API_KEY` 或 `OPENAI_API_KEY`
- `RAGBOT_API_KEYS`
- `RAGBOT_API_KEY_PRINCIPALS`

生产模式不会静默使用 `InMemoryRepo`、`InMemoryQdrant` 或 `HashEmbedder`。

### API-key principal

示例：

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

请求仍可携带 `tenant_id/user_id` 作为 API contract 的一部分，但必须与 API key principal 一致。Source/Ingest CRUD、`/search`、`/chat` 和 OpenAI-compatible adapter 都执行同一授权边界。全局 metrics/cost/cache 运维接口在 principal 模式下要求 `admin=true`。

这是一套 service-to-service v1 身份模型；完整 OIDC/OAuth2/SAML/企业目录集成属于后续 IAM 演进。

## Source 安全边界

企业知识服务会读取外部 URL 和本地文件，因此 Source 本身属于安全边界。

生产环境：

- Web / remote PDF / remote Git 默认拒绝 loopback、private、link-local、reserved 等目的地址；
- redirect 每跳重新验证；
- 可用 `RAGBOT_WEB_ALLOWED_HOSTS`、`RAGBOT_PDF_ALLOWED_HOSTS`、`RAGBOT_GIT_ALLOWED_HOSTS` 建立 hostname allowlist；
- `local_fs`、local PDF、local Git 必须位于 `RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS`；
- Web/PDF 下载有 hard size limit；
- remote Git production source 使用 HTTPS。

高安全部署仍应在 VPC/firewall/service-mesh 层实施 egress policy；应用层 URL 校验不是网络隔离的替代品。

## Embedding 与检索

典型配置：

```bash
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=                 # 空时使用 OPENAI_API_KEY
EMBEDDING_BASE_URL=https://api.openai.com
QDRANT_DIM=1536
QDRANT_COLLECTION=rag_chunks
```

开发环境未配置可用 embedding API 时会使用确定性的 HashEmbedder。**生产模式明确拒绝该回退。**

Hybrid retrieval 使用 Qdrant vector search + PostgreSQL FTS + RRF。启用 optional reranker 后，如果 reranker provider 临时不可用，系统会回退到 RRF 排序而不是让检索整体失败。

PostgreSQL FTS 当前采用 `simple` text-search configuration；功能正确，但针对大规模中文语料可进一步评估 pg_jieba / PGroonga / 独立 lexical index。

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/chat` | POST | Agentic RAG，支持 SSE |
| `/search` | POST | 纯检索，适合其他 Agent 作为 knowledge tool 调用 |
| `/v1/chat/completions` | POST | OpenAI-compatible adapter |
| `/sources` | CRUD | Source 管理 |
| `/ingest/jobs` | POST/GET | 摄取任务触发、查询与 retry |
| `/admin/health` | GET | liveness |
| `/admin/ready` | GET | dependency readiness |
| `/admin/metrics` | GET | 质量/运行指标 |
| `/admin/cache` | GET | 缓存统计 |

HTTP OpenAPI 的 source of truth 是 FastAPI 运行时 `/openapi.json`。共享非 HTTP contract 位于 `contracts/`。详细接口见 [`docs/API.md`](docs/API.md)。

## 部署

- Docker Compose：本地/单机完整栈 + migration service。
- Helm：migration initContainer、readiness/liveness、Ingress、HPA、多副本 shared-store guard；支持 `extraVolumes/extraVolumeMounts` 将企业知识目录挂载到 `/data`。
- 多副本/HPA 必须使用共享 PostgreSQL + Qdrant。

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

GitHub Actions 当前覆盖：

- Python 3.10 / 3.12；
- PostgreSQL migration + native FTS integration；
- Node SDK TypeScript typecheck；
- 两套 Docker Compose config；
- Helm lint / render；
- v1 production/security regression tests。

不要把 README 中的固定测试数字作为 release 依据；应以待发布 commit 对应的 GitHub Actions 结果为准。

## v1.0 Release 状态

代码能力已进入 v1 hardening 阶段，但 **`0.5.0` 不是通过文档修改自动变成 `1.0.0`**。正式 release 前仍需完成真实 provider/staging smoke、生产安全配置、License 决策和 release-only version/tag 流程。

完整 checklist：[`docs/V1_RELEASE_READINESS.md`](docs/V1_RELEASE_READINESS.md)。变更历史：[`CHANGELOG.md`](CHANGELOG.md)。

## License

仓库当前未包含开源许可证文件。公开可见不等于自动授予开源使用许可。若 v1.0 目标是正式开源发行，**选择并加入 LICENSE 是 release gate**；许可证选择应由仓库所有者基于预期商业/开源策略决定，而不应由自动化工具代为假设。
