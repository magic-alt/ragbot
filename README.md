# Ragbot

> 面向本地与企业知识库的 Agentic RAG 服务：把 PDF、网页、Git、本地目录和 SaaS 数据源快速变成可检索、可引用、可供 Agent 使用的知识库。

Ragbot 提供完整的 ingestion、hybrid retrieval、Agentic RAG、多租户 ACL、Admin UI、durable worker、PostgreSQL/Qdrant 持久化与生产部署能力，同时保留一个无需 Docker 的轻量本地开发路径。

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

Windows PowerShell 可以直接运行：

```powershell
python .\scripts\ragbot.py up --mode auto
```

也可以使用包装器：

```powershell
.\scripts\ragbot.ps1 up --mode auto
```

Linux/macOS：

```bash
bash scripts/ragbot.sh up --mode auto
```

完整部署与 CLI 操作手册：[`docs/CLI_DEPLOYMENT.md`](docs/CLI_DEPLOYMENT.md)。

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

导入文本目录：

```bash
python scripts/ragbot.py ingest data/manuals \
  --tenant engineering \
  --tag manuals
```

导入 PDF：

```bash
python scripts/ragbot.py ingest data/pdf/product_manual.pdf \
  --tenant engineering \
  --type pdf
```

检索：

```bash
python scripts/ragbot.py search "How does ingestion recovery work?" \
  --tenant engineering \
  --top-k 5
```

Agentic RAG：

```bash
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

Bootstrap helper 会：

1. 创建 `.env`、`data/`、`.venv/`；
2. 检查 Python 版本；
3. 如果 `.venv` 缺少 pip，自动执行 `ensurepip` 修复；
4. 安装 Ragbot 及所需 extras；
5. 启动服务；
6. 等待 `/admin/ready` 通过；
7. 记录本地 PID / runtime state；
8. 为 Docker 本地 Source 自动把 `./data/...` 转换成 `/data/...`。

因此不需要手工激活虚拟环境，也不需要直接调用 `uvicorn`。

## Semantic Embedding / LLM

首次启动会从 `.env.example` 创建 `.env`。

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

development 环境在 embedding key 缺失时允许 HashEmbedder fallback。它只适合 pipeline smoke，不应拿来评估中文或真实语义检索质量。

## 核心能力

- **快速建库**：`ingest` 创建/复用 Source、提交 ingestion 并等待知识可用。
- **多源摄取**：PDF / Web / Git / local filesystem / S3-MinIO / Google Drive / Notion / Confluence。
- **可恢复摄取**：PostgreSQL durable queue、worker claim、lease、heartbeat、retry/backoff、DLQ、reconciliation。
- **周期同步**：Source-level scheduled sync，多 worker deterministic Job ID + atomic insert-if-absent。
- **SaaS 增量复用**：Drive / Notion / Confluence metadata-first refresh，未变化内容跳过正文下载和 embedding。
- **混合检索**：Qdrant vector + PostgreSQL FTS/CJK bigram + RRF，可选 reranker。
- **Agentic RAG**：route → retrieve/sql/code/web → synthesize → verify → finalize，输出 citation。
- **多租户与 ACL**：API-key principal 绑定 tenant/user/groups/roles。
- **RBAC**：reader / operator / owner / global admin。
- **管理控制面**：内置 `/admin/ui`、Source Catalog、Job progress、Retry/Requeue、scheduled sync、queue health。
- **生产部署**：Docker Compose、Helm、KEDA、health/readiness、metrics/tracing、backup/restore。
- **AI 工具链兼容**：`/search`、`/chat`、OpenAI-compatible `/v1/chat/completions`、CLI、Node SDK。

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
│ connector → parse/chunk/dedup/embed                           │
│ → PostgreSQL metadata/FTS + Qdrant vectors                    │
│ → durable retry/backoff → DLQ                                 │
│ scheduler + queue reconciliation                              │
└───────────────────────────────────────────────────────────────┘
```

## 原生 `rag` CLI

Bootstrap helper 负责安装、启动和常见运维；正式产品 CLI 仍然可以直接使用：

```bash
rag --server http://localhost:8000 doctor

rag --server http://localhost:8000 \
  --tenant engineering \
  ingest /data/manuals \
  --wait

rag --server http://localhost:8000 \
  --tenant engineering \
  search "query" \
  --top-k 5

rag --server http://localhost:8000 \
  --tenant engineering \
  ask "question"
```

支持的主要命令：

```text
rag ingest
rag import
rag doctor
rag search
rag ask
rag patch
```

## Local Source 注意事项

`local_fs` 当前默认扫描：

```text
.txt  .md  .markdown  .rst  .csv  .log
```

PDF 使用独立 `pdf` connector。扫描型 PDF 需要先 OCR。

Docker 模式建议所有本地 Source 放在 `./data`；Compose 中对应 `/data`。使用 `scripts/ragbot.py ingest data/...` 时 helper 会自动处理路径转换。

## Production 不变量

生产模式至少要求：

- `RAGBOT_ENV=production`；
- durable worker；
- PostgreSQL；
- Qdrant；
- semantic embedding；
- scoped API keys / principal mappings；
- TLS / ingress / rate limit / egress policy；
- PostgreSQL + Qdrant backup / restore；
- real-provider staging smoke。

Production 不会静默降级到 InMemory storage、HashEmbedder 或 inline ingestion。

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
