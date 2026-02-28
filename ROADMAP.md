# Ragbot — 商业级 Agentic RAG 平台

> 版本 0.5.0 | Milestone A–D 已完成 | 167 项测试全部通过

---

## 1. Product Vision

### 1.1 定位

Ragbot 是一个商业级 Agentic RAG 平台，支持多种 API 接口，提供类似 Cursor 的编程/问答体验：对本地文档、数据库、代码仓库进行检索、总结、引用与多步工具调用；LLM 同时支持云端模型与本地大模型（可插拔）。

### 1.2 用户价值

- **Knowledge Worker**：跨文档/数据库的可追溯总结 + 行动建议
- **Developer（Cursor-like）**：代码仓库 + 本地文档 + 上下文编程助手
- **Enterprise/Team**：多租户、权限隔离、审计、可观测、可评测、可运维

### 1.3 核心原则

- **Evidence-first**：回答必须能引用证据（chunk/row/code/url）并可追溯
- **Secure-by-default**：ACL 前置过滤、密钥脱敏、最小权限、只读 SQL
- **Composable**：工具/模型/连接器可插拔（Python 主实现 + Node 工具代理）
- **Observable & Evaluable**：每次回答可追踪、可回放、可评测、可回归

---

## 2. 当前架构

### 2.1 系统总览

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                         │
│   CLI (rag ask/search/patch/ingest)  ·  Node SDK  ·  IDE   │
└────────────────┬────────────────────────────────────────────┘
                 │  REST + SSE / OpenAI compat
┌────────────────▼────────────────────────────────────────────┐
│                   API Gateway (FastAPI)                      │
│  /chat  /search  /v1/chat/completions  /sources  /ingest    │
│  /admin/health  /admin/metrics  /admin/feedback  /admin/cost│
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Agent Pipeline (graph.py)               │    │
│  │  route → retrieve/sql/code/web → synthesize         │    │
│  │       → verify → [loop or finalize]                 │    │
│  └─────────────────────────────────────────────────────┘    │
│  Middleware: CORS · RequestLogging · API Key Auth            │
│  Observability: RequestTracer · MetricsCollector             │
│  Cache: LRU(TTL) · RetrievalCache · EmbeddingCache          │
└──────┬──────────┬──────────┬───────────┬────────────────────┘
       │          │          │           │
┌──────▼───┐ ┌───▼────┐ ┌──▼────┐ ┌────▼─────┐
│ Qdrant   │ │Postgres│ │ LLM   │ │ Worker   │
│ (vector) │ │(FTS+   │ │OpenAI │ │connectors│
│          │ │ meta)  │ │Ollama │ │PDF/Git/  │
│          │ │        │ │       │ │Web/FS    │
└──────────┘ └────────┘ └───────┘ └──────────┘
```

### 2.2 工程目录

```
ragbot/
├── contracts/                    # 跨语言共享契约
│   ├── types.py                  #   Python 类型（Citation, EvidenceItem, AgentState...）
│   ├── types.ts                  #   TypeScript 镜像
│   ├── openapi.yaml              #   OpenAPI 3.1 规范（含 SSE 示例）
│   └── tools.schema.json         #   工具 JSON Schema
├── services/
│   ├── api/app/                  # API 网关
│   │   ├── agent/                #   Agent 状态机
│   │   │   ├── graph.py          #     迭代循环 + 工具调度
│   │   │   ├── state.py          #     状态定义 + build_initial_state()
│   │   │   ├── callbacks.py      #     SSE 事件回调（QueueCallback）
│   │   │   ├── context.py        #     IDE 上下文注入 + 证据去重/压缩
│   │   │   ├── reliability.py    #     timeout / retry / circuit breaker
│   │   │   ├── session.py        #     会话存储
│   │   │   └── nodes/            #     Agent 节点
│   │   │       ├── route.py      #       LLM/关键词路由
│   │   │       ├── retrieve.py   #       文档检索
│   │   │       ├── sql.py        #       SQL 查询（NL2SQL）
│   │   │       ├── code.py       #       代码搜索 + open_file + apply_patch + explain_error
│   │   │       ├── web.py        #       Web 搜索
│   │   │       ├── synthesize.py #       证据合成草稿
│   │   │       ├── verify.py     #       证据充分性验证
│   │   │       └── finalize.py   #       最终回答输出
│   │   ├── routes/               #   FastAPI 路由
│   │   │   ├── chat.py           #     /chat 端点
│   │   │   ├── search.py         #     /search 纯检索
│   │   │   ├── sources.py        #     /sources CRUD
│   │   │   ├── ingest.py         #     /ingest/jobs 任务管理
│   │   │   ├── admin.py          #     /admin/* 运维端点
│   │   │   └── openai_compat.py  #     /v1/chat/completions
│   │   ├── retrieval/            #   混合检索
│   │   │   ├── qdrant.py         #     向量检索（InMemory + QdrantClient）
│   │   │   ├── pg_fts.py         #     全文检索（倒排索引 + PG tsvector）
│   │   │   ├── service.py        #     Retriever 编排 + RRF 融合
│   │   │   └── rerank.py         #     RRF 排序
│   │   ├── storage/              #   数据存储
│   │   │   ├── models.py         #     Document, Chunk, Source, Policy, Job, TableData
│   │   │   └── repo.py           #     InMemoryRepo（线程安全）
│   │   ├── llm/                  #   LLM 后端
│   │   │   ├── provider.py       #     ModelProvider Protocol
│   │   │   ├── client.py         #     OpenAI 客户端
│   │   │   ├── ollama.py         #     Ollama 适配器
│   │   │   └── router.py         #     ModelRouter(fast/strong) + CostTracker
│   │   ├── auth/                 #   访问控制
│   │   │   ├── acl.py            #     ACL 计算（user/group/role）
│   │   │   └── policy.py         #     策略哈希
│   │   ├── observability/        #   可观测性
│   │   │   ├── tracing.py        #     RequestTracer + Span
│   │   │   └── metrics.py        #     MetricsCollector + AggregateMetrics
│   │   ├── cache/                #   缓存
│   │   │   └── cache.py          #     LRUCache + RetrievalCache + EmbeddingCache
│   │   ├── api.py                #   FastAPI 应用入口
│   │   ├── main.py               #   chat() 编排函数
│   │   ├── factory.py            #   环境构建工厂
│   │   ├── middleware.py         #   CORS + 请求日志
│   │   └── logging_config.py    #   结构化日志配置
│   └── worker/                   # 摄取 Worker
│       ├── pipeline.py           #   Source → Job → connector → chunk → dedup → embed
│       ├── queue.py              #   任务队列（InProcess / Celery 预留）
│       ├── connectors/           #   数据源连接器
│       │   ├── local_fs.py       #     目录遍历
│       │   ├── pdf.py            #     PDF 提取（PyPDF2）
│       │   ├── web.py            #     网页抓取（requests + BeautifulSoup）
│       │   └── git.py            #     Git 仓库（GitPython）
│       ├── jobs/                 #   摄取任务
│       │   ├── ingest_text.py    #     TXT/MD 切分
│       │   ├── ingest_pdf.py     #     PDF 切分（滑动窗口 800/100）
│       │   ├── ingest_repo.py    #     代码切分（Python AST / regex / 行回退）
│       │   ├── ingest_web.py     #     网页切分
│       │   └── embed_and_upsert.py #   批量嵌入 + Qdrant upsert
│       └── dedup/                #   去重
│           ├── hashing.py        #     内容哈希
│           └── versioning.py     #     版本管理
├── cli/                          # CLI 客户端
│   └── rag.py                    #   rag ask / search / patch / ingest
├── eval/                         # 评测框架
│   ├── datasets.py               #   EvalCase + 数据集管理
│   ├── runner.py                 #   自动回归 + 失败分析
│   └── ragas/evaluate.py         #   RAGAS 评测
├── packages/node-client/         # Node SDK
│   └── src/
│       ├── client.ts             #   chat() + chatStream() SSE
│       ├── tools.ts              #   工具类型定义
│       └── index.ts              #   导出
├── infra/
│   ├── docker/
│   │   ├── Dockerfile            #   生产镜像
│   │   └── docker-compose.yml    #   API + PG + Qdrant + Ollama + Jaeger
│   ├── migrations/
│   │   ├── 001_init.sql          #   documents / chunks / acl_policies / ingestion_jobs
│   │   ├── 002_sources.sql       #   sources 表
│   │   └── 003_observability.sql #   feedback / audit_log / request_metrics
│   ├── qdrant/
│   │   └── init_collection.sh    #   Collection + payload 索引创建
│   └── helm/ragbot/              #   Kubernetes 部署
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── deployment.yaml
│           └── service.yaml
└── tests/
    └── test_agent.py             # 167 项单元/集成测试
```

### 2.3 Agent 状态机

```
Request
  │
  ▼
┌─────────┐
│ Route   │  LLM 意图分类（可回退关键词匹配）
│         │  → doc_rag / sql / code / mixed / web_fallback
└────┬────┘
     │
     ▼  ╔══════════════════════════════════╗
  ┌──▶  ║  Action Loop (max 3 iterations) ║
  │     ╚══════════════════════════════════╝
  │         │
  │     ┌───▼───────────────────────────┐
  │     │ retrieve / sql / code / web   │  工具调用（带 timeout/retry/breaker）
  │     │ open_file / apply_patch /     │
  │     │ explain_error                 │
  │     └───┬───────────────────────────┘
  │         │
  │     ┌───▼────────┐
  │     │ Synthesize │  基于证据生成草稿（LLM 或模板回退）
  │     └───┬────────┘
  │         │
  │     ┌───▼─────┐
  │     │ Verify  │  证据充分性检查
  │     └───┬─────┘
  │         │
  │    enough_evidence?
  │    ├── No  → 生成 next_query → 回到 Action Loop
  │    └── Yes ↓
  │
  │     ┌─────────┐
  └─    │Finalize │  confidence = high/medium/low + citations
        └─────────┘
```

### 2.4 数据模型

**Postgres 核心表：**

| 表 | 主要字段 | 用途 |
|---|---|---|
| `documents` | doc_id, tenant_id, source_type, title, uri, version, tags, acl_policy_id | 文档元数据 |
| `chunks` | chunk_id, doc_id, tenant_id, chunk_index, text, checksum, tsv(GIN) | 片段 + FTS |
| `acl_policies` | acl_policy_id, tenant_id, rules(jsonb), policy_hash | 权限策略 |
| `sources` | source_id, tenant_id, source_type, name, config, status | 数据源管理 |
| `ingestion_jobs` | job_id, tenant_id, source_id, status, chunk_count, error | 摄取任务 |
| `feedback` | id, request_id, feedback, created_at | 用户反馈 |
| `audit_log` | id, request_id, tenant_id, user_id, action, detail | 审计日志 |
| `request_metrics` | id, request_id, tenant_id, confidence, duration_ms, tool_calls | 请求指标 |

**Qdrant payload（向量检索过滤）：**

`tenant_id`, `source_type`, `doc_id`, `chunk_index`, `path`, `url`, `page`, `section`, `language`, `ingested_at`, `doc_updated_at`, `version`, `checksum`, `acl_hash`, `tags[]`, `embedding_model`

### 2.5 API 端点

| 端点 | 方法 | 功能 |
|---|---|---|
| `/chat` | POST | 主 Agentic RAG（JSON + SSE 流式） |
| `/search` | POST | 纯检索（不触发 Agent） |
| `/v1/chat/completions` | POST | OpenAI 兼容层 |
| `/sources` | CRUD | 数据源管理 |
| `/ingest/jobs` | POST/GET | 摄取任务触发/查询 |
| `/admin/health` | GET | 健康检查 |
| `/admin/metrics` | GET | 聚合质量指标 |
| `/admin/metrics/history` | GET | 请求历史 |
| `/admin/feedback` | POST | 用户反馈录入 |
| `/admin/cost` | GET | LLM 成本追踪 |
| `/admin/cache` | GET | 缓存统计 |

### 2.6 环境变量

```bash
# LLM
RAGBOT_LLM_PROVIDER=openai|ollama    # 默认 openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b

# 向量存储
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_chunks
QDRANT_DIM=1536

# 数据库
POSTGRES_DSN=postgresql://user:pass@localhost:5432/ragbot
POSTGRES_ALLOWED_SCHEMAS=public

# 安全
RAGBOT_API_KEYS=key1,key2            # 空值允许所有请求

# 可观测
RAGBOT_TRACING_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# 缓存 & 路由
RAGBOT_CACHE_ENABLED=true
RAGBOT_CACHE_TTL_SECONDS=300
RAGBOT_MODEL_ROUTING=false

# 中间件
RAGBOT_CORS_ORIGINS=http://localhost:3000
LOG_LEVEL=INFO
LOG_FORMAT=json
```

---

## 3. 已完成里程碑摘要

### Milestone A — 可用内测版 (v0.2.0)

- ModelProvider Protocol（OpenAI + Ollama 可切换）
- 工具可靠性（timeout / retry / circuit breaker，6 个 node 全接入）
- SSE 真实流式（QueueCallback 线程安全队列）
- OpenAI 兼容层 `/v1/chat/completions`
- 纯检索端点 `/search` + CORS 中间件 + Request-ID 传播
- Docker + docker-compose + SQL 迁移 + Qdrant 初始化
- 51 项测试全部通过

### Milestone B — 企业级数据接入 (v0.3.0)

- Source CRUD API + Ingestion Pipeline 编排器
- 连接器实现：Local FS / PDF / Web / Git
- DB Schema Introspection（InMemory + PostgreSQL）
- ACL 增强（user / group / role 三级权限）
- 文档版本化 + chunk 去重（checksum）
- 86 项测试全部通过

### Milestone C — Cursor-like 编程助手 (v0.4.0)

- 符号级 Repo Ingestion（Python AST + regex C-like + 行回退）
- 编程工具集：open_file / apply_patch / explain_error
- CLI 客户端：`rag ask` / `search` / `patch` / `ingest`
- IDE 上下文注入：selected_text / open_files / git_diff / recent_errors
- Evidence 去重 + 压缩（MD5 哈希 + 分数排序 + 预算控制）
- 129 项测试全部通过

### Milestone D — 商业级可运维 (v0.5.0)

- 全链路追踪：RequestTracer + context-manager Span（OpenTelemetry 兼容）
- 质量指标：citation_coverage / retrieval_hit_rate / tool_failure_rate / user_feedback
- 评测回归：EvalCase 数据集 + run_eval_suite() + 失败分类（5 类）
- 模型路由：ModelRouter fast/strong 分级 + CostTracker
- 多级缓存：LRU (TTL+LRU 双淘汰) + RetrievalCache + EmbeddingCache
- Helm Chart（rolling update + health probes + K8s secrets）
- 167 项测试全部通过

---

## 4. 代码审查与修复记录

> 全量源码审查于 2026-02-27 完成，发现 32 项优化建议，已全部修复。

### P0 修复（6 项 — 安全/正确性）

| 问题 | 修复 |
|------|------|
| tenant_id/user_id 无认证 | API Key 认证中间件（`RAGBOT_API_KEYS`） |
| sql_node 直接执行用户原文 | NL2SQL：`_resolve_sql()` + LLM 转换 + 回退 |
| code_search 正则注入 | `re.escape()` + `re.IGNORECASE` |
| CodeSearch 可读取敏感文件 | 白名单后缀 + 排除 .git/.env + `resolve()` 防遍历 |
| API Key 泄露风险 | `_sanitize_error()` 异常脱敏 |
| pg_fts 缺少 source_types 过滤 | 添加过滤 + 时间比较统一为 `_to_epoch()` |

### P1 修复（9 项 — 功能/性能）

| 问题 | 修复 |
|------|------|
| 同步阻塞 LLM 调用 | `async def` + `asyncio.to_thread()` |
| SSE 假流式 | QueueCallback 真实时推送 |
| verify next_query 未生效 | `_next_step()` 中更新 state.query |
| FTS 全表扫描 | `InvertedIndex` 倒排索引 O(K) |
| embed_text 伪嵌入 | `get_embed_fn()` 工厂 + 真实 API 回退 |
| 路由不使用 LLM | `_llm_route()` + 关键词回退 |
| web_node 空结果误报 | LLM 不可用时 `ok=False` |
| LLM 异常静默 | `logger.warning()` 记录 |
| types.py 与 state.py 重复 | state.py 从 contracts.types 导入 |

### P2/P3 修复（17 项 — 质量/长期改进）

包括：AgentServices Protocol 类型化、Citation `__hash__`/`__eq__`、payload_map O(1) 查找、pyproject.toml 包管理、InMemoryRepo threading.Lock、embed_and_upsert 批量、Worker 真实连接器实现、任务队列 Protocol、会话存储、Node SSE 支持、日志框架、infra 补齐、RAGAS 评测、versioning 健壮化等。

---

## 5. 下一步规划（Milestone E+）

### Milestone E：生产级检索质量 + 异步化

> 目标：让检索质量达到可度量的生产水平，消除同步瓶颈。

**E1 — 真实 Embedding 集成**
- 接入 OpenAI `text-embedding-3-small/large`（已有 `get_embed_fn()` 工厂）
- 本地 embedding：sentence-transformers / BGE / E5（通过 Ollama 或独立服务）
- Embedding 模型升级策略：增量重建 + `embedding_model` payload 字段版本区分

**E2 — Cross-Encoder Rerank**
- 接入 Cohere rerank / BGE-reranker / ms-marco-MiniLM
- `RetrievalService` 在 RRF 融合后增加 rerank 阶段
- 可配置开关 `RAGBOT_RERANK_ENABLED` + `RAGBOT_RERANK_MODEL`

**E3 — 异步化改造**
- `OpenAIClient` → `httpx.AsyncClient`
- Agent 核心路径改为 async（`run_agent` → `async run_agent`）
- mixed 路由下 `asyncio.gather` 并发多工具（retrieve + sql + code）
- WebSocket 端点替代 SSE（可选）

**E4 — 检索质量评测基线**
- 基于 eval/runner.py 建立回归基线：至少 200 条（doc_qa 100 + db_qa 50 + code_task 50）
- CI 门禁：pass_rate < 阈值 → 阻断合并
- 检索命中率（MRR@10 / Recall@10）纳入自动评测

**E5 — PostgreSQL Repo 实现**
- `PostgresRepo` 替代 `InMemoryRepo`（生产必须）
- 连接池：`psycopg_pool.AsyncConnectionPool`
- migration 工具（alembic 或手动 .sql 管理）

### Milestone F：多数据源扩展 + 安全加固

> 目标：覆盖企业常见数据源，安全性达到生产标准。

**F1 — 邮件连接器**
- Gmail（IMAP / Google API）
- Outlook（Microsoft Graph API）
- 权限模型：按账户隔离

**F2 — Office 文档连接器**
- DOCX（python-docx）、PPTX（python-pptx）、XLSX（openpyxl）
- 表格数据 → 自动注册为 TableData（供 SQL 工具查询）

**F3 — JWT 认证**
- 替代当前 API Key 认证
- 从 JWT claims 提取 tenant_id / user_id / groups / roles
- Rate limiting（令牌桶 / 滑动窗口）

**F4 — Secrets Redaction**
- 日志/traces 中自动脱敏 API Key / DSN / PII
- SQL 结果中的敏感列标记与遮蔽

**F5 — 多数据库支持**
- MySQL adapter（SQLAlchemy backend）
- 通用 JDBC/SQLAlchemy 接口

### Milestone G：IDE 集成 + 前端

> 目标：端到端用户体验闭环。

**G1 — VSCode 插件**
- 基于 Node SDK + SSE 流式
- 侧边栏：问答 + 引用跳转
- 编辑器内：选中代码 → 解释/重构/生成测试
- apply_patch：直接在编辑器中显示 diff 预览

**G2 — Web 前端**
- 轻量 Chat UI（React / Vue）
- 对话历史 + 引用展示 + 反馈按钮
- Admin Dashboard：指标可视化 + 数据源管理

**G3 — 多轮对话**
- 基于 `session.py` 的 InMemorySessionStore → PostgresSessionStore
- 上下文窗口管理：历史 turn 摘要 + 最近 N 轮原文
- 指代消解（LLM 或规则）

### 架构演进方向

**短期（当前 → Milestone E）：**
- 保持 monolith 单进程架构（FastAPI + Worker 同进程）
- InMemoryRepo → PostgresRepo 切换为强一致存储
- 异步化消除 LLM 调用阻塞

**中期（Milestone F–G）：**
- Worker 独立进程：Celery / Dramatiq 任务队列
- 检索服务拆分：独立 retrieval-service 微服务
- API Gateway：统一认证 + 限流 + 路由

**长期：**
- 多区域部署（数据驻留合规）
- Embedding 模型在线热切换 + 增量重建
- A/B 测试框架（路由策略 / prompt 变体）
- Plugin 系统：第三方工具注册 + 沙箱执行

---

## 6. 工具 Schema（跨语言契约）

所有工具入参/出参定义于 `contracts/tools.schema.json`，Python/Node 共用。

| 工具 | 入参 | 出参 |
|------|------|------|
| `retrieve` | query, top_k, filters(tenant_id, source_types, tags, time_range, security_scope) | chunks[{chunk_id, doc_id, text, score, citations, metadata}] |
| `sql_query` | dialect, query, params, limit, timeout_ms | rows[], columns[{name,type}], stats{row_count, elapsed_ms} |
| `code_search` | query, repo, ref, path_glob, max_hits | snippets[{path, ref, line_start, line_end, content}] |
| `open_file` | path, repo, start_line, end_line | content(带行号) |
| `apply_patch` | path, original, replacement | PatchResult{path, diff, original_lines, modified_lines} |
| `explain_error` | error_text, repo | locations[{path, line, context}] |
| `web_search` | query, recency_days, domains | snippets[] |

> `security_scope` 由服务端计算注入，客户端不可伪造。

---

## 7. 技术决策与风险

| 决策项 | 当前选择 | 风险 / 备注 |
|--------|----------|-------------|
| Agent 同步执行 | `asyncio.to_thread` 卸载 | Milestone E 需全面异步化 |
| 存储 | InMemoryRepo (dev) / Postgres (prod) | PostgresRepo 尚未实现生产版 |
| Embedding | hash-based 伪嵌入 (dev) / OpenAI API (prod) | 本地 embedding 待接入 |
| 认证 | API Key | 生产需 JWT + RBAC |
| 任务队列 | InProcessQueue（同步） | 生产需 Celery / Dramatiq |
| SQL 安全 | READ ONLY 事务 + 关键词黑名单 | 可增加 sqlglot 语法级校验 |
| 代码工具 | apply_patch 输出 diff，客户端落盘 | 需审计 + 回滚机制 |
| 邮件权限 | 未实现 | 需确定隔离模型 |

---

## 8. Definition of Done（商业级）

- [x] 多租户隔离 + ACL 前置过滤 + 审计日志
- [x] 文档/DB/代码三类数据源闭环（ingest → search → answer → cite）
- [x] Cursor-like：repo 代码问答 + 生成 patch（diff）
- [x] 多模型：云端 + 本地可切换、可回退
- [x] 可观测 + 可评测 + CI 回归门禁
- [x] 部署与升级策略明确（迁移/回滚/embedding 重建）
- [ ] 邮件数据源闭环
- [ ] JWT 认证 + Rate Limiting
- [ ] PostgresRepo 生产级实现
- [ ] 检索质量 MRR@10 基线达标
- [ ] 异步化 agent 路径
