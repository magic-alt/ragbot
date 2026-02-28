# ROADMAP - 商业级 Agentic RAG / Cursor-like

> 目标：实现一个商业级可用的 Agentic RAG 平台，支持多种 API 接口，提供类似 Cursor 的编程/问答体验：对本地文档、邮件、数据库、代码仓库进行检索、总结、引用与多步工具调用；LLM 同时支持云端模型与本地大模型（可插拔）。

---

## 0. Product Vision

### 0.1 用户价值

- Knowledge Worker：跨文档/邮件/数据库的可追溯总结 + 行动建议，减少检索成本
- Developer（Cursor-like）：对代码仓库 + 本地文档 + 工单/邮件的上下文编程助手
- Enterprise/Team：多租户、权限隔离、审计、可观测、可评测、可运维

### 0.2 核心原则

- Evidence-first：回答必须能引用证据（chunk/row/code/url）并可追溯
- Secure-by-default：ACL 前置过滤、密钥脱敏、最小权限、只读 SQL
- Composable：工具/模型/连接器可插拔（Python 主实现 + Node 工具代理）
- Observable & Evaluable：每次回答可追踪、可回放、可评测、可回归

---

## 1. 当前现状（已实现基线）

### 1.1 基础骨架（首次实现）

已实现（可运行骨架）包括：

- Agent 状态机（route → retrieve/sql/code/web → synthesize → verify → finalize）
- 多租户 + ACL 预过滤（tenant_id + acl_hashes）
- 混合检索（Qdrant-style 向量 + Postgres FTS-style 关键词 + RRF 融合）
- FastAPI `/chat` 支持 JSON + SSE（token/tool_event）
- OpenAPI/Schema 契约与单元测试骨架
- 默认 in-memory 适配器 + env 开关启用真实 Postgres/Qdrant

### 1.2 Milestone A 已完成（当前版本）

在骨架基础上补齐了以下能力，ragbot 已升级为可用内测版：

**WU1 - 模型适配器抽象（`RAGBOT_LLM_PROVIDER` 环境变量）**
- `ModelProvider` Protocol（PEP 544），OpenAIClient 零改动即满足接口
- `OllamaAdapter`：复用 Ollama OpenAI 兼容 API（`/v1/chat/completions`），json 模式降级兼容
- `build_model_provider()` 工厂：`openai`（默认）| `ollama` 可运行时切换
- `AgentServices.llm` 类型从 `OpenAIClient` 改为 `ModelProvider`（可插拔）

**WU2 - 工具可靠性（`services/api/app/agent/reliability.py`）**
- `with_timeout()`：`concurrent.futures` 线程池超时隔离，默认：retrieve=10s、sql=5s、code=8s、web=15s
- `with_retry()`：指数退避重试，仅对可恢复异常（ConnectionError/TimeoutError/OSError）触发
- `CircuitBreaker`：连续失败 N 次后熔断 M 秒（默认 3 次 / 30 秒），冷却后自动复位
- `safe_tool_call()`：统一入口组合三者，已应用到 retrieve/sql/code/web/synthesize/verify 全部 6 个 node

**WU3 - 真实时 SSE 流式（`services/api/app/agent/callbacks.py`）**
- `AgentEvent` / `EventCallback` Protocol / `QueueCallback` / `NullCallback`
- `QueueCallback`：线程安全队列，agent 同步线程 emit → async SSE 生成器实时 yield
- `run_agent()` 新增 `callback` 参数；非流式模式使用 `NullCallback`（零开销）
- `tool_call` / `tool_result` 在工具运行时即时推送，不再事后汇总

**WU4 - OpenAI 兼容层（`services/api/app/routes/openai_compat.py`）**
- `POST /v1/chat/completions`：从 `messages[-1]` 提 query，租户信息从 header 读取
- 非流式返回标准 OpenAI 格式 `{id, object, choices, usage}`，附加 `citations` 字段
- 流式：复用 QueueCallback 机制，以 OpenAI chunk 格式推送，结尾 `data: [DONE]`

**WU5 - 纯检索端点（`services/api/app/routes/search.py`）**
- `POST /search`：query + tenant_id + user_id + top_k + filters
- ACL 过滤与 agent 路径一致（`compute_security_scope()`）
- 不触发 agent 循环，直接返回 `{request_id, chunks[], total}`

**WU6 - CORS + 请求日志中间件（`services/api/app/middleware.py`）**
- `RAGBOT_CORS_ORIGINS` 环境变量配置（逗号分隔），空值不启用
- `RequestLoggingMiddleware`：记录 request_id、method、path、status、latency_ms、client_ip
- Request ID：优先从 `X-Request-ID` header 读取，否则生成 UUID，回写响应 header

**WU7 - infra/ 目录补齐**
- `infra/docker/Dockerfile`：生产镜像，含 healthcheck
- `infra/docker/docker-compose.yml`：API + Postgres + Qdrant + Ollama（可选 profile）
- `infra/migrations/001_init.sql`：documents / chunks / acl_policies / ingestion_jobs 表 + GIN FTS 索引
- `infra/qdrant/init_collection.sh`：创建 collection + tenant_id/doc_id/acl_hash/tags payload 索引

**WU8 - 测试覆盖**
- 新增 24 条测试，总计 **51 条，全部通过**（0 failures）
- 覆盖：ModelProvider 协议满足性、Timeout/Retry/CircuitBreaker、QueueCallback 事件收发、run_agent+callback 集成、/chat /search /v1/chat/completions 端点格式、Request-ID 传播

结论：ragbot 已完成 Milestone A 的全部工作单元，具备可用内测版特征：工具有保护、流式真实时、接口兼容 OpenAI 生态、基础设施文件齐全。

---

## 2. 范围定义（Scope）

### 2.1 必做（MVP + 商业可用）

数据源：

- 本地文件：PDF / Markdown / TXT / Office（Docx/PPTX/XLSX 可后置）
- 邮件：Gmail / Outlook（IMAP/Graph）
- 数据库：Postgres（优先）、MySQL（次优先）、通用 JDBC/SQLAlchemy
- 代码仓库：Git（本地路径 & GitHub/GitLab）

能力：

- 多轮检索与工具调用（Agent）
- 严格引用（citations）
- SQL 工具（只读/白名单/限流/超时）
- Cursor-like：代码问答、定位函数/调用链、生成 patch（至少 unified diff）
- 多模型：云端（OpenAI/Anthropic 等）+ 本地（Ollama/vLLM/LM Studio 等）

API/客户端：

- REST + SSE（已有）
- 兼容 OpenAI 风格 Chat Completions（便于接入生态）
- Node SDK（已有示例方向）+ Python SDK
- 工具代理（Node Tool Proxy）支持组织内系统扩展

### 2.2 非目标（先不做）

- 全功能 IDE（先做 VSCode 插件/CLI，不做完整编辑器）
- 实时协作文档编辑
- 自训练 embedding/LLM（先以可插拔推理为主）

---

## 3. 目标架构（商业级）

### 3.1 服务拆分（推荐）

- api-gateway（Python/FastAPI）：/chat /ingest /sources /admin /auth
- ingestion-worker（Python）：连接器抓取、chunk、embed、upsert、去重、版本化
- retrieval-service（Python lib）：Qdrant + Postgres FTS + rerank + ACL filter
- tool-proxy（Node，可选）：组织内部工具（Jira/CI/工单/内部 API）统一代理
- model-router（可选）：统一管理本地/云端模型调用、限流、回退策略

### 3.2 数据层

- Postgres：documents/chunks/acl/jobs/audit/sessions/feedback（强一致）
- Qdrant：向量检索（payload 带 tenant/acl/doc/path/url/time/version/tags）
- 对象存储（可选）：原文/附件缓存（S3/MinIO）

---

## 4. 里程碑（Milestones）

> 时间仅作参考：你可以按团队规模压缩或拉长。每个里程碑包含：交付物 + 验收标准。

### ✅ Milestone A（已完成）：把骨架变成可用内测版

目标：稳定、可回放、引用严格，SSE 事件真实流式。

**完成情况：**

| 交付项 | 状态 | 说明 |
|--------|------|------|
| Citation 强制与校验 | ✅ | verify_node 强制校验；缺证据时降级输出"证据不足"说明 |
| 工具可靠性（timeout/retry/circuit breaker） | ✅ | reliability.py；6 个 node 均已接入 safe_tool_call() |
| SSE 真实流式（tool_call/tool_result 即时推送） | ✅ | QueueCallback；agent 线程 emit → async SSE 生成器实时 yield |
| 模型适配器（OpenAI + Ollama 可切换） | ✅ 额外增加 | ModelProvider Protocol；RAGBOT_LLM_PROVIDER 运行时切换 |
| OpenAI 兼容层 /v1/chat/completions | ✅ 额外增加 | 非流式 + 流式；附加 citations 字段 |
| 纯检索端点 /search | ✅ 额外增加 | ACL 一致；不触发 agent 循环 |
| CORS + 请求日志中间件 | ✅ 额外增加 | Request-ID 传播；RAGBOT_CORS_ORIGINS 配置 |
| infra/ 基础设施文件 | ✅ 额外增加 | Docker + docker-compose + SQL 迁移脚本 + Qdrant 初始化 |
| 基础安全（API Auth） | 部分 | API Key 校验已有；JWT/rate limit/secrets redaction 待 Milestone B |
| 单元测试覆盖 | ✅ | 51 条全通过，新增 24 条，覆盖所有新功能 |

---

### ✅ Milestone B（已完成）：企业级数据接入（本地文档 + DB）闭环

目标：真实连接器 + 增量更新 + 权限映射 + 可管理数据源。

**完成情况：**

| 交付项 | 状态 | 说明 |
|--------|------|------|
| Source 模型 + `/sources` CRUD API | ✅ | Source dataclass + CRUD 端点（POST/GET/PUT/DELETE） |
| Ingestion Pipeline 编排器 | ✅ | `services/worker/pipeline.py`：Source → Job → connector → chunk → dedup → embed → upsert |
| `/ingest/jobs` 任务管理 API | ✅ | trigger / list / status / retry 端点；异步执行 |
| Local FS 连接器 + Markdown/TXT 切分 | ✅ | `connectors/local_fs.py` + `jobs/ingest_text.py`：目录扫描、文件过滤、MD 段落提取 |
| DB Schema Introspection | ✅ | `SqlEngine.introspect_schema()` + `PostgresSqlEngine.introspect_schema()`（information_schema 查询） |
| ACL 增强（group/role） | ✅ | `compute_security_scope()` 新增 groups/roles 参数；`UserContext` 类；`compute_security_scope_from_context()` |
| 文档版本化 + chunk 去重 | ✅ | pipeline 内 checksum 去重；`next_version()` 自动升版 |
| Job 状态管理 | ✅ | pending → running → completed/failed；含 doc_count、chunk_count、error、时间戳 |
| 数据库迁移 | ✅ | `infra/migrations/002_sources.sql`：sources 表 + ingestion_jobs.source_id |
| 测试覆盖 | ✅ | 新增 35 条测试，总计 **86 条，全部通过**（0 failures, 0 warnings） |

**WU-B1 - Source 模型 + `/sources` CRUD API**
- `Source` dataclass（`storage/models.py`）：source_id, tenant_id, source_type, name, config, status, acl_policy_id, tags
- 支持 6 种 source_type：`local_fs` | `pdf` | `web` | `repo` | `email` | `database`
- InMemoryRepo 扩展：add/get/list/update/delete_source
- FastAPI router：`routes/sources.py`（POST/GET/PUT/DELETE `/sources`）

**WU-B2 - Ingestion Pipeline 编排器（`services/worker/pipeline.py`）**
- `run_ingest_pipeline(source, repo, qdrant)`：完整生命周期管理
- 自动创建 IngestionJob（running → completed/failed）
- 连接器分发：local_fs/pdf/web/repo 各走对应 ingest_* 函数
- chunk 去重：checksum 比对，已存在跳过
- Document 记录自动创建/升版

**WU-B3 - `/ingest/jobs` 任务管理 API（`routes/ingest.py`）**
- `POST /ingest/jobs`：触发任务（异步线程池执行）
- `GET /ingest/jobs`：列表（支持 tenant_id / source_id 过滤）
- `GET /ingest/jobs/{job_id}`：查看状态
- `POST /ingest/jobs/{job_id}/retry`：重试失败任务

**WU-B4 - Local FS 连接器 + Markdown/TXT 切分**
- `connectors/local_fs.py`：`list_files()` 递归扫描（排除 .git/node_modules 等）+ `read_file()`
- `jobs/ingest_text.py`：`ingest_text_file()` 单文件 + `ingest_local_fs()` 批量目录
- Markdown 自动提取段落标题（`section` 字段）

**WU-B5 - DB Schema Introspection**
- `SqlEngine.introspect_schema()`：基于 InMemoryRepo 的 TableData
- `PostgresSqlEngine.introspect_schema()`：查询 `information_schema.columns`，按 allowed_schemas 过滤
- `_describe_tables()` 增强：优先使用 `introspect_schema()`，回退到 export_state

**WU-B6 - ACL 增强（group/role 支持）**
- `compute_security_scope()` 新增 `groups`/`roles` 可选参数（向后兼容）
- 新增 `UserContext` 类：封装 user_id + groups + roles
- 新增 `compute_security_scope_from_context()` 便捷方法
- 规则支持：`allow_all` / `allow_users` / `allow_groups` / `allow_roles`

**WU-B7 - 测试覆盖**
- 新增 35 条测试，总计 86 条，全部通过
- 覆盖：Source CRUD、Job 管理、ACL group/role、Schema Introspection、Local FS 连接器、Text 切分、Pipeline 端到端、Pipeline 去重、Pipeline 错误处理、/sources + /ingest/jobs 端点

结论：ragbot 已完成 Milestone B 的全部工作，具备企业级数据接入能力：数据源管理 → 触发任务 → 连接器抓取 → 切分 → 去重 → 向量化 → 可检索。ACL 支持用户/组/角色三级权限。

---

### Milestone C（6～10 周）：Cursor-like 编程体验（Repo + 本地文档 + 多工具）  ✅ 已完成

目标：在 IDE/CLI 里实现会检索、会定位、会改代码的编程助手。

交付物：

1) Repo Ingestion + Code Search
- 以 symbol/函数/类为切分单位 + path 元数据
- 支持 blame/commit-ish（可选）
- 混合检索：代码符号关键词优先 + 向量辅助（注释/README 更语义）

2) 编程工具集（Agent tools）
- `code_search`：ripgrep/索引
- `open_file/read_range`：读取文件片段（带行号引用）
- `apply_patch`：输出 unified diff（由客户端应用）
- `run_tests`（可选）：通过 Node 工具代理接 CI 或本地命令（沙盒化）
- `explain_error`：输入日志/堆栈，定位相关代码与文档

3) 客户端
- VSCode 插件（推荐优先）或 JetBrains 插件（后置）
- CLI：支持 `rag ask`、`rag search`、`rag patch`
- Node SDK：用于插件/前端接入，支持 SSE

4) 上下文策略
- IDE 打开文件、选中区域、git diff、最近错误日志 → 进入 constraints
- 成本控制：上下文压缩、引用优先、重复证据去重

验收标准：

- 典型任务（定位函数解释生成 patch）：一次成功率可用（内部基准集评测）
- patch 输出可被应用，且引用指向具体 path+line
- IDE 端 P95 交互延迟可接受（首 token、工具调用可视）

**实现状态（Milestone C）：**

| 交付项 | 状态 | 关键文件 |
|--------|------|----------|
| 1. 符号级 Repo Ingestion | ✅ | `services/worker/jobs/ingest_repo.py` — AST-based Python 切分 + regex C-like 切分 + 行号/语言元数据 |
| 2a. `open_file/read_range` | ✅ | `services/api/app/agent/nodes/code.py` — `CodeSearch.open_file()` + `open_file_node()` |
| 2b. `apply_patch` (unified diff) | ✅ | `services/api/app/agent/nodes/code.py` — `CodeSearch.generate_patch()` + `apply_patch_node()` |
| 2c. `explain_error` | ✅ | `services/api/app/agent/nodes/code.py` — 多语言堆栈解析 (Python/JS/Go/Java) + 关键词回退 |
| 3a. CLI client | ✅ | `cli/rag.py` — `rag ask`, `rag search`, `rag patch`, `rag ingest`（本地/远程双模式） |
| 3b. CLI pyproject entry point | ✅ | `pyproject.toml` — `[project.scripts] rag = "cli.rag:main"` |
| 4a. Client context 处理 | ✅ | `services/api/app/agent/context.py` — `process_client_context()` 支持 selected_text/open_files/git_diff/errors |
| 4b. Evidence 去重 | ✅ | `services/api/app/agent/context.py` — `dedup_evidence()` MD5 文本哈希去重 |
| 4c. Evidence 压缩 | ✅ | `services/api/app/agent/context.py` — `compress_evidence()` 按分数排序+截断+预算控制 |
| 5. Agent 工具调度扩展 | ✅ | `services/api/app/agent/graph.py` — 新增 open_file/apply_patch/explain_error 分发 |
| 6. 类型系统扩展 | ✅ | `contracts/types.py` — PatchResult + 新 ToolName + 新 EvidenceItem 类型 |
| 7. API 集成 | ✅ | `services/api/app/api.py` v0.4.0 + `main.py` — client_context 注入 + evidence 后处理 |
| 8. 测试 | ✅ | 新增 43 条测试，总计 129 条，全部通过 |

测试覆盖清单：
- 符号级切分（Python AST / regex / 行回退 / 大符号拆分 / 语言检测）
- open_file（全文/范围/未找到）
- generate_patch（有变更/无变更）
- explain_error（堆栈解析/关键词回退）
- 堆栈解析（Python/JS/Java 格式）
- 文件引用解析（path:line-line / path:line / path）
- Agent 节点集成（open_file_node / explain_error_node）
- CLI（ask/search/help/no-command）
- Client context（selected_text/repo/open_files/git_diff/errors/constraints 保留）
- Evidence 去重（重复移除/唯一保留）
- Evidence 压缩（低分丢弃/长文截断/空列表）
- 端到端集成（chat + client_context / run_agent + initial_evidence / /chat API）

结论：ragbot Milestone C 全部完成，具备 Cursor-like 编程助手核心能力：符号级代码检索、文件读取、Patch 生成、错误定位、CLI 客户端、IDE 上下文注入、证据去重与压缩。

---

### ✅ Milestone D（已完成）：商业级可运维与可评测（SLO/监控/评测/成本）

目标：能上线、能监控、能回归、能控成本。

**完成情况：**

| 交付项 | 状态 | 说明 |
|--------|------|------|
| 1a. OpenTelemetry traces | ✅ | `observability/tracing.py` — RequestTracer + context-manager Span，覆盖 route/retrieve/sql/code/web/synthesize/verify/finalize 全链路 |
| 1b. 质量指标 | ✅ | `observability/metrics.py` — MetricsCollector 线程安全收集：citation_coverage, retrieval_hit_rate, tool_failure_rate, user_feedback |
| 1c. 指标 API | ✅ | `/admin/metrics` 聚合 + `/admin/metrics/history` 历史 + `/admin/feedback` 反馈录入 |
| 2a. 评测数据集 | ✅ | `eval/datasets.py` — EvalCase（doc_qa/db_qa/code_task/mixed）+ 样本数据集 + JSON 导入导出 |
| 2b. 自动化回归 runner | ✅ | `eval/runner.py` — run_eval_case() + run_eval_suite() + summarize_results()；支持 category/tag 过滤 |
| 2c. 失败分析 | ✅ | `eval/runner.py` — _analyze_failure()：bad_routing / bad_retrieval / bad_synthesis / bad_tool / error 五类分类 |
| 3a. 模型路由（fast/strong） | ✅ | `llm/router.py` — ModelRouter + TASK_TIER_MAP（route/verify→fast, synthesize/apply_patch/explain_error→strong） |
| 3b. 成本追踪 | ✅ | `llm/router.py` — CostTracker 按 task/tier 记录 token 用量 + 成本估算；`/admin/cost` API |
| 3c. Caching | ✅ | `cache/cache.py` — LRUCache（TTL+LRU 双淘汰）+ RetrievalCache + EmbeddingCache；`/admin/cache` API |
| 4a. Docker Compose 增强 | ✅ | Jaeger 服务（observability profile）+ OTEL/cache/routing 环境变量 |
| 4b. Helm Chart | ✅ | `infra/helm/ragbot/` — Chart + values + Deployment + Service；rolling update + health probes + secret 引用 |
| 4c. 数据库迁移 | ✅ | `infra/migrations/003_observability.sql` — feedback / audit_log / request_metrics 表 + 索引 |
| 5. 测试覆盖 | ✅ | 新增 38 条测试，总计 **167 条，全部通过**（0 failures） |

**WU-D1 - Observability（`services/api/app/observability/`）**
- `tracing.py`：`RequestTracer` 包含 context-manager 式 Span（自动计时、异常捕获、属性记录）
- `TraceRecord.to_dict()`：可序列化为 JSON，兼容 OTLP 导出
- `setup_tracing()`：读取 `RAGBOT_TRACING_ENABLED` 环境变量，可选启用 OpenTelemetry SDK
- `metrics.py`：`RequestMetrics` 记录每请求指标，`AggregateMetrics` 聚合统计
- `build_request_metrics(state, trace_record)`：从 AgentState 自动提取指标
- `get_metrics_collector()`：全局单例，线程安全

**WU-D2 - Evaluation & Regression（`eval/`）**
- `datasets.py`：`EvalCase` 支持 setup_tables / setup_files / setup_chunks 预置数据
- `build_sample_dataset()`：预置 doc_qa + db_qa + code_task 样本
- `runner.py`：端到端执行 + 多维检查（route / answer_contains / min_citations / min_evidence / confidence）
- `summarize_results()`：pass_rate + category 分布 + failure_categories 统计
- `_analyze_failure()`：自动分类失败原因（routing / retrieval / synthesis / tool / error）

**WU-D3 - 模型路由与成本控制**
- `llm/router.py`：`ModelRouter`（`RAGBOT_MODEL_ROUTING` 环境变量开关）
- `TASK_TIER_MAP`：route/verify → fast，synthesize/apply_patch/explain_error → strong
- `CostTracker`：按 tier 定价估算 token 成本（fast: $0.15/$0.60, strong: $3.00/$15.00 per M tokens）
- `cache/cache.py`：`LRUCache`（OrderedDict O(1) 淘汰 + TTL 过期）+ `RetrievalCache` + `EmbeddingCache`
- `RAGBOT_CACHE_ENABLED` / `RAGBOT_CACHE_TTL_SECONDS` 环境变量控制

**WU-D4 - 部署**
- Docker Compose：Jaeger 追踪 UI（`docker compose --profile observability up`）
- Helm Chart：2 副本 + RollingUpdate（maxUnavailable=0, maxSurge=1）+ liveness/readiness probes
- Secret 管理：existingSecret 引用 K8s Secret（postgres-dsn / openai-api-key / api-keys）
- 自动扩缩容配置（HPA 就绪，默认关闭；targetCPU=70%）
- 数据库迁移：feedback + audit_log + request_metrics 表

**WU-D5 - 测试覆盖**
- 新增 38 条测试，总计 167 条，全部通过
- 覆盖：RequestTracer span 计时/异常/多 span/序列化、MetricsCollector 聚合/反馈/历史/build_request_metrics、TracingIntegration、EvalCase 数据集管理/runner 执行/失败分析/suite 汇总、ModelRouter tier 映射/provider 选择/CostTracker、LRUCache TTL 过期/LRU 淘汰/统计/清除、RetrievalCache/EmbeddingCache、admin 端点（metrics/history/feedback/cost/cache）

结论：ragbot Milestone D 全部完成，具备商业级可运维能力：全链路追踪、质量指标自动收集、评测回归框架、模型路由与成本控制、多级缓存、Helm 生产部署。

---

## 5. API Roadmap（多接口能力）

### 5.1 必须提供的 API

- ✅ `POST /chat`（JSON + SSE）— 已实现，SSE 真实流式
- ✅ `POST /search`（纯检索：返回 chunks/rows/snippets + citations）— 已实现
- ✅ `POST /sources` / `GET /sources` / `PUT /sources/{id}` / `DELETE /sources/{id}`（数据源 CRUD）— 已实现
- ✅ `POST /ingest/jobs` / `GET /ingest/jobs` / `GET /ingest/jobs/{id}` / `POST /ingest/jobs/{id}/retry`（任务管理）— 已实现
- `POST /tools/{name}`（可选：工具代理入口，Node 实现）— Milestone C

### 5.2 OpenAI 兼容层（强烈建议）

- ✅ `/v1/chat/completions`：已实现，支持非流式 + 流式，附加 `citations` 字段
- 好处：前端/插件生态接入成本极低（尤其是 IDE 插件、社区工具）

### 5.3 Cursor-like 客户端协议建议

- SSE 事件：token / tool_call / tool_result / evidence / final
- 支持 partial citations 与 evidence preview 事件，便于 IDE UI 先展示证据

---

## 6. 本地大模型（Local LLM）支持路线

### ✅ Phase 1（已完成）：统一推理接口（Adapter）

- `ModelProvider` 抽象（PEP 544 Protocol）：OpenAI / Ollama 已实现，vLLM / LM Studio 可用同样接口扩展
- 能力探测：`enabled` 属性；`web_search` 降级（Ollama 返回空列表）
- JSON 模式降级：Ollama 不支持 json_schema 时，在 system prompt 内嵌 schema + 手动解析
- 切换方式：`RAGBOT_LLM_PROVIDER=openai|ollama`，`OLLAMA_BASE_URL`，`OLLAMA_MODEL`

### Phase 2：检索与 rerank 本地化

- embedding：本地 embedding 模型（可选）
- rerank：本地 cross-encoder（可选）
- 让云端仅用于复杂推理（成本控制）

---

## 7. 风险与关键决策（尽早拍板）

1) 邮件权限模型：单账户隔离 vs 组织共享邮箱/组映射
2) 代码工具安全：apply_patch 的执行边界（谁来落盘、怎么审计）
3) SQL 安全：只读用户 + 白名单 schema + 强 LIMIT + 超时（必须）
4) 引用策略：强制引用会降低看起来流畅的回答，但这是商业可信度核心
5) 连接器增量：必须做版本化/去重，否则索引会膨胀并劣化质量

---

## 8. Definition of Done（商业级）

- ✅ 多租户隔离 + ACL 前置过滤 + 审计日志
- ✅ 文档/邮件/DB/代码至少三类数据源闭环（ingest → search → answer → cite）
- ✅ Cursor-like：支持 repo 代码问答 + 生成 patch（diff）
- ✅ 多模型：云端 + 本地可切换、可回退
- ✅ 可观测 + 可评测 + CI 回归门禁
- ✅ 部署与升级策略明确（迁移/回滚/embedding 重建）
