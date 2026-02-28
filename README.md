# Ragbot

商业级 Agentic RAG 平台 — 支持文档/数据库/代码仓库的检索、总结、引用与多步工具调用。

> **v0.5.0** | Milestone A–D 已完成 | 167 项测试全部通过

## 核心特性

- **Agentic RAG**：route → retrieve/sql/code/web → synthesize → verify → finalize 多步循环
- **多数据源**：PDF / Web / Git / 本地文件，Source CRUD + Ingestion Pipeline
- **Cursor-like 编程助手**：代码问答、open_file、apply_patch、explain_error
- **混合检索**：Qdrant 向量 + PostgreSQL FTS → RRF 融合
- **多模型**：OpenAI + Ollama 可切换（ModelProvider Protocol）
- **工具可靠性**：timeout / retry / circuit breaker
- **SSE 流式**：真实时事件推送（QueueCallback 线程安全队列）
- **OpenAI 兼容**：`/v1/chat/completions` 端点
- **多租户安全**：ACL 前置过滤（user/group/role）+ API Key 认证 + 审计日志
- **可观测**：全链路追踪（OpenTelemetry 兼容）+ 质量指标 + 成本追踪
- **可评测**：EvalCase 数据集 + 自动回归 + RAGAS 评测
- **多级缓存**：LRU (TTL+LRU) + RetrievalCache + EmbeddingCache
- **部署就绪**：Docker Compose + Helm Chart + SQL 迁移

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行测试

```bash
python -m pytest tests/test_agent.py -v
```

### 启动服务

```bash
# 内存适配器模式（开发）
uvicorn services.api.app.api:app --reload --host 0.0.0.0 --port 8000

# Docker Compose（含 Postgres + Qdrant + Ollama + Jaeger）
docker-compose -f infra/docker/docker-compose.yml up -d
```

### 环境变量

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

# 数据库
POSTGRES_DSN=postgresql://user:pass@localhost:5432/ragbot

# 安全
RAGBOT_API_KEYS=key1,key2            # 空值允许所有请求

# 可观测
RAGBOT_TRACING_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

完整环境变量列表见 [ROADMAP.md §2.6](ROADMAP.md)。

## API 端点

| 端点 | 方法 | 功能 |
|---|---|---|
| `/chat` | POST | 主 Agentic RAG（JSON + SSE 流式） |
| `/search` | POST | 纯检索（不触发 Agent） |
| `/v1/chat/completions` | POST | OpenAI 兼容层 |
| `/sources` | CRUD | 数据源管理 |
| `/ingest/jobs` | POST/GET | 摄取任务管理 |
| `/admin/health` | GET | 健康检查 |
| `/admin/metrics` | GET | 质量指标 |
| `/admin/feedback` | POST | 用户反馈 |
| `/admin/cost` | GET | LLM 成本追踪 |

## CLI

```bash
rag ask "Postgres 在系统中的作用"      # 问答
rag search "向量检索"                   # 纯检索
rag patch "修复类型错误" --repo .       # 生成 patch
rag ingest ./docs --source-type pdf    # 数据摄取
```

## 架构

```
Client (CLI / Node SDK / IDE)
    │  REST + SSE
    ▼
API Gateway (FastAPI)
    │  Agent Pipeline: route → action loop → synthesize → verify → finalize
    ▼
┌──────────┬──────────┬─────────┬───────────┐
│ Qdrant   │ Postgres │  LLM    │  Worker   │
│ (vector) │ (FTS+DB) │ OpenAI/ │ PDF/Git/  │
│          │          │ Ollama  │ Web/FS    │
└──────────┴──────────┴─────────┴───────────┘
```

## 目录结构

```
ragbot/
├── contracts/          # 跨语言共享契约（types.py/ts, openapi.yaml, tools.schema.json）
├── services/
│   ├── api/app/        # API 网关 + Agent 状态机 + 检索 + 存储 + LLM + 认证 + 可观测
│   └── worker/         # 摄取 Pipeline（connectors + jobs + dedup）
├── cli/                # CLI 客户端（rag ask/search/patch/ingest）
├── eval/               # 评测框架（datasets + runner + RAGAS）
├── packages/node-client/ # Node SDK（chat + chatStream SSE）
├── infra/              # Docker + Helm + SQL 迁移 + Qdrant 初始化
└── tests/              # 167 项单元/集成测试
```

详细架构、数据模型、技术路线见 [ROADMAP.md](ROADMAP.md)。

## 里程碑

| 版本 | 里程碑 | 测试数 |
|------|--------|--------|
| v0.2.0 | A — 可用内测版（ModelProvider + 可靠性 + SSE + OpenAI 兼容） | 51 |
| v0.3.0 | B — 企业数据接入（Source CRUD + 连接器 + ACL 增强） | 86 |
| v0.4.0 | C — Cursor-like 编程助手（代码工具 + CLI + IDE 上下文） | 129 |
| v0.5.0 | D — 商业级可运维（追踪 + 评测 + 路由 + 缓存 + Helm） | 167 |

## License

内部项目，暂不公开。
