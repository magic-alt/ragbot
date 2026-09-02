# Ragbot v1.0 Release Readiness

本文档把 Ragbot 的能力声明映射到当前实现、自动验证和 v1.0 发布门槛。它区分两个问题：

1. **功能/产品能力是否已经实现？** —— 看 Capability Matrix。
2. **是否已经可以打 `v1.0.0` tag？** —— 看 Release Gates。

当前 Python package / FastAPI / Helm metadata 仍为 `0.5.0`。版本号、tag 和 GitHub Release 必须在所有 blocking gate 满足后通过独立 release-only PR 完成。

## 1. Capability Matrix

| 能力 | 状态 | 主要实现 | v1 验证/约束 |
|---|---|---|---|
| Quick Import / 快速建库 | **已实现** | `routes/quick_import.py`, `cli/rag.py` | 单请求 Source upsert + Job enqueue；自动类型判断；Source reuse；active Job dedupe；显式 idempotency；manifest batch；CLI `--wait` |
| PDF / Web / Git / local filesystem 摄取 | **已实现** | `services/worker/jobs/*`, `services/worker/pipeline.py` | Source → chunk → dedup → shared embedder → PostgreSQL/Qdrant；重摄取复用未变化 chunk 并清理 stale SQL/vector 数据 |
| Durable ingestion | **已实现** | `routes/ingest.py`, `services/worker/main.py`, migration 006 | PostgreSQL pending queue → `FOR UPDATE SKIP LOCKED` claim → lease/heartbeat → worker；过期 lease 可恢复，达到 max attempts 后失败 |
| Qdrant vector search | **已实现** | `retrieval/qdrant.py`, `retrieval/service.py` | tenant/source/doc/tag/path/url/time/ACL filter；collection dimension 校验 |
| PostgreSQL FTS | **已实现，含 CJK baseline** | `storage/pg_repo.py`, `retrieval/lexical.py`, `retrieval/pg_fts.py` | ASCII `simple` FTS；CJK application-generated bigram lexemes + GIN；Recall@5/MRR regression gate |
| RRF hybrid fusion | **已实现** | `retrieval/rerank.py`, `retrieval/service.py` | vector + lexical ranking 融合；有 modality-crowd-out regression test |
| optional cross-encoder rerank | **已实现** | `retrieval/cross_encoder.py` | provider 失败时回退到 RRF，不使检索整体失败 |
| Agentic RAG graph | **已实现** | `agent/graph.py`, `agent/nodes/*` | route → tool → synthesize → verify → retry/finalize；支持 doc/sql/code/web 路由 |
| Tenant / user / ACL | **已实现为 API-key principal 模型** | `auth/acl.py`, `auth/principal.py` | API key 绑定 tenant/user/groups/roles；HTTP payload/header 不能扩大授权范围 |
| OpenAI-compatible LLM | **已实现** | `llm/client.py`, `llm/provider.py` | `/v1/chat/completions` adapter 与内部 provider |
| Ollama LLM adapter | **已实现** | `llm/ollama.py` | Ollama OpenAI-compatible chat path；embedding 单独配置兼容 endpoint/model |
| `/search` | **已实现** | `routes/search.py` | pure retrieval tool surface，带 trusted tenant/ACL filtering |
| `/chat` + SSE | **已实现** | `api.py`, callbacks, graph | streaming/non-streaming 共用身份与约束语义；错误路径保证 stream 终止 |
| Source / Ingest low-level API | **已实现** | `routes/sources.py`, `routes/ingest.py` | CRUD/job 查询按 principal tenant 隔离；PostgreSQL 部署只入队，由独立 worker 执行 |
| Deployment doctor / probes | **已实现** | `cli/rag.py`, `/admin/health`, `/admin/ready` | CLI 可快速验证 API liveness 与 PostgreSQL/Qdrant readiness |
| Metrics / tracing / cache | **已实现** | `observability/*`, `cache/*` | global metrics/cost/cache 仅允许 admin principal；liveness/readiness 分离 |
| Docker Compose | **已实现** | root + `infra/docker` Compose | API + durable worker + migration + PostgreSQL + Qdrant；可选 Ollama/Jaeger |
| Helm | **已实现** | `infra/helm/ragbot` | API/worker Deployments、migration initContainer、Ingress/HPA、readiness、shared-store guard、source mounts |
| Node client | **已实现基础 SDK** | `packages/node-client` | TypeScript strict typecheck 进入 CI |
| Evaluation | **已实现基础框架** | `eval/`, `benchmarks/` | CJK regression 自动执行；1000-PDF PostgreSQL/Qdrant offline integration baseline 已建立；真实 provider/domain corpus 仍需 staging evidence |

## 2. 已封闭的主要 v1 blockers

### 2.1 Production 不允许静默降级

`RAGBOT_ENV=production` 要求 PostgreSQL、Qdrant、semantic embedding、API keys 与 API-key principal mappings。生产模式不会静默落到 `InMemoryRepo`、`InMemoryQdrant` 或 `HashEmbedder`。

生产环境同时禁止 `RAGBOT_INGESTION_MODE=inline`：摄取必须经 PostgreSQL durable queue + independent worker。

### 2.2 快速建库不再要求调用者手工编排两阶段 API

普通产品流程可以直接使用：

```bash
rag --server http://localhost:8000 --tenant engineering ingest /data/manuals --wait
rag --server http://localhost:8000 import examples/ragbot-manifest.json --wait
```

或 HTTP `POST /ingest/quick` / `POST /ingest/batch`。

默认 Source identity 基于 `tenant + source type + canonical location` 稳定派生；重复 bootstrap 默认复用 Source，并复用已有 pending/running Job。显式 `idempotency_key` 为重复请求提供 exact Job replay；该模式要求 `reuse_source=true`，避免“每次新 Source”与“严格幂等”语义冲突。

这使 Ragbot 能作为部署后可直接建立知识库的产品，而不是要求每个客户端自行实现 Source/Job orchestration。

### 2.3 Durable ingestion queue

API 创建任务时首先在 PostgreSQL 保存 `pending` Job。独立 worker：

1. 使用 `SELECT ... FOR UPDATE SKIP LOCKED` 原子 claim；
2. 记录 worker lease、heartbeat 与 attempts；
3. 周期性续租；
4. worker/API/节点异常退出后，过期 lease 可被其他 worker 回收；
5. 达到 `RAGBOT_WORKER_MAX_ATTEMPTS` 后任务进入 `failed`，不会无限重试。

Compose 默认启动 worker；Helm production render 要求 `worker.enabled=true`。

### 2.4 API identity 不由客户端自行扩大

`RAGBOT_API_KEY_PRINCIPALS` 将 API key 映射到 allowed tenant、stable user、groups、roles 和 optional admin。`/chat`、`/search`、OpenAI-compatible API、Quick Import、Source API 与 Job API 使用同一 tenant authorization boundary。

这是适合 service-to-service 的 v1 身份模型，不等价于完整企业 OIDC/OAuth2/SAML。

### 2.5 Source ingestion boundary

- Web / remote PDF / remote Git 默认拒绝 loopback、private、link-local、reserved 等地址；
- redirect 每跳重新验证；
- hostname allowlist 可进一步收窄 Web/PDF/Git；
- local_fs / local PDF / local Git 必须位于 `RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS`；
- Web/PDF 使用 bounded download；
- remote Git production path 限定 HTTPS；
- connector dependency 缺失时明确失败。

应用层校验不能替代网络隔离；高安全部署仍应配置 egress firewall/service mesh/proxy policy。

### 2.6 Vector-index contract

Embedding API 实际输出维度必须严格等于 `QDRANT_DIM`，不会静默 truncate/zero-pad。模型或维度变化必须执行 reindex/migration。

CLI 本地摄取也显式传递同一个 configured embedder，避免“本地 ingest 使用 fallback embedding、query 使用 semantic embedding”的索引合同漂移。

### 2.7 Hybrid / CJK retrieval baseline

PostgreSQL 使用 `simple` text-search configuration，并为连续 CJK 文本额外生成 overlapping bigram lexemes；查询使用相同 lexical representation，再通过 GIN-backed FTS 检索。

固定 CJK regression corpus 的已验证 baseline：

- **Recall@5 = 1.000**
- **MRR = 1.000**

CI floor 为 Recall@5 ≥ 0.90、MRR ≥ 0.80。

RRF 还包含 disjoint vector/lexical ranking regression coverage，防止某一 modality 因权重/候选窗口组合而被结构性挤出 final window。

这类结果只是防回归 baseline，不代表任意企业语料都达到相同 semantic quality。

### 2.8 1000-PDF integration/capacity baseline

仓库提供真实 PostgreSQL + Qdrant 的 deterministic 1000-PDF benchmark，覆盖 PDF 生成/解析、ingestion、chunk/vector count、hybrid retrieval、Recall@5、MRR、latency、re-ingestion reuse、数据库/RSS footprint。

已验证 benchmark 证明当前架构可以在 GitHub-hosted runner 上完成 1000 PDF 的端到端离线 integration/capacity gate。详见 [`BENCHMARK_1000_PDF.md`](BENCHMARK_1000_PDF.md)。

该 benchmark 使用 deterministic HashEmbedder 以消除外部 provider 依赖，因此不能替代真实模型 + 真实 domain corpus 的质量评测。

### 2.9 Optional reranker availability

Reranker 是质量增强层，不是检索可用性的单点故障。provider 失败时保留 Qdrant + PostgreSQL FTS + RRF 结果。

### 2.10 Open-source license

仓库采用 **Apache License 2.0**，具备明确版权许可与 patent grant。CLA/DCO 如需引入应单独决定。

## 3. v1.0 Release Gates

以下 blocking 条件满足后才应把 package/chart 从 `0.5.0` 提升到 `1.0.0`。

### Gate A — Automated CI

- [x] Python 3.10 tests
- [x] Python 3.12 tests
- [x] dependency consistency / compileall
- [x] PostgreSQL full migration chain，包括 migration 006
- [x] PostgreSQL durable queue lease/recovery integration
- [x] PostgreSQL native FTS + CJK integration smoke
- [x] CJK Recall@5/MRR regression gate
- [x] hybrid RRF regression coverage
- [x] Quick Import / idempotency / manifest regression coverage
- [x] Node SDK typecheck
- [x] bundled executable example
- [x] Docker Compose config validation，包括 worker service
- [x] Helm lint/default render
- [x] Helm production render with durable worker
- [x] Helm production config rejects missing worker
- [x] v1 production/security regression tests
- [x] deterministic 1000-PDF PostgreSQL/Qdrant capacity/integration baseline

这些项目只表示仓库已有自动化 gate；每个拟合并 PR 和 release commit 仍必须以其 **exact head SHA** 对应的 Actions 结果为准。

### Gate B — Real dependency staging smoke

仓库提供 `.github/workflows/staging-smoke.yml` 与 `eval/staging_smoke.py`。workflow 使用 production mode、PostgreSQL、Qdrant、independent worker 和真实 OpenAI-compatible credential，需实际完成：

- [ ] production-compatible embedding provider/model
- [ ] production-compatible LLM provider/model
- [ ] local filesystem ingestion
- [ ] Web ingestion
- [ ] PDF ingestion
- [ ] Git ingestion
- [ ] hybrid `/search`
- [ ] Agentic `/chat`
- [ ] ACL negative isolation

运行前在 GitHub `staging` environment 配置 `STAGING_OPENAI_API_KEY`；可通过 environment variables 覆盖 base URL、model 与 embedding dimension。

**在该 workflow 使用目标生产 provider 实际成功运行之前，不应发布 `v1.0.0`。**

如果 v1 production 将启用 reranker，还应补一次该 provider staging smoke。

### Gate C — Security / operations

- [ ] staging/production 使用 `RAGBOT_ENV=production`
- [ ] secrets 存储在 secret manager/Kubernetes Secret/GitHub Environment，而非 Git
- [ ] every API key has principal mapping
- [ ] Web/PDF/Git allowlist 或受控 egress policy 已定义
- [ ] local source roots 显式挂载且尽可能 read-only
- [ ] TLS / ingress auth / rate limiting policy 已定义
- [ ] PostgreSQL/Qdrant 不直接暴露公网
- [ ] immutable application/container image tag or digest 已选择
- [ ] PostgreSQL backup/restore 已实际验证
- [ ] Qdrant snapshot/restore 已实际验证
- [ ] rollback、database migration、embedding reindex runbook 已 review

### Gate D — Project/release metadata

- [x] Apache-2.0 `LICENSE`
- [x] `CHANGELOG.md` maintains `[Unreleased]`
- [ ] complete Gate B / Gate C evidence
- [ ] bump Python package version to `1.0.0`
- [ ] bump FastAPI/Helm chart/app/image metadata consistently
- [ ] freeze exact release commit and pass full CI on it
- [ ] create signed/annotated `v1.0.0` tag according to project policy
- [ ] publish GitHub Release with upgrade/migration/security notes

## 4. Product maturity roadmap after this gate

以下项目很重要，但不是当前 service-oriented v1.0 的 blocking 条件：

1. **Web/Admin UI**：Source catalog、drag/drop/upload onboarding、ingestion progress、failed-job retry、retrieval inspection、evaluation dashboard。
2. **Scheduled synchronization**：cron/source refresh policy、changed-source detection、connector sync history、webhook/event-driven update。
3. **Connector expansion**：S3/MinIO、Google Drive/SharePoint/Notion/Confluence 等企业知识源，并为每种 connector 保持同一 SSRF/credential boundary。
4. **Queue observability / autoscaling**：queue depth、oldest-job-age、lease expiry、throughput、failure reason、基于 backlog 的 worker autoscaling。
5. **Enterprise IAM**：OIDC/OAuth2/SAML、组织目录同步、centrally managed RBAC。
6. **Larger domain retrieval evaluation**：真实企业 corpus 上比较 embedding/reranker、Recall/MRR/NDCG/latency，并按收益决定 PGroonga/pg_jieba/external lexical index。
7. **Cross-store activation semantics**：PostgreSQL/Qdrant 之间从当前 retry/reconcile 进一步演进 staged source version + outbox/reconciler。
8. **Authoritative token accounting**：OpenAI-compatible usage 直接使用 provider authoritative token data。
9. **Defense-in-depth egress**：应用层 connector validation 与网络层 egress policy 联动。

## 5. Recommended release sequence

1. 把当前产品化/快速建库 PR 的 **最终 head** 完整 CI 跑绿并合并。
2. 在 GitHub `staging` environment 配置目标真实 provider credential，执行 `Staging Smoke`。
3. 保存 Gate B/Gate C evidence，特别是 ACL negative、backup/restore、egress/TLS/secrets policy。
4. Freeze API/contracts、Quick Import semantics 与 evaluation baselines。
5. 从已验证 main 创建小型 `release/v1.0.0` PR，只包含版本、changelog/date 与 immutable release metadata。
6. 在该 exact release commit 上再次运行完整 CI，并确认与 main 无冲突。
7. 仅从该 commit 创建 `v1.0.0` tag 与 GitHub Release。
