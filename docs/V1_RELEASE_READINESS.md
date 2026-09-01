# Ragbot v1.0 Release Readiness

本文档把 README 的能力声明映射到当前实现、验证方式和 v1.0 发布门槛。它回答两个不同问题：

1. **功能是否已经实现？** —— 看 Capability Matrix。
2. **是否已经可以打 `v1.0.0` tag？** —— 看 Release Gates。

当前 Python package / FastAPI / Helm metadata 仍为 `0.5.0`。版本号、tag 和 GitHub Release 必须在所有 blocking gate 满足后通过独立 release-only PR 完成。

## 1. Capability Matrix

| README 能力 | 状态 | 主要实现 | v1 验证/约束 |
|---|---|---|---|
| PDF / Web / Git / local filesystem 摄取 | **已实现** | `services/worker/jobs/*`, `services/worker/pipeline.py` | Source → chunk → dedup → shared embedder → Qdrant；重摄取复用未变化 chunk 并清理 stale SQL/vector 数据 |
| Durable ingestion | **已实现** | `routes/ingest.py`, `services/worker/main.py`, migration 006 | PostgreSQL pending queue → `FOR UPDATE SKIP LOCKED` claim → lease/heartbeat → worker；过期 lease 可恢复，达到最大 attempts 后失败 |
| Qdrant vector search | **已实现** | `retrieval/qdrant.py`, `retrieval/service.py` | tenant/source/doc/tag/path/url/time/ACL filter；collection dimension 校验 |
| PostgreSQL FTS | **已实现，含 CJK baseline** | `storage/pg_repo.py`, `retrieval/lexical.py`, `retrieval/pg_fts.py` | ASCII 保持 `simple` FTS；CJK 使用 application-generated bigram lexemes + GIN；有 Recall@5/MRR regression gate |
| RRF hybrid fusion | **已实现** | `retrieval/rerank.py`, `retrieval/service.py` | vector + lexical ranking 融合 |
| optional cross-encoder rerank | **已实现** | `retrieval/cross_encoder.py` | optional service 失败时回退到 RRF，不使检索整体失败 |
| Agentic RAG graph | **已实现** | `agent/graph.py`, `agent/nodes/*` | route → tool → synthesize → verify → retry/finalize；支持 doc/sql/code/web 路由 |
| Tenant / user / ACL | **已实现为 API-key principal 模型** | `auth/acl.py`, `auth/principal.py` | API key 可绑定 tenant/user/groups/roles；HTTP 入口不能通过 payload/header 扩大授权范围 |
| OpenAI-compatible LLM | **已实现** | `llm/client.py`, `llm/provider.py` | `/v1/chat/completions` adapter 与内部 provider |
| Ollama LLM adapter | **已实现** | `llm/ollama.py` | Ollama OpenAI-compatible chat path；embedding 仍需单独配置兼容 endpoint/model |
| `/search` | **已实现** | `routes/search.py` | pure retrieval tool surface，带 trusted tenant/ACL filtering |
| `/chat` + SSE | **已实现** | `api.py`, callbacks, graph | streaming/non-streaming 共用身份与约束语义；错误路径保证 stream 终止 |
| `/v1/chat/completions` | **已实现** | `routes/openai_compat.py` | stream/non-stream；usage 当前为明确标记的 estimate |
| Source/Ingest API | **已实现** | `routes/sources.py`, `routes/ingest.py` | CRUD/job 查询按 API principal tenant 隔离；PostgreSQL 部署默认只入队，不在 API 进程执行 |
| Metrics / tracing / cache | **已实现** | `observability/*`, `cache/*` | global metrics/cost/cache 仅允许 admin principal；liveness/readiness 分离 |
| Docker Compose | **已实现** | root + `infra/docker` Compose | API + durable worker + migration + Postgres + Qdrant；可选 Ollama/Jaeger |
| Helm | **已实现** | `infra/helm/ragbot` | API/worker Deployments、migration initContainer、Ingress/HPA、readiness、shared-store guard、source mounts |
| Node client | **已实现基础 SDK** | `packages/node-client` | TypeScript strict typecheck 进入 CI |
| Evaluation | **已实现基础框架和 CJK release corpus** | `eval/` | CJK lexical baseline 自动执行；真实 provider/corpus 仍需 staging evidence |

## 2. 已封闭的主要 v1 blockers

### 2.1 Production 不允许静默降级

`RAGBOT_ENV=production` 要求 PostgreSQL、Qdrant、semantic embedding、API keys 与 API-key principal mappings。生产模式不会静默落到 `InMemoryRepo`、`InMemoryQdrant` 或 `HashEmbedder`。

同时生产环境禁止 `RAGBOT_INGESTION_MODE=inline`：摄取必须走 PostgreSQL durable queue + independent worker。

### 2.2 Durable ingestion queue

API 创建任务时首先在 PostgreSQL 保存 `pending` job。独立 worker：

1. 使用 `SELECT ... FOR UPDATE SKIP LOCKED` 原子 claim；
2. 记录 worker lease、heartbeat 与 attempts；
3. 周期性续租；
4. worker/API/节点异常退出后，过期 lease 可被其他 worker 回收；
5. 达到 `RAGBOT_WORKER_MAX_ATTEMPTS` 后任务进入 `failed`，不会无限重试。

这解决了旧的 API-process executor 在 rolling restart 时丢失执行上下文的问题。Compose 默认启动 worker；Helm 在 production render 时要求 `worker.enabled=true`。

### 2.3 API identity 不由客户端自行声明

`RAGBOT_API_KEY_PRINCIPALS` 将 API key 映射到 allowed tenant、stable user、groups、roles 和 optional admin。`/chat`、`/search`、OpenAI-compatible API、Source API 和 Ingest API 使用同一授权边界。

这是适合 service-to-service 的 v1 身份模型，不等价于完整企业 OIDC/OAuth2/SAML。

### 2.4 Source ingestion boundary

- Web / remote PDF / remote Git 默认拒绝 loopback、private、link-local、reserved 等地址；
- redirect 每跳重新验证；
- 可用 hostname allowlist 收窄 Web/PDF/Git；
- local_fs / local PDF / local Git 必须位于 `RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS`；
- Web/PDF 使用 bounded download；
- remote Git production path 限定 HTTPS；
- connector dependency 缺失时明确失败。

DNS 校验不能替代网络隔离；高安全环境仍应配置 egress firewall/service mesh/proxy allowlist。

### 2.5 Vector-index contract

Embedding API 实际输出维度必须严格等于 `QDRANT_DIM`，不再静默 truncate/zero-pad。模型或维度变化必须执行 reindex/migration。

### 2.6 CJK lexical retrieval baseline

PostgreSQL 仍使用内置 `simple` text-search configuration，但中文文本在写入时额外生成 overlapping CJK bigram lexemes；中文查询生成相同 lexical representation，再通过 GIN-backed FTS 检索。

`eval/fixtures/cjk_lexical.json` + `python -m eval.cjk_retrieval` 提供小型确定性回归语料。当前 CI 基线上该 corpus 结果为：

- **Recall@5 = 1.000**
- **MRR = 1.000**

CI floor 为 Recall@5 ≥ 0.90、MRR ≥ 0.80。这个结果只证明当前代表性 regression corpus 没有明显中文 lexical regression，不代表任意企业中文语料都达到相同质量。真实 domain corpus 应继续扩充，并用相同指标比较 PGroonga / pg_jieba / external lexical index；只有收益足以覆盖部署复杂度时再引入扩展。

### 2.7 Optional reranker availability

Reranker 是质量增强层，不是检索可用性的单点故障。provider 失败时保留 Qdrant + PostgreSQL FTS + RRF 结果。

### 2.8 Open-source license

仓库已采用 **Apache License 2.0**。相较无许可证状态，v1 开源发行现在具备明确的版权许可和 patent grant。后续贡献策略如需 CLA/DCO，应单独决定。

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
- [x] Node SDK typecheck
- [x] Docker Compose config validation，包括 worker service
- [x] Helm lint/default render
- [x] Helm production render with durable worker
- [x] Helm production config rejects missing worker
- [x] v1 production/security regression tests

PR #5 首轮代码验证中 Python 3.12 为 **255 passed / 4 skipped**；PostgreSQL integration 为 **4 passed**，CJK benchmark 为 **Recall@5=1.000 / MRR=1.000**。最终合并/发布仍以最终 head 和 release commit 的 GitHub Actions 结果为准。

### Gate B — Real dependency staging smoke

仓库已提供 `.github/workflows/staging-smoke.yml` 和 `eval/staging_smoke.py`。该 workflow 使用 production mode、PostgreSQL、Qdrant、independent worker 和真实 OpenAI-compatible credential，自动执行：

- [ ] actual production-compatible embedding provider/model
- [ ] actual production-compatible LLM provider/model
- [ ] local filesystem ingestion
- [ ] Web ingestion
- [ ] PDF ingestion
- [ ] Git ingestion
- [ ] hybrid `/search`
- [ ] Agentic `/chat`
- [ ] ACL negative isolation

运行前在 GitHub `staging` environment 配置 `STAGING_OPENAI_API_KEY`；可用 environment variables 覆盖 base URL、model 和 embedding dimension。**在该 workflow 实际成功运行之前，不应发布 `v1.0.0`。**

如果计划在 v1 production 启用 reranker，还应补一次该 provider 的 staging smoke。备份/恢复仍属于部署运维证据，不由普通 PR CI 自动完成。

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
- [ ] freeze the exact release commit and pass full CI on it
- [ ] create signed/annotated `v1.0.0` tag according to project policy
- [ ] publish GitHub Release with upgrade/migration/security notes

## 4. Remaining non-blocking v1.x roadmap

1. **Enterprise IAM**：OIDC/OAuth2/SAML、组织目录同步和 centrally managed RBAC。
2. **Larger Chinese/domain retrieval evaluation**：继续增加真实企业 corpus；以 Recall/MRR/NDCG/latency 比较 bigram baseline 与 PGroonga/pg_jieba/external lexical index。
3. **Cross-store activation semantics**：PostgreSQL/Qdrant 之间仍非分布式事务，可演进 staged source version + outbox/reconciler。
4. **Authoritative token accounting**：OpenAI-compatible adapter 的 usage 仍是 estimate。
5. **Defense-in-depth egress**：应用层 SSRF 防护仍需网络层策略配合。
6. **Queue observability / autoscaling**：可进一步增加 queue depth/oldest-job-age/lease-expiry 指标以及基于 backlog 的 worker autoscaling。

## 5. Recommended release sequence

1. 让 PR #5 的最终 head 完整 CI 绿色并合并。
2. 在 GitHub `staging` environment 配置真实 provider credential，执行 `Staging Smoke` workflow。
3. 保存 Gate B/Gate C evidence，特别是 ACL negative、备份/恢复与 egress policy。
4. Freeze API/contracts 和 evaluation baseline。
5. 从已验证 main 创建小型 `release/v1.0.0` PR，只做版本、changelog/date 和 immutable release metadata。
6. 在该 exact release commit 上再次运行完整 CI，并确认与 main 无冲突。
7. 仅从该 commit 创建 `v1.0.0` tag 和 GitHub Release。
