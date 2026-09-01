# Ragbot v1.0 Release Readiness

本文档把 README 的能力声明映射到当前实现、验证方式和 v1.0 发布门槛。它回答两个不同问题：

1. **功能是否已经实现？** —— 看 Capability Matrix。
2. **是否已经可以打 `v1.0.0` tag？** —— 看 Release Gates。

当前 Python package / Helm metadata 仍为 `0.5.0`。本轮工作只做 release hardening；版本号、tag 和 GitHub Release 应在所有 release gate 满足后单独执行。

## 1. Capability Matrix

| README 能力 | 状态 | 主要实现 | v1 验证/约束 |
|---|---|---|---|
| PDF / Web / Git / local filesystem 摄取 | **已实现** | `services/worker/jobs/*`, `services/worker/pipeline.py` | Source → chunk → dedup → shared embedder → Qdrant；重摄取会复用未变化 chunk 并清理 stale SQL/vector 数据 |
| Qdrant vector search | **已实现** | `retrieval/qdrant.py`, `retrieval/service.py` | tenant/source/doc/tag/path/url/time/ACL filter；collection dimension 校验 |
| PostgreSQL FTS | **已实现，有明确限制** | `storage/pg_repo.py`, `retrieval/pg_fts.py` | 使用 PostgreSQL native FTS；当前 `simple` config 对中文分词质量不是最优 |
| RRF hybrid fusion | **已实现** | `retrieval/rerank.py`, `retrieval/service.py` | vector + lexical ranking 融合 |
| optional cross-encoder rerank | **已实现** | `retrieval/cross_encoder.py` | optional service 失败时回退到 RRF，不使检索整体失败 |
| Agentic RAG graph | **已实现** | `agent/graph.py`, `agent/nodes/*` | route → tool → synthesize → verify → retry/finalize；支持 doc/sql/code/web 路由 |
| Tenant / user / ACL | **已实现为 API-key principal 模型** | `auth/acl.py`, `auth/principal.py` | API key 可绑定 tenant/user/groups/roles；HTTP 入口不能通过 payload/header 扩大授权范围 |
| OpenAI-compatible LLM | **已实现** | `llm/client.py`, `llm/provider.py` | `/v1/chat/completions` adapter 与内部 provider |
| Ollama LLM adapter | **已实现** | `llm/ollama.py` | Ollama OpenAI-compatible chat path；不等于自动配置 Ollama embedding |
| `/search` | **已实现** | `routes/search.py` | pure retrieval tool surface，带 trusted tenant/ACL filtering |
| `/chat` + SSE | **已实现** | `api.py`, callbacks, graph | streaming/non-streaming 共用身份与约束语义；错误路径保证 stream 终止 |
| `/v1/chat/completions` | **已实现** | `routes/openai_compat.py` | stream/non-stream；usage 当前为明确标记的 estimate |
| Source/Ingest API | **已实现** | `routes/sources.py`, `routes/ingest.py` | CRUD/job 查询按 API principal tenant 隔离 |
| Metrics / tracing / cache | **已实现** | `observability/*`, `cache/*` | global metrics/cost/cache 仅允许 admin principal；liveness/readiness 分离 |
| Docker Compose | **已实现** | root + `infra/docker` Compose | migration service、Postgres、Qdrant、可选 Ollama/Jaeger；生产模式 fail-fast |
| Helm | **已实现** | `infra/helm/ragbot` | migration initContainer、Ingress、HPA、readiness、多副本 shared-store guard、可选 source volume mounts |
| Node client | **已实现基础 SDK** | `packages/node-client` | TypeScript strict typecheck 进入 CI |
| Evaluation | **已实现基础框架** | `eval/` | 可运行 retrieval/agent evaluation；正式 v1 应保存真实 corpus baseline |

## 2. v1.0 本轮已经封闭的 release blockers

### 2.1 禁止 production 静默降级

`RAGBOT_ENV=production` 时启动必须具备：

- PostgreSQL DSN；
- Qdrant URL；
- semantic embedding model；
- embedding credential；
- API keys；
- API-key principal mappings。

生产模式不会再静默落到 `InMemoryRepo`、`InMemoryQdrant` 或 `HashEmbedder`。

### 2.2 API identity 不再由客户端自行声明

`RAGBOT_API_KEY_PRINCIPALS` 把 API key 映射到：

- allowed `tenant_ids`；
- stable `user_id`；
- `groups`；
- `roles`；
- optional `admin`。

`/chat`、`/search`、OpenAI-compatible API、Source API 和 Ingest API 都使用同一授权边界。

这是一套适合 service-to-service 的 v1 身份模型，不等价于完整企业 IAM/OIDC/SSO。

### 2.3 Source ingestion boundary

生产 Source 不应拥有任意网络和文件系统访问能力：

- Web / remote PDF / remote Git 默认拒绝 loopback、private、link-local、reserved 等地址；
- redirect 每跳重新验证；
- 可用独立 hostname allowlist 收窄 Web/PDF/Git；
- local_fs / local PDF / local Git 必须位于 `RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS`；
- Web/PDF 下载使用 hard byte limit；
- Git remote production path 限定为 HTTPS；
- 缺失 connector runtime dependency 时明确失败，而不是返回 placeholder。

注意：DNS 校验与实际网络连接之间仍存在通用 DNS rebinding/TOCTOU 风险。高安全环境应同时使用 egress firewall / service mesh / proxy allowlist，而不是只依赖应用层校验。

### 2.4 Vector-index contract

Embedding API 实际输出维度必须严格等于 `QDRANT_DIM`。Ragbot 不再把错误维度向量静默截断或补零。

模型或维度变化必须执行 reindex/migration。

### 2.5 Optional reranker availability

Reranker 是质量增强层，不是检索可用性的单点故障。Reranker endpoint/provider 失败时保留 Qdrant + PostgreSQL FTS + RRF 结果。

## 3. v1.0 Release Gates

以下条件全部满足后才建议把 package/chart 从 `0.5.0` 提升到 `1.0.0`。

### Gate A — Automated CI

- [x] Python 3.10 tests
- [x] Python 3.12 tests
- [x] dependency consistency / compileall
- [x] PostgreSQL full migration chain
- [x] PostgreSQL native FTS integration smoke
- [x] Node SDK typecheck
- [x] Docker Compose config validation
- [x] Helm lint + render
- [x] v1 production/security regression tests

当前 hardening 首轮 CI 已达到 249 passed / 2 skipped；最终发布以 release commit 对应 CI 为准，而不是固定测试数字。

### Gate B — Real dependency smoke

发布前至少在 staging 环境用实际服务跑一次：

- [ ] external PostgreSQL
- [ ] external Qdrant
- [ ] intended production embedding provider/model
- [ ] intended production LLM provider/model
- [ ] ingest one PDF, one Web source, one Git source, one local_fs source
- [ ] verify vector + FTS + RRF retrieval
- [ ] verify ACL negative test: tenant/user A 无法读取 tenant/user B
- [ ] exercise both `/chat` SSE and `/v1/chat/completions` streaming
- [ ] reranker enabled smoke if v1 deployment plans to enable it
- [ ] backup/restore smoke for PostgreSQL and Qdrant snapshots

CI cannot substitute for these provider/network-specific checks.

### Gate C — Security / operations

- [ ] `RAGBOT_ENV=production`
- [ ] API keys stored in secret manager/Kubernetes Secret, not Git
- [ ] every API key has principal mapping
- [ ] Web/PDF/Git allowlists or controlled egress policy defined
- [ ] local source roots explicitly mounted read-only where practical
- [ ] TLS / ingress auth / rate limiting policy defined
- [ ] PostgreSQL/Qdrant are not publicly exposed
- [ ] immutable container image tag/digest selected
- [ ] rollback, database migration and embedding reindex runbook reviewed

### Gate D — Project/release metadata

- [ ] repository owner chooses and adds an appropriate `LICENSE`
- [ ] create/update `CHANGELOG.md` release section
- [ ] bump Python package version to `1.0.0`
- [ ] bump FastAPI/Helm chart/app/image metadata consistently
- [ ] create signed/annotated `v1.0.0` tag according to project policy
- [ ] publish GitHub Release with upgrade/migration/security notes

**License is a release blocker for an intended open-source v1.0.** Public repository visibility alone does not grant an open-source license; this choice cannot be safely automated on behalf of the owner.

## 4. Explicit non-blocking limitations / v1.x roadmap

These are real limitations but do not need to be hidden behind a false “complete” claim:

1. **Ingestion execution is not durable.** Jobs currently execute through the API process executor. A rolling restart can interrupt work. If v1 SLA promises reliable background ingestion, move this item into the blocking gates and introduce a durable queue/worker before release.
2. **Enterprise IAM is service-key based.** OIDC/OAuth2/SAML, organization directory sync and centrally managed RBAC remain v1.x work.
3. **Chinese lexical retrieval can improve.** PostgreSQL `simple` FTS is operationally correct but not a specialized Chinese tokenizer. Consider pg_jieba/PGroonga/external lexical index after measuring corpus quality.
4. **No distributed transaction across PostgreSQL and Qdrant.** Replacement ingestion minimizes stale state, but stronger cross-store activation semantics can use staged source versions + outbox/reconciler.
5. **Usage accounting in OpenAI-compatible adapter is estimated.** Authoritative token billing requires provider-level usage propagation.
6. **Web security should be defense-in-depth.** Application URL validation should be combined with network egress policy in sensitive environments.

## 5. Recommended release sequence

1. Merge architecture/security hardening with all CI green.
2. Configure a staging deployment with the intended v1 providers and credentials.
3. Complete Gate B and Gate C evidence.
4. Decide/add LICENSE.
5. Freeze API/contracts and capture evaluation baseline.
6. Create a small release-only PR for `0.5.0 → 1.0.0`, changelog finalization and immutable image/tag references.
7. Run the same complete CI on the release commit.
8. Tag and publish v1.0.0 only from that validated commit.
