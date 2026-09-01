# Ragbot Roadmap

> Package metadata: `0.5.0` · Development status: Milestone E implemented on `main`

Ragbot 的核心定位是 **可被其他 Agent 复用的知识服务层**：负责把本地/企业数据转成受 ACL 约束、可追溯、可评测的检索证据，并通过稳定 API 提供给 Agent、CLI、SDK 与 IDE。

## 1. Current architecture

```text
Sources
  ├─ local filesystem
  ├─ PDF
  ├─ Web
  ├─ Git repository
  └─ database metadata / SQL access
        │
        ▼
Ingestion Pipeline
  connector → chunk → dedup → shared Embedder → Qdrant
        │                                  │
        └──────── metadata / jobs ─────────┤
                                           ▼
                                    PostgreSQL / Repo

Client / Other Agent
        │
        ├─ POST /search  ───────────────┐
        ├─ POST /chat                   │
        └─ OpenAI-compatible endpoint   │
                                        ▼
Agent: route → retrieve/sql/code/web → synthesize → verify → finalize
                       │
                       ├─ Qdrant vector search
                       ├─ PostgreSQL FTS
                       └─ RRF + optional reranker
```

### Core modules

- `services/api/app/agent/`: Agent state, graph, nodes, reliability, callbacks, session/context.
- `services/api/app/retrieval/`: Qdrant adapter, embedder, PostgreSQL FTS, RRF and cross-encoder reranking.
- `services/api/app/storage/`: in-memory repo, storage protocol, PostgresRepo and models.
- `services/worker/`: source connectors and ingestion pipeline.
- `contracts/`: Python/TypeScript/OpenAPI/tool contracts.
- `eval/`: dataset, runner, CI gate and RAGAS integration.
- `infra/`: Docker, SQL migrations, Qdrant bootstrap and Helm.

## 2. Completed milestones

| Milestone | Outcome | Current implementation |
|---|---|---|
| A | Usable Agent API | provider abstraction, reliability, SSE, OpenAI-compatible endpoint |
| B | Enterprise data ingestion | Source CRUD, connectors, ingestion jobs, ACL support |
| C | Code/IDE workflow | repo/code tools, open-file/patch/error explanation, CLI/context |
| D | Operability | tracing/metrics primitives, evaluation, cache primitives, Helm |
| E | Retrieval quality + persistence | API Embedder, deterministic hash fallback, reranker, async agent path, PostgresRepo, migration/index work, larger eval dataset |

The historical fixed test counts in earlier documentation have been removed. CI is the source of truth for current test status.

## 3. RAG invariants

These are architectural contracts that future work must preserve:

1. **One embedding space per index.** Ingestion and retrieval use the same embedder and vector dimension.
2. **Stable document identity.** Chunks and Document metadata share the same deterministic `doc_id`.
3. **Reindex on embedding changes.** Model/dimension changes are data migrations, not simple environment changes.
4. **Evidence-first answers.** Synthesis receives evidence with citations; verification can reject insufficient evidence.
5. **ACL before synthesis.** Tenant/user constraints are applied during retrieval, not only after results are returned.
6. **Contracts move together.** API/tool changes update Python, TypeScript and OpenAPI/schema surfaces together.

## 4. Next: Milestone F — durable knowledge ingestion

Goal: turn the current in-process ingestion implementation into a production data plane.

- Replace `run_in_executor` background work with a durable queue/worker deployment.
- Persist retry state, idempotency key, heartbeat and cancellation state.
- Add source-level incremental sync/checkpoints instead of full rescans.
- Add ingestion concurrency/rate controls and per-tenant quotas.
- Add structured parser quality metrics: empty-page rate, chunk distribution, parse failures.
- Define and implement the currently reserved connectors (for example email) before advertising them as production source types.
- Add object-storage staging for large remote documents.

Exit criteria: process restarts do not lose accepted jobs; repeated sync is idempotent; failures are observable and retryable.

## 5. Milestone G — knowledge service for other agents

Goal: make Ragbot a clean lower-level knowledge substrate rather than requiring every consumer to understand internal endpoints.

- Stabilize `/search` as the primary low-latency knowledge API with versioned schemas.
- Add a dedicated tool adapter/MCP-style surface for `search`, `fetch citation`, `list sources`, and constrained ingest operations.
- Add service-to-service authentication and scoped keys.
- Publish maintained Python and Node SDKs generated/validated against contracts.
- Add request budgets (`top_k`, max evidence bytes, latency budget) for agent orchestration.
- Provide citation-fetch endpoints so downstream agents can verify source context independently.
- Define error taxonomy and retry semantics for tool callers.

Exit criteria: another Agent can integrate Ragbot as a knowledge tool without importing Ragbot internals.

## 6. Milestone H — retrieval quality and lifecycle

- Model-aware chunking for prose/code/table content.
- Hybrid retrieval calibration beyond fixed RRF parameters.
- Query rewrite/decomposition and multi-hop retrieval evaluation.
- First-class local embedding deployment and batch throughput tuning.
- Collection alias/version management for zero-downtime reindex.
- Hard-negative datasets and regression thresholds for Recall@K/MRR/citation coverage.
- Reranker latency/cost controls and fallback strategy.
- Duplicate/near-duplicate detection beyond exact checksum.

Exit criteria: retrieval changes are accepted/rejected by measured quality and latency budgets rather than subjective examples.

## 7. Milestone I — production security and operations

- JWT/OIDC/RBAC beyond shared API keys.
- Tenant-scoped audit trails and retention policy.
- Secret-manager integration and key rotation.
- Network policy, TLS, rate limiting and abuse controls.
- Postgres/Qdrant backup/restore and disaster-recovery runbooks.
- Pinned production image versions and automated dependency/security updates.
- SLOs for search latency, ingestion success, availability and retrieval quality.
- Required CI checks + protected `main` branch/ruleset.

## 8. Known gaps / technical debt

- Ingestion jobs currently execute in-process/background executor; this is not a durable distributed worker model.
- `HashEmbedder` is useful for deterministic development tests but has no semantic quality guarantee.
- Model-router primitives exist, but environment-driven fast/strong routing is not yet the default service factory path.
- Some cache/observability primitives are ahead of their end-to-end production wiring.
- Docker Compose currently uses several upstream `latest` service images; production deployments should pin versions after validation.
- The repository has no open-source LICENSE yet; license selection is a project governance decision.

## 9. Release discipline

A release should include:

1. CI green on supported Python versions;
2. schema/contract compatibility notes;
3. storage migration instructions;
4. reindex instructions for embedding changes;
5. upgrade/rollback notes for Docker/Helm;
6. version bump and changelog/release notes as a separate release concern.

This roadmap intentionally separates completed implementation from future plans so README/architecture documentation does not present planned work as already available.
