# Ragbot System Design

## 1. Purpose and boundaries

Ragbot is an agent-facing, multi-tenant knowledge service. It turns PDF, web, Git, local filesystem and cloud/SaaS Sources into tenant-scoped retrievable evidence, combines semantic and lexical retrieval, and optionally invokes explicitly enabled Agent tools before synthesizing cited answers.

The production topology is intentionally split into an API/query plane and a durable ingestion worker plane. PostgreSQL is authoritative for Sources, Jobs, schedules, documents/chunks, ACL policies and lexical retrieval. Qdrant stores semantic vectors. Workers own ingestion execution through a PostgreSQL-backed lease/retry/DLQ contract.

Ragbot does not use LangChain, LangGraph or LlamaIndex for orchestration; the Agent graph, retrieval fusion, connector lifecycle and durability contracts are implemented directly in this repository.

## 2. Runtime architecture

```mermaid
flowchart LR
  C[CLI / Admin UI / SDK / Applications] -->|REST / SSE / OpenAI-shaped API| API[FastAPI API]

  API --> AUTH[Principal / tenant / groups / roles]
  AUTH --> AG[Agent graph]
  AG --> RET[Hybrid retriever]
  AG -. explicit capability .-> SQL[Isolated SQL tool]
  AG --> CODE[Code tools]
  AG --> WEB[Web tool]

  RET --> EMB[Embedder]
  RET --> Q[(Qdrant)]
  RET --> FTS[PostgreSQL FTS + CJK bigrams]
  FTS --> PG[(PostgreSQL)]

  API --> CP[Source / Job / schedule control plane]
  CP --> PG

  W[Durable ingestion workers] -->|claim / lease / heartbeat| PG
  W --> CONN[PDF / Web / Git / FS / S3 / Drive / Notion / Confluence]
  W --> EMB
  W --> PG
  W --> Q

  P[Prometheus] -->|scrape admin-protected /metrics| API
  API --> OTEL[OpenTelemetry / OTLP]
```

### API/query plane

`services/api/app/api.py` owns HTTP lifecycle, auth dependencies, `/chat`, `/search`, OpenAI-shaped chat completions, health/readiness, Prometheus export and router registration. `factory.py` is the composition root and decides between development fallbacks and durable external services.

Production refuses InMemory metadata/vector storage, HashEmbedder and inline ingestion. API keys are mapped to trusted principals with tenant IDs, user ID, groups, roles and optional global-admin capability.

### Agent layer

`services/api/app/agent/graph.py` owns the loop:

```text
route → tool action → synthesize → verify → optional next action → finalize
```

Event callbacks are transport-neutral and always close in `finally` so SSE consumers cannot hang when a tool/provider fails.

Tool availability and data authorization are separate concerns. In particular, SQL is a privileged data-plane capability and is fail-closed by default; a query merely looking like SQL must never grant access to Ragbot's internal PostgreSQL database.

### Retrieval layer

`Retriever` executes hybrid retrieval:

```text
query
  ├─ semantic embedding → Qdrant vector candidates
  └─ lexicalization     → PostgreSQL FTS candidates
                ↓
               RRF
                ↓
         optional reranker
                ↓
              top-k
```

Production lexical retrieval runs server-side through PostgreSQL GIN-backed FTS. CJK-heavy queries use Ragbot's bigram lexicalization path rather than relying only on whitespace tokenization. Tenant, ACL hash, source type, document, tag, path/URL prefix and time filters are applied before evidence reaches synthesis.

## 3. Durable ingestion and queue ownership

The API persists ingestion Jobs. Dedicated workers atomically claim executable Jobs from PostgreSQL, maintain leases with heartbeat, and apply bounded retry/backoff. Expired leases are reconciled; exhausted or permanent failures become `dead_lettered`.

```mermaid
stateDiagram-v2
  [*] --> pending
  pending --> running: atomic claim
  running --> completed: successful publish
  running --> pending: retryable failure + backoff
  running --> dead_lettered: permanent/exhausted failure
  running --> pending: expired lease + attempts remain
  running --> dead_lettered: expired lease + attempts exhausted
  failed --> pending: reconciliation / retryable
  failed --> dead_lettered: permanent/exhausted
```

The CLI treats `completed`, `failed` and `dead_lettered` as terminal states. A DLQ Job therefore fails `rag ingest --wait` immediately instead of being polled until timeout.

### Scheduled sync

Every worker may scan due Sources. Scheduled Job IDs are deterministic over Source ID plus schedule window; the repository uses atomic insert-if-absent. Multiple workers can therefore race without enqueueing duplicate work for one schedule window.

### SaaS incremental refresh

Drive, Notion and Confluence use metadata-first refresh. Remote metadata is compared against the previous indexed snapshot; unchanged documents reuse existing chunks/vectors and skip body download and embedding. Changed/new documents continue through the normal replacement-oriented ingestion path.

## 4. Source generation fencing

Source mutations and deletion need to invalidate older queued/running work. Ragbot uses a durable lifecycle token derived from Source timestamps and stores the token in each Job's `stats.source_generation` at submission time.

Worker behavior:

```text
Job.source_generation
        │
        ▼
compare current Source generation
        ├─ mismatch before execution → dead_lettered
        └─ match
             ↓
        connector/fetch
             ↓
        fence check before write
             ↓
        PG/Qdrant publish + stale cleanup
             ↓
        fence check before completion
```

Source deletion is **tombstone-first**:

```text
mark Source deleted + advance updated_at generation
        ↓
fences old queued/running Jobs
        ↓
purge PostgreSQL/Qdrant knowledge
```

If an in-flight pipeline observes a fence failure after work has started, the failure is permanent rather than retryable. For deleted Sources, the pipeline also attempts a cleanup purge before returning failure.

This fencing prevents the common "purge, then old worker writes the Source back" race. It is still not a distributed transaction or atomic generation cutover across PostgreSQL and Qdrant. A future version that requires strict zero-partial-view semantics should stage a complete generation and atomically activate it, with an outbox/reconciler for vector side effects.

## 5. Ingestion data flow

```mermaid
sequenceDiagram
  participant API
  participant PG as PostgreSQL
  participant W as Worker
  participant C as Connector
  participant E as Embedder
  participant Q as Qdrant

  API->>PG: persist Job + immutable Source config + source_generation
  W->>PG: claim Job / lease
  W->>PG: validate current Source generation
  W->>PG: read previous source documents/chunks
  W->>C: metadata/fetch/chunk
  W->>W: normalize + dedup + reuse unchanged chunks
  W->>PG: revalidate Source generation
  W->>E: embed changed/new chunks
  W->>PG: upsert documents/chunks
  W->>Q: upsert changed/new vectors
  W->>Q: delete stale points/doc vectors
  W->>PG: delete stale chunks/documents
  W->>PG: final Source-generation check
  W->>PG: complete Job
```

### Replacement semantics

Re-ingestion is Source reconciliation, not repository-wide checksum deduplication. Identical text in two documents or tenants remains independent evidence. Unchanged chunks preserve chunk/point IDs and are not re-embedded when content and retrieval metadata are unchanged. New/changed chunks are written before stale content is deleted to preserve a retryable last-good view where possible.

PostgreSQL and Qdrant do not participate in a distributed transaction. Backup/restore likewise requires both stores, and strict point-in-time consistency requires quiescing ingestion or coordinated infrastructure snapshots.

## 6. Storage responsibilities

### PostgreSQL control-plane database

`POSTGRES_DSN` is reserved for Ragbot internal durable state:

- Sources and sync schedules;
- ingestion Jobs, leases, retry/DLQ state;
- Documents and Chunks;
- ACL policies;
- PostgreSQL lexical/FTS state;
- related control-plane metadata.

This database is **not** the Agent SQL query surface.

### Optional Agent SQL database

SQL is disabled by default:

```dotenv
RAGBOT_SQL_TOOL_ENABLED=false
```

Production enablement requires:

```dotenv
RAGBOT_SQL_TOOL_ENABLED=true
RAGBOT_SQL_DSN=postgresql://read_only_user:***@analytics-db/analytics
RAGBOT_SQL_ALLOWED_SCHEMAS=rag_views,analytics
```

Production startup rejects a SQL DSN identical to `POSTGRES_DSN`. `PostgresSqlEngine` additionally enforces single read-only SELECT statements, transaction read-only mode, timeout and row limits. Those application checks are defense-in-depth only; the real security boundary must be the database identity itself: dedicated read-only role, least-privilege views/schema grants and, for multi-tenant business data, RLS or tenant-safe views.

### Qdrant

Qdrant stores semantic vector points and retrieval payload metadata. The configured vector dimension must match the embedder dimension. Ragbot fails fast on dimension mismatches rather than querying an incompatible index.

### In-memory implementations

InMemoryRepo, InMemoryQdrant and HashEmbedder are development/test conveniences. They are process-local and are rejected by production composition.

## 7. Security invariants

1. API credentials resolve to a trusted principal before tenant/user claims are used.
2. RAG retrieval always carries tenant scope and ACL scope into vector/lexical filtering before synthesis.
3. Source connector secrets are references to deployment secrets; cloud tokens/passwords are not stored inline in Source config.
4. Local filesystem/Git/PDF reads are constrained by configured roots in production.
5. Remote web/source fetching rejects credential-bearing URLs and private/loopback/link-local destinations by default; redirects are revalidated.
6. Agent SQL is disabled by default and must never reuse the Ragbot control-plane database in production.
7. Source mutations fence older Jobs; deletion tombstones before purging knowledge.
8. Changing embedding model/dimension requires a compatible collection/reindex operation.

RBAC currently maps most read surfaces to tenant-scoped authenticated principals, mutations to `operator`/`owner`, and cross-tenant/global operations to `admin`. `owner` is presently an operator superset rather than a broad independent tenant-administration subsystem.

## 8. OpenAI-shaped compatibility

`POST /v1/chat/completions` preserves:

- system messages;
- prior user/assistant turns as conversation context;
- the last non-empty user turn as the current retrieval query;
- `temperature`;
- `max_tokens`;
- non-stream and SSE transport.

Conversation context helps resolve intent/references but is not treated as evidence; factual synthesis still requires retrieved/tool evidence and citations.

Current limitations:

- usage token counts are estimated rather than tokenizer/provider authoritative;
- streaming sends chunks after the Agent final answer exists rather than exposing provider-native token-by-token generation;
- the endpoint does not claim complete OpenAI API field/tool-call equivalence.

## 9. Observability

### Health/readiness

`/admin/health` is liveness only. `/admin/ready` checks configured repository and vector-store readiness and returns 503 when dependencies are unavailable.

### Prometheus

`GET /metrics` exposes Prometheus exposition format and requires a global-admin API principal. Metrics include:

- HTTP requests and latency histogram;
- ingestion Job counts by state;
- oldest pending Job age;
- stale running leases;
- Source counts/state;
- process-local Agent citation coverage, retrieval hit rate, tool failure ratio and latency gauges.

HTTP counters/histograms are naturally per process and Prometheus aggregates across replicas. Some Agent quality gauges currently derive from Ragbot's bounded in-process request history; production dashboards should interpret them as per-replica rolling diagnostics unless/until event-level metrics are exported directly.

### Tracing

OpenTelemetry tracing can export Agent stage/tool spans over OTLP. Request IDs are propagated through application logs and responses.

## 10. Deployment and recovery

Docker Compose provides the local durable topology. Helm provides API/worker Deployments, readiness/liveness, rolling updates, optional API HPA and optional KEDA worker scaling. KEDA scales workers from PostgreSQL queue depth, including ready pending Jobs and expired running leases.

Production Helm refuses unsafe process-local storage and requires the durable worker path. PostgreSQL migrations run through the explicit migration runner rather than relying only on first-boot container initialization.

Disaster recovery covers both PostgreSQL and Qdrant: PostgreSQL custom-format dump plus Qdrant collection snapshot, SHA-256 manifest, destructive restore and post-restore validation. The backup is not a distributed transaction; strict consistency requires ingestion quiescence or coordinated infrastructure snapshots.

## 11. API and contract strategy

FastAPI/Pydantic is the source of truth for HTTP OpenAPI. Use `/openapi.json`, `/docs` or `scripts/export_openapi.py`. Shared Agent/tool domain types remain under `contracts/`, and the Node client is typechecked in CI.

## 12. Remaining architecture work

The highest-value remaining hardening after the current production-safety boundary is:

1. **Atomic source-version activation/outbox** — stage complete generations and switch an active pointer only after PG/Qdrant side effects are ready.
2. **Broader tool capability policy** — extend explicit per-principal/per-tenant capability controls to code/web tools, not only SQL's fail-closed composition.
3. **Database-native SQL tenancy policy** — standardize RLS/tenant-safe views for deployments that expose multi-tenant business SQL data.
4. **Native streaming/token accounting** — surface provider streaming and authoritative usage where supported.
5. **Distributed quality telemetry** — export event-level retrieval/citation/tool metrics rather than relying partly on process-local rolling history.
6. **Generated SDK contracts** — once the public API is stable, generate clients from runtime OpenAPI rather than maintaining duplicated DTOs manually.
