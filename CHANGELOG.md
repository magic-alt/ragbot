# Changelog

All notable project changes intended for release are recorded here. Ragbot still reports package version `0.5.0`; the entries below remain **Unreleased** until a dedicated release commit promotes them to `1.0.0`.

## [Unreleased]

### Added

- high-level `POST /ingest/quick` product API that creates/reuses a Source and submits ingestion in one call;
- `POST /ingest/batch` for manifest-style onboarding of up to 100 knowledge sources per request;
- stable Source identity from tenant + source type + canonicalized location;
- active-ingestion deduplication and explicit deterministic ingestion idempotency keys;
- product CLI workflow with automatic source-type inference, ingestion progress polling and `--wait`;
- `rag import` JSON-manifest knowledge-base bootstrap command;
- `rag doctor` deployment liveness/readiness command;
- 60-second product quickstart and example multi-source manifest;
- reusable ingestion queue helpers shared by low-level and product APIs;
- real PostgreSQL + Qdrant 1000-PDF capacity/integration benchmark and benchmark documentation;
- durable SQL migration runner with advisory locking and migration history;
- native PostgreSQL FTS integration in the production repository path;
- replacement-oriented source ingestion and stale PostgreSQL/Qdrant cleanup;
- explicit Qdrant health/dimension validation and vector deletion support;
- Node client TypeScript build/typecheck configuration;
- Helm Ingress/HPA templates, migration initContainer and source-volume extension points;
- system design, API, deployment and v1 release-readiness documentation;
- API-key principal model for tenant/user/group/role scoped service identities;
- production runtime validation through `RAGBOT_ENV=production`;
- Web/PDF/Git network-source policy and local source-root policy;
- Apache License 2.0;
- PostgreSQL-backed durable ingestion queue with atomic worker claims, leases, heartbeat, crash recovery and bounded attempts;
- independent ingestion worker deployments for Docker Compose and Helm;
- CJK bigram lexical representation on top of PostgreSQL `simple` FTS;
- deterministic PostgreSQL CJK retrieval regression corpus with Recall@5/MRR release floors;
- manual production-style staging smoke workflow for real LLM/embedding provider credentials, PostgreSQL and Qdrant;
- v1 release/security/queue/CJK regression tests.

### Changed

- the default onboarding workflow is now Quick Import/CLI instead of requiring callers to manually orchestrate Source creation followed by Job creation;
- repeated product bootstrap calls reuse an existing Source and same-config pending/running Job by default;
- durable ingestion Jobs execute the connector `source_type/source_config` snapshot captured at enqueue time, rather than silently switching to a later mutable Source config;
- local CLI ingestion explicitly uses the configured shared embedder, preserving the same embedding contract as retrieval;
- FastAPI `/openapi.json` is the canonical HTTP API schema instead of a separately maintained static OpenAPI file;
- ingestion/query paths share one embedding contract and reject incompatible vector dimensions;
- PostgreSQL migrations are executed explicitly during Compose/Helm deployment rather than only on first database-volume creation;
- PostgreSQL-backed deployments enqueue ingestion jobs for independent workers instead of executing them in the API process;
- production mode rejects inline ingestion and Helm production renders require the durable worker;
- Source and Ingest APIs enforce tenant authorization when API-key principals are configured;
- `/search`, `/chat` and `/v1/chat/completions` derive ACL scope from trusted API-key principals;
- optional reranker failures fall back to RRF ordering instead of failing retrieval;
- Web/PDF downloads are bounded and redirects are revalidated;
- local source paths are constrained to configured roots in production;
- global metrics/cost/cache endpoints require an admin principal when scoped principals are enabled;
- unchanged chunks carry a lexical representation version so retrieval-format upgrades trigger one controlled rewrite/reindex pass.

### Fixed

- strict quick-import idempotency can no longer be combined with non-reusable Source identity;
- Quick Import no longer mutates/reuses an active Source Job when the newly requested connector configuration differs;
- batch Quick Import no longer returns raw unexpected exception text to clients;
- queued durable Jobs are no longer redirected by Source config edits made after submission;
- hybrid RRF modality crowd-out that could suppress an exact lexical candidate behind the vector candidate window;
- invalid single-marker identity assumptions in the deterministic 1000-PDF hash-embedding benchmark;
- stable deterministic hash embeddings for development/testing;
- ingestion/query embedding-dimension drift;
- stale chunks/vectors/documents after replacement ingestion;
- PostgreSQL row/JSONB/array/schema adaptation issues;
- migration ordering/schema alignment problems;
- SSE failure paths that could leave a stream waiting indefinitely;
- Node client missing shared types/build metadata;
- previously advertised unsupported source types and contract drift;
- unsafe remote-source SSRF exposure to loopback/private/link-local destinations;
- silent embedding vector truncation/zero-padding;
- cross-tenant Source/Ingest access possible with service-level API keys;
- ingestion jobs being lost when an API process restarted mid-execution;
- poor Chinese lexical recall caused by treating long CJK strings as unsplit `simple`-FTS tokens.

### Remaining release gates

- run the manual `Staging Smoke` workflow successfully with the intended production-compatible LLM/embedding credential;
- record staging evidence for backup/restore, network policy and any reranker that will be enabled in production;
- create a release-only change for `0.5.0 -> 1.0.0`, synchronize Python/FastAPI/Helm/image metadata, and publish the tag/release only from the exact CI-validated release commit.

### Non-blocking v1.x roadmap

- web/admin UI for source catalog, ingestion progress, retrieval inspection and evaluation;
- scheduled/synchronized connectors for frequently changing knowledge sources;
- OIDC/OAuth2/SAML and centrally managed enterprise IAM beyond service-to-service API-key principals;
- larger customer/domain CJK corpora and optional PGroonga/pg_jieba/external lexical-index comparison when they materially outperform the built-in bigram baseline;
- stronger PostgreSQL/Qdrant cross-store activation semantics such as outbox/reconciler or staged source versions;
- authoritative provider token accounting for the OpenAI-compatible `usage` object.

## Release policy

A future `1.0.0` release should move the accepted Unreleased entries into a dated `## [1.0.0] - YYYY-MM-DD` section, update package/Helm/application metadata together, pass the full CI matrix on that exact commit, complete the real-provider staging gate, and only then create the `v1.0.0` tag and GitHub Release.
