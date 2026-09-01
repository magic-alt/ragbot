# Changelog

All notable project changes intended for release are recorded here. Ragbot still reports package version `0.5.0`; the entries below remain **Unreleased** until a dedicated release commit promotes them to `1.0.0`.

## [Unreleased]

### Added

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
- v1 release/security regression tests.

### Changed

- FastAPI `/openapi.json` is the canonical HTTP API schema instead of a separately maintained static OpenAPI file;
- ingestion/query paths share one embedding contract and reject incompatible vector dimensions;
- PostgreSQL migrations are executed explicitly during Compose/Helm deployment rather than only on first database-volume creation;
- Source and Ingest APIs enforce tenant authorization when API-key principals are configured;
- `/search`, `/chat` and `/v1/chat/completions` derive ACL scope from trusted API-key principals;
- optional reranker failures fall back to RRF ordering instead of failing retrieval;
- Web/PDF downloads are bounded and redirects are revalidated;
- local source paths are constrained to configured roots in production;
- global metrics/cost/cache endpoints require an admin principal when scoped principals are enabled.

### Fixed

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
- cross-tenant Source/Ingest access possible with service-level API keys.

### Known limitations before v1.0

- ingestion jobs still execute in the API process and are not backed by a durable worker queue;
- PostgreSQL FTS currently uses `simple` text search and is not specialized for Chinese segmentation;
- API-key principals are service-to-service authorization, not full OIDC/SAML enterprise IAM;
- OpenAI-compatible `usage` is estimated until provider token accounting is propagated;
- repository currently has no LICENSE; the owner must choose one before an intended open-source v1 release.

## Release policy

A future `1.0.0` release should move the accepted Unreleased entries into a dated `## [1.0.0] - YYYY-MM-DD` section, update package/Helm/application metadata together, pass the full CI matrix on that exact commit, and only then create the `v1.0.0` tag and GitHub Release.
