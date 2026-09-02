# Changelog

All notable project changes intended for release are recorded here. Ragbot still reports package version `0.5.0`; the entries below remain **Unreleased** until a dedicated release-only change promotes them to `1.0.0`.

## [Unreleased]

### Added

- Quick Import (`POST /ingest/quick`) and batch onboarding (`POST /ingest/batch`);
- stable Source identity, explicit idempotency keys, CLI `rag ingest --wait`, `rag import`, and `rag doctor`;
- PostgreSQL-backed durable ingestion queue with worker claim, lease, heartbeat, crash recovery, retry/backoff, scheduler, and reconciliation;
- explicit `dead_lettered` Job state, `failure_class`, DLQ Requeue, and global queue reconcile endpoint;
- shared provider HTTP reliability layer with `Retry-After`, bounded exponential backoff, and no blind retry of permanent 4xx responses;
- reader/operator/owner/admin RBAC on tenant mutation and global operations;
- `/catalog/session` non-secret capability endpoint for product UIs;
- built-in zero-build Admin UI with Source Catalog, queue health, scheduled sync, failed Retry, DLQ Requeue, and principal capability guidance;
- S3/MinIO, Google Drive, Notion, and Confluence connectors using worker-only secret references;
- metadata-first Drive/Notion/Confluence incremental reuse for unchanged remote documents;
- Docker Compose worker-only credential env files and Helm `worker.extraEnv` / `worker.extraEnvFrom`;
- optional KEDA PostgreSQL backlog autoscaling for ingestion workers;
- PostgreSQL native FTS plus CJK bigram lexical representation and Recall@5/MRR regression gates;
- deterministic 1000-PDF PostgreSQL/Qdrant capacity/integration benchmark;
- Apache License 2.0;
- executable PostgreSQL + Qdrant backup/restore scripts with SHA-256 manifest validation;
- CI `Backup + restore smoke` that performs seed → backup → destroy → restore → verify against PostgreSQL 16 and Qdrant v1.19.0;
- `docs/DISASTER_RECOVERY.md` plus production/admin/deployment/v1 readiness runbooks.

### Changed

- product onboarding now defaults to Quick Import instead of manually orchestrating Source creation and Job submission;
- queued Jobs execute immutable connector snapshots captured at submission time;
- production mode rejects inline ingestion and in-memory/hash fallbacks;
- Source/Ingest mutation routes require operator/owner capability when scoped principals are configured;
- Admin UI write controls follow `/catalog/session`, while API RBAC remains authoritative;
- PostgreSQL-backed ingestion failure handling now uses two layers: provider request retry followed by whole-ingestion durable retry;
- exhausted/permanent work is separated from ordinary failure state through DLQ;
- both Docker Compose variants expose the same retry/reconcile/provider settings and pin Qdrant to v1.19.0;
- Helm worker values/template expose durable retry, reconciliation, scheduler, and provider backoff settings;
- hybrid retrieval uses Qdrant vector + PostgreSQL lexical/CJK + balanced RRF with optional reranking;
- reranker failure remains fail-open to the base hybrid ordering;
- cloud connector secrets remain worker-only and never become Source credential values.

### Fixed

- worker helper compatibility regression after introducing `max_attempts`;
- Quick Import active-Job/config drift;
- queued Job redirection after Source config edits;
- hybrid RRF modality crowd-out;
- Qdrant point-ID/logical chunk-ID mismatch;
- deterministic 1000-PDF marker collisions;
- ingestion/query embedding contract drift;
- stale SQL/vector state after replacement ingestion;
- remote-source SSRF and redirect gaps;
- cross-tenant Source/Ingest access under service-level API keys;
- ingestion loss on API restart;
- long CJK-token lexical recall regression.

### Known v1 limitations

- Drive/Notion/Confluence synchronization is metadata-first and still enumerates the configured remote scope; provider change-feed/delta cursors are not yet implemented;
- API-key principal IAM is service-oriented and is not full OIDC/SAML enterprise directory integration;
- PostgreSQL and Qdrant do not share a distributed transaction;
- CJK bigrams are a baseline rather than a universal Chinese search solution;
- network-layer egress controls are still required in addition to application connector validation.

### Remaining release gates

- exact-head production-hardening CI must remain green after final docs/configuration changes;
- run the real core staging workflow with intended production LLM/embedding credentials;
- run SaaS staging for every connector enabled at launch;
- record production network, secret, image, backup retention, RPO/RTO, and rollback evidence;
- create a separate `release/v1.0.0` change that synchronizes Python/FastAPI/Helm/image metadata;
- run CI/staging/recovery on the exact release SHA before creating the `v1.0.0` tag and GitHub Release.

## Release policy

A future `1.0.0` release should move accepted Unreleased entries into a dated `## [1.0.0] - YYYY-MM-DD` section, update all release metadata together, pass the complete CI matrix on that exact commit, complete real-provider staging and recovery gates, and only then create the `v1.0.0` tag and GitHub Release.
