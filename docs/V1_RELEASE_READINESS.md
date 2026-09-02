# Ragbot v1.0 Release Readiness

This document separates **implemented product capability** from **evidence required to publish `v1.0.0`**. Package/FastAPI/Helm metadata intentionally remains `0.5.0` until all blocking gates are satisfied on an exact release commit.

## 1. Current product capability

| Capability | Status | Release significance |
| --- | --- | --- |
| Quick Import / manifest batch / CLI `--wait` | implemented | fast knowledge-base bootstrap |
| PDF/Web/Git/local/S3/Drive/Notion/Confluence | implemented | production connector set |
| PostgreSQL durable ingestion queue | implemented | API restarts do not lose pending work |
| lease / heartbeat / crash reclaim | implemented | multi-worker recovery |
| provider short retry / `Retry-After` | implemented | protects against transient 429/5xx/network faults |
| whole-ingestion durable retry/backoff | implemented | retries attempts beyond one HTTP call |
| explicit dead-letter queue | implemented | permanent/exhausted work is separated from ordinary failures |
| queue reconciliation | implemented | expired/stranded state repair |
| scheduled Source sync | implemented | deterministic multi-worker scheduling |
| metadata-first SaaS incremental reuse | implemented | unchanged documents skip body download/embedding |
| provider delta/change feed | **not implemented** | optional optimization; not represented as existing functionality |
| Qdrant + PostgreSQL hybrid retrieval/RRF | implemented | production retrieval path |
| CJK lexical baseline | implemented | regression-tested Chinese lexical support |
| tenant/user/ACL principal model | implemented | trusted request scope |
| reader/operator/owner/admin RBAC | implemented | separates read and mutation duties |
| built-in Admin UI | implemented | Source/queue/DLQ operations without extra frontend build |
| Docker Compose / Helm / KEDA | implemented | local and Kubernetes deployment |
| PostgreSQL + Qdrant backup/restore tools | implemented | executable DR path |
| 1000-PDF integration/capacity benchmark | implemented | scale regression evidence |

## 2. Production invariants

### Durable storage

`RAGBOT_ENV=production` requires PostgreSQL, Qdrant, semantic embeddings, API keys, and complete principal mappings. Production does not silently fall back to in-memory storage, HashEmbedder, or inline ingestion.

### Immutable Job connector snapshot

A Job captures `source_type/source_config` when submitted. Later Source edits apply to future Jobs, not already queued work. DLQ requeue defaults to that immutable snapshot unless an operator explicitly selects current Source config.

### Two-layer retry contract

```text
provider request
  → bounded retry/backoff for transient HTTP/transport failures
  → whole-ingestion durable retry/backoff
  → dead_lettered when permanent or attempts exhausted
```

Provider retry honors `Retry-After`; permanent 4xx classes are not blindly retried.

### RBAC

- `reader`: query/chat/catalog/job reads;
- `operator`: tenant Source/ingestion/schedule/retry/requeue writes;
- `owner`: tenant operator superset;
- `admin=true`: global administrative surfaces/reconciliation.

The Admin UI reflects capabilities, but the API remains the authorization boundary.

### Secrets

SaaS Source config stores only `credential_ref=env:VARIABLE`. Docker Compose and Helm can inject those secrets only into worker workloads.

## 3. Automated release gates

Every candidate PR and release commit must use its **exact head SHA**. A prior green run cannot prove a later documentation/configuration commit.

Required ordinary CI jobs:

- Python 3.10 full tests;
- Python 3.12 full tests;
- PostgreSQL migration + queue + FTS + CJK integration;
- Node client typecheck;
- deployment configuration (both Compose variants, Helm, KEDA, worker-only secret isolation);
- bundled OpenVLA example;
- PostgreSQL + Qdrant **Backup + restore smoke**.

Additional repository gates:

- deterministic 1000-PDF PostgreSQL/Qdrant capacity/integration benchmark for retrieval/indexing changes;
- CJK Recall@5/MRR regression;
- RRF modality regression;
- production/security regression tests.

The recovery smoke performs a destructive seed → backup → delete → restore → verify sequence against PostgreSQL 16 and Qdrant v1.19.0.

## 4. Real staging gate

Ordinary CI uses deterministic/local infrastructure and does not prove production credentials or network policy.

Before `v1.0.0`, run the core staging workflow with the intended production-compatible LLM/embedding provider and validate:

- production startup;
- PostgreSQL/Qdrant;
- independent worker;
- local filesystem ingestion;
- Web ingestion;
- PDF ingestion;
- Git ingestion;
- hybrid `/search`;
- Agentic `/chat`;
- ACL negative isolation.

For every SaaS connector intended to be enabled at launch, run the dedicated SaaS staging smoke with a least-privilege test service identity and validate first ingestion plus unchanged-version chunk reuse.

## 5. Security and operations gate

Before release, record evidence for:

- `RAGBOT_ENV=production`;
- all API keys mapped to principals;
- reader/operator/admin duties separated;
- TLS / ingress auth / external rate-limit policy;
- egress allowlists/firewall/service-mesh policy;
- local Source roots explicitly mounted/read-only where possible;
- PostgreSQL/Qdrant not publicly exposed;
- worker-only SaaS secrets;
- immutable application image tag/digest;
- provider quotas and worker concurrency reviewed;
- PostgreSQL + Qdrant restore procedure tested;
- RPO/RTO and backup retention recorded;
- rollback/database migration/embedding reindex runbooks reviewed;
- observability retained outside the application process.

See `docs/DEPLOYMENT.md`, `docs/ADMIN_OPERATIONS.md`, and `docs/DISASTER_RECOVERY.md`.

## 6. Provider delta API decision

Current Drive/Notion/Confluence synchronization is **metadata-first**, not provider change-feed/cursor based. Each refresh still enumerates the configured remote scope, then downloads/embeds only changed content.

A true provider delta implementation requires persistent cursor/token state plus correct handling of:

- membership changes;
- moves/renames;
- deletions;
- cursor invalidation/full-resync fallback;
- permission changes;
- provider-specific eventual consistency.

Treat this as a separate PR. It is a v1 blocker only if the v1 product promise explicitly requires change-feed scale for very large SaaS repositories.

## 7. Release metadata gate

After code/staging/operations gates pass, create a small **release-only PR** that changes no architecture:

1. `pyproject.toml`: `0.5.0 → 1.0.0`;
2. FastAPI app version → `1.0.0`;
3. Helm Chart `version` / `appVersion` → `1.0.0`;
4. Helm default image tag → `1.0.0`;
5. finalize CHANGELOG release section;
6. run release-metadata consistency check;
7. run complete CI on the exact release SHA;
8. run core staging, applicable SaaS staging, and recovery gate on the release candidate;
9. create project-policy tag `v1.0.0`;
10. publish GitHub Release with migration, security, backup/restore, and known-limitations notes.

Do not bump to `1.0.0` in the production-hardening PR.

## 8. Known v1 limitations

- API-key principal IAM is service-oriented and is not full enterprise OIDC/SAML directory integration;
- SaaS synchronization is metadata-first rather than provider change-feed based;
- PostgreSQL `simple` FTS + CJK bigrams is a baseline, not a universal Chinese search solution;
- PostgreSQL and Qdrant do not share a distributed transaction; recovery/retry/reconciliation and replacement semantics provide consistency controls;
- OpenAI-compatible token accounting may be estimated when a provider does not supply authoritative usage;
- application connector validation does not replace network-layer egress controls.

## 9. Release sequence

```text
production-hardening PR
  → exact-head 7-job CI green
  → merge
  → core + SaaS staging evidence
  → optional provider-delta PR if required by launch scope
  → release/v1.0.0 PR
  → exact release SHA CI + staging + recovery green
  → v1.0.0 tag
  → GitHub Release
```
