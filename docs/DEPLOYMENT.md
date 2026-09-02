# Ragbot Deployment Guide

## 1. Deployment modes

### Local Python development

When `POSTGRES_DSN`/`QDRANT_URL` are unset, Ragbot can use in-memory stores and inline ingestion. This mode is for tests and single-process development only; state is process-local.

### Docker Compose

Root `docker-compose.yml` and `infra/docker/docker-compose.yml` start API, independent ingestion worker, one-shot migration service, PostgreSQL and Qdrant. Optional profiles add Ollama and Jaeger.

```bash
cp .env.example .env
# configure provider credentials
docker compose up -d --build
```

Compose sets `RAGBOT_INGESTION_MODE=worker`, so ingestion is durable rather than tied to the API process lifecycle.

## 2. Database migration lifecycle

Migrations under `infra/migrations/` run in filename order. The runner maintains `schema_migrations` and uses a PostgreSQL advisory lock so concurrent deployment instances cannot apply the same migration concurrently.

Rules:

1. released migrations are immutable;
2. schema changes create a new ordered migration;
3. CI proves a clean DB can apply the complete chain and that a second run is a no-op;
4. back up PostgreSQL before destructive migration work.

```bash
POSTGRES_DSN='postgresql://...' python -m services.api.app.storage.migrations
```

Migration 006 adds durable ingestion leases/attempts and CJK lexical support. Migration 008 adds durable Source synchronization state.

## 3. Durable ingestion worker and scheduler

```bash
RAGBOT_WORKER_POLL_SECONDS=1
RAGBOT_WORKER_LEASE_SECONDS=120
RAGBOT_WORKER_MAX_ATTEMPTS=3
RAGBOT_SCHEDULER_SCAN_SECONDS=30
python -m services.worker.main
```

Execution:

```text
API / recurring Source scheduler
            │
            ▼
PostgreSQL pending Job
            │
            ▼
FOR UPDATE SKIP LOCKED claim
            │
        lease + heartbeat
            │
            ▼
connector → chunks → PostgreSQL + Qdrant
```

Properties:

- API restart does not delete pending Jobs;
- workers compete safely for Jobs;
- crashed workers are recovered after lease expiration;
- max attempts stop infinite crash loops;
- Job connector config is an immutable submission snapshot;
- recurring scheduler uses deterministic Job IDs + atomic insert-if-absent across replicas;
- missed recurring windows collapse to one current refresh rather than replaying every missed interval;
- active work for a Source delays its scheduled refresh.

This is an at-least-once execution model, so connector/upsert paths must remain idempotent.

## 4. Kubernetes / Helm

Chart: `infra/helm/ragbot`.

```bash
helm lint infra/helm/ragbot
helm template ragbot infra/helm/ragbot
```

Minimum production shape:

```yaml
env:
  RAGBOT_ENV: production

worker:
  enabled: true
  replicaCount: 1
  schedulerScanSeconds: "30"

postgres:
  dsn: postgresql://...

qdrant:
  url: http://qdrant:6333
```

Production rendering rejects missing durable worker, PostgreSQL or Qdrant. Multi-replica API/HPA also requires shared PostgreSQL and Qdrant.

### Core existing Secret

`existingSecret` supports the core fixed keys:

- `postgres-dsn`
- `openai-api-key`
- `embedding-api-key`
- `api-keys`
- `api-key-principals`
- `qdrant-api-key`

### Worker-only cloud/SaaS credentials

Google Drive, Notion and Confluence Source configs store only `credential_ref=env:VARIABLE`. The referenced environment variable should normally exist **only on worker pods**.

Use `worker.extraEnvFrom` with a dedicated Secret/ExternalSecret:

```yaml
worker:
  enabled: true
  extraEnvFrom:
    - secretRef:
        name: ragbot-saas-connectors
```

Example Secret keys:

```text
RAGBOT_DRIVE_CREDENTIALS_JSON
RAGBOT_NOTION_TOKEN
RAGBOT_CONFLUENCE_TOKEN
```

Then Source config references them without persisting their values:

```json
{"credential_ref":"env:RAGBOT_NOTION_TOKEN"}
```

For fine-grained mapping use `worker.extraEnv` with `valueFrom.secretKeyRef`. Do not commit literal SaaS token values to Helm values.

This worker-only injection is preferable to placing SaaS credentials into `.Values.env`, because the API deployment does not need them.

### Source mounts

`extraVolumes` / `extraVolumeMounts` remain available for local_fs/local PDF/local Git Sources. Prefer read-only mounts below `RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS`.

## 5. Worker autoscaling

Ragbot supports optional KEDA backlog scaling for workers. The scaler queries the durable PostgreSQL Job queue rather than using CPU as an indirect backlog proxy.

```yaml
worker:
  enabled: true
  autoscaling:
    enabled: true
    minReplicaCount: 1
    maxReplicaCount: 10
    pollingInterval: 15
    cooldownPeriod: 120
    targetJobsPerWorker: "4"
    activationJobs: "1"
```

The query counts ready pending Jobs plus expired running leases. KEDA must already be installed in the cluster.

When cloud/SaaS Sources are enabled, size `maxReplicaCount` against upstream rate limits. Scaling workers faster than Google/Notion/Atlassian API quotas permit can turn a queue recovery into provider throttling.

## 6. Cloud/SaaS network and secret boundaries

Full connector configuration is in `docs/CLOUD_CONNECTORS.md`.

Security expectations:

- secret values are worker environment/secret-store data, never Source config;
- Google Drive and Notion use fixed official API hosts;
- Confluence base URLs are tenant-configurable, therefore production requires `RAGBOT_CONFLUENCE_ALLOWED_HOSTS`;
- private/self-hosted Confluence additionally requires explicit private-network opt-in when its address is non-public;
- S3-compatible custom endpoints require their own explicit host allowlist;
- provider service identities should have read-only/least-privilege access to only the intended content;
- cluster/VPC/service-mesh egress policy remains the final network boundary.

Credential rotation does not require rewriting Sources: keep the same `credential_ref` and rotate the worker Secret value.

## 7. Incremental SaaS synchronization

Scheduled Google Drive, Notion and Confluence refreshes are metadata-first:

```text
remote metadata listing
        │
        ├─ unchanged version → reuse existing chunks
        │                      no body download / no embedding
        │
        └─ new/changed       → fetch body → chunk → embed
                               │
complete replacement snapshot ┘
        │
        └─ prune remotely deleted documents
```

This reduces content-download and embedding cost while keeping the same replacement/retry semantics as other Ragbot Sources.

Current metadata discovery still scans the configured remote tree/space; it is not yet based on provider delta/change feeds.

## 8. Health and operations

- `/admin/health`: process liveness.
- `/admin/ready`: repository/vector readiness.
- `/admin/ui`: built-in operator control plane.
- `/catalog/overview`: tenant-scoped Source/knowledge/queue summary.
- `/admin/queue/metrics`: global admin backlog and schedule metrics.

API readiness does not prove a worker is consuming Jobs. Production alerting should include oldest pending age, failed Job count, stale running leases and worker process health.

## 9. Embedding / Qdrant upgrades

`QDRANT_DIM` must equal the actual embedder dimension and collection size. An embedding model/dimension change is a data migration:

1. provision a new collection;
2. deploy the target embedder;
3. re-ingest Sources;
4. validate retrieval/ACL/citations;
5. cut query traffic;
6. retain old collection for rollback;
7. remove only after acceptance.

Do not silently reuse an incompatible collection.

## 10. CJK lexical upgrade behavior

New writes create the current lexical representation and chunk metadata carries `lexical_version`. After a lexical-version change, ordinary re-ingest rewrites stale representation once; subsequent unchanged runs can reuse it.

Validate disk growth, index build time and retrieval quality before production-wide reindex.

## 11. Source/network boundaries

Production local Sources must remain inside `RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS`. Remote Web/PDF/Git connectors enforce URL/redirect rules; S3 and Confluence have connector-specific endpoint allowlists. These application checks complement, not replace, network egress controls.

Do not expose PostgreSQL or Qdrant admin endpoints publicly.

## 12. Staging gates

`.github/workflows/staging-smoke.yml` is the manual core production-mode gate using real model credentials, PostgreSQL and Qdrant. It covers local/Web/PDF/Git ingestion, hybrid search, Agent chat and ACL-negative isolation.

Cloud/SaaS connectors have protocol-level fake-server tests in ordinary CI and should additionally be exercised with dedicated staging service identities before production enablement. The optional SaaS smoke workflow is intended for that purpose when connector staging credentials are configured.

A green ordinary PR CI is not a substitute for production credentials/network validation.

## 13. Upgrade procedure

1. back up PostgreSQL and verify Qdrant snapshot availability;
2. build/pull an immutable image;
3. inspect schema/embedding/connector release notes;
4. run migrations;
5. deploy workers and API with compatible secrets/configuration;
6. confirm readiness and worker queue consumption;
7. run controlled re-ingestion if index representation changed;
8. run core and applicable SaaS staging smoke;
9. monitor queue age, provider throttling, retrieval/tool errors and latency;
10. use the rollback/reindex runbook if acceptance fails.

## 14. v1 production checklist

- production mode and scoped principals enabled;
- durable workers consuming Jobs;
- external PostgreSQL/Qdrant;
- TLS/rate limiting/upstream auth policy;
- source allowlists and network egress policy;
- SaaS secrets injected only into required worker workloads;
- least-privilege provider service identities;
- PostgreSQL backup/restore test;
- Qdrant snapshot/restore test;
- pinned image tag/digest;
- resource/load evidence;
- logs/traces exported durably;
- embedding collection/model/dimension recorded;
- retrieval baseline captured;
- core staging smoke green;
- SaaS staging smoke green for enabled production connectors;
- rollback/migration/reindex runbook reviewed.
