# Ragbot Deployment Guide

## 1. Deployment modes

### Local Python development

When `POSTGRES_DSN`/`QDRANT_URL` are unset, Ragbot can use in-memory stores and inline ingestion. This mode is for tests and single-process development only; state is process-local.

### Docker Compose

Root `docker-compose.yml` and `infra/docker/docker-compose.yml` start:

- API;
- independent ingestion worker;
- one-shot migration service;
- PostgreSQL;
- Qdrant.

Optional profiles add Ollama and Jaeger.

```bash
cp .env.example .env
# configure provider credentials
docker compose up -d --build
```

Compose sets `RAGBOT_INGESTION_MODE=worker`, so `/ingest/jobs` persists a pending job and returns; the worker owns execution.

## 2. Database migration lifecycle

Migrations live in `infra/migrations/` and are applied in filename order. The runner creates `schema_migrations`, acquires a PostgreSQL advisory lock and records each applied filename. Multiple deployment instances can therefore attempt migration safely.

Rules:

1. Released migration files are immutable.
2. Schema changes use a new ordered migration.
3. CI proves a clean database can apply the complete chain.
4. CI re-runs the migration runner to prove it becomes a no-op.
5. Back up PostgreSQL before destructive migrations.

Manual execution:

```bash
POSTGRES_DSN='postgresql://...' python -m services.api.app.storage.migrations
```

Migration 006 adds durable ingestion leases/attempts and CJK `fts_text` support.

## 3. Durable ingestion worker

Worker entry point:

```bash
python -m services.worker.main
```

Execution model:

```text
POST /ingest/jobs
      │
      ▼
PostgreSQL pending job
      │
      ▼
worker claim: FOR UPDATE SKIP LOCKED
      │
      ├─ lease_owner
      ├─ lease_expires_at
      ├─ heartbeat_at
      └─ attempts
      │
      ▼
ingestion pipeline → PostgreSQL chunks + Qdrant vectors
```

If a worker crashes, its running job remains durable. Once the lease expires another worker changes eligible work back to pending and reclaims it. When attempts reach `RAGBOT_WORKER_MAX_ATTEMPTS`, the job is marked failed instead of retrying forever.

Important properties:

- API restart does not delete pending jobs;
- worker replicas can safely compete for work;
- leases prevent one healthy job from being concurrently claimed by multiple workers;
- heartbeat protects long-running ingestion from premature reclaim;
- retry API creates a new job rather than mutating historical failed execution evidence.

This is an **at-least-once execution model**. Connector/upsert paths therefore need to remain idempotent; Ragbot's replacement/dedup semantics are designed around that assumption.

## 4. Kubernetes / Helm

Chart: `infra/helm/ragbot`.

```bash
helm lint infra/helm/ragbot
helm template ragbot infra/helm/ragbot
```

### Development render

The default chart remains development-friendly: one API replica, worker disabled, empty external-store URLs allowed.

### Production render

Set at minimum:

```yaml
env:
  RAGBOT_ENV: production

worker:
  enabled: true
  replicaCount: 1

postgres:
  dsn: postgresql://...

qdrant:
  url: http://qdrant:6333
```

Production chart rendering fails when:

- worker is disabled;
- PostgreSQL is absent;
- Qdrant is absent.

Multi-replica/HPA additionally requires shared PostgreSQL and Qdrant.

The worker has its own Deployment and can be scaled independently from API replicas. All workers share the same PostgreSQL lease queue.

### Existing Secret

Current chart can reference these keys:

- `postgres-dsn`
- `openai-api-key`
- `embedding-api-key` (worker optional)
- `api-keys`
- `api-key-principals`
- `qdrant-api-key`

Do not place credentials directly in committed `values.yaml`.

### Source mounts

`extraVolumes` / `extraVolumeMounts` are applied to both API and worker so a local Source path has identical meaning in both components. Prefer read-only mounts under `RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS`.

### Ingress / HPA

Ingress and API HPA have chart templates. Worker scaling is currently explicit via `worker.replicaCount`; backlog-based worker autoscaling is a v1.x improvement.

## 5. Health / readiness

- `/admin/health`: liveness only.
- `/admin/ready`: checks repository and vector-store readiness; returns `503` on failure.

API readiness deliberately does not imply that a worker is available. Production operations should additionally monitor queue depth/oldest pending age and worker health/logs; dedicated queue metrics are a follow-up observability improvement.

## 6. Embedding / Qdrant upgrades

`QDRANT_DIM` must equal the actual embedder dimension and collection vector size. Changing embedding model or dimension is a data migration:

1. provision a new collection;
2. deploy the target embedder;
3. re-ingest sources;
4. validate retrieval quality and ACL/citation coverage;
5. cut query traffic;
6. retain old collection for rollback;
7. clean up after acceptance.

Do not point a new-dimension embedder at an old collection.

## 7. CJK lexical upgrade behavior

Migration 006 initially backfills `fts_text=text`. New writes generate CJK bigram lexemes. Chunk metadata includes `lexical_version`; after upgrading, a normal re-ingest treats the prior representation as stale and rewrites it once. Subsequent unchanged re-ingests reuse the new representation normally.

Before large production re-indexes, validate disk growth, GIN index build time and query latency on a staging-sized corpus.

## 8. Source/network boundaries

Compose mounts `RAGBOT_DATA_DIR` read-only at `/data` for both API and worker. Production local sources must remain inside `RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS`.

Remote Web/PDF/Git source policy is application-enforced, but production should also use VPC/firewall/service-mesh egress controls. Do not expose PostgreSQL or Qdrant admin endpoints to the public Internet.

## 9. Real-provider staging gate

`.github/workflows/staging-smoke.yml` is intentionally manual because it consumes a real model credential. Configure GitHub environment `staging` with `STAGING_OPENAI_API_KEY` and optional provider variables, then dispatch **Staging Smoke**.

The workflow starts production-mode API + durable worker against real PostgreSQL/Qdrant and executes:

- local_fs ingestion;
- Web ingestion;
- PDF ingestion;
- Git ingestion;
- hybrid `/search`;
- Agentic `/chat`;
- ACL negative isolation.

A green ordinary PR CI is not a substitute for this gate.

## 10. Upgrade procedure

For a normal release:

1. back up PostgreSQL and verify Qdrant snapshot availability;
2. build/pull an immutable application image;
3. inspect schema/embedding/lexical release notes;
4. run migrations;
5. deploy workers and API with compatible configuration;
6. confirm API readiness and worker queue consumption;
7. trigger controlled re-ingestion when index/lexical representation changed;
8. run staging/equivalent smoke;
9. monitor queue age, retrieval errors, tool errors and latency;
10. rollback application and/or collection according to runbook if needed.

## 11. v1 production checklist

- production mode enabled;
- scoped service authentication/principals;
- durable worker enabled and consuming jobs;
- external PostgreSQL/Qdrant;
- TLS + rate limiting + upstream authentication policy;
- egress policy/source allowlists;
- PostgreSQL backup/restore test completed;
- Qdrant snapshot/restore test completed;
- pinned application image tag/digest;
- resource limits/load test evidence;
- logs/traces exported to a durable backend;
- embedding collection/model/dimension recorded as release metadata;
- CJK retrieval baseline captured for the deployment corpus;
- real-provider staging smoke green;
- rollback/migration/reindex runbook reviewed.
