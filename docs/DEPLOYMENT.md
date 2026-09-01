# Ragbot Deployment Guide

## 1. Deployment modes

### Local Python

Use in-memory storage when `POSTGRES_DSN`/`QDRANT_URL` are unset. This mode is useful for tests and single-process development only; state is lost on restart and cannot be shared across replicas.

### Docker Compose

The root `docker-compose.yml` and `infra/docker/docker-compose.yml` start API, PostgreSQL and Qdrant. Optional profiles add Ollama and Jaeger.

Copy `.env.example` to `.env`, configure model credentials, then run:

```bash
docker compose up -d --build
```

The API now depends on a one-shot `migrate` service. The migration container waits for PostgreSQL health, runs `python -m services.api.app.storage.migrations`, and only after successful migration may the API start. This works for both fresh and existing PostgreSQL volumes.

Do not reintroduce `/docker-entrypoint-initdb.d` as the upgrade mechanism: PostgreSQL executes that directory only during first cluster initialization.

## 2. Database migration lifecycle

Migrations live in `infra/migrations/` and are applied in filename order. The runner creates `schema_migrations`, acquires a PostgreSQL advisory lock, and records each applied filename. Multiple deployment instances can therefore attempt migration safely without racing each other.

Rules:

1. Released migration files are immutable.
2. Schema changes use a new ordered migration.
3. CI must prove a clean database can apply the complete chain.
4. CI also re-runs the migration runner to prove it becomes a no-op.
5. Back up PostgreSQL before destructive migrations.

Manual execution:

```bash
POSTGRES_DSN='postgresql://...' python -m services.api.app.storage.migrations
```

## 3. Kubernetes / Helm

Chart: `infra/helm/ragbot`.

Render/lint locally:

```bash
helm lint infra/helm/ragbot
helm template ragbot infra/helm/ragbot
```

### Single-replica development

The chart defaults to one replica. With empty `postgres.dsn` and `qdrant.url`, Ragbot falls back to process-local stores. This is not a production topology.

### Production / horizontal scale

Before setting `replicaCount > 1` or enabling HPA, configure both:

- a shared PostgreSQL DSN (`postgres.dsn` or `existingSecret`), and
- a shared `qdrant.url`.

The deployment template fails rendering for multi-replica/autoscaled configurations that would otherwise use process-local persistence/vector state.

When a PostgreSQL DSN is configured, each Pod runs the same migration command in an initContainer. The advisory lock + `schema_migrations` table serializes concurrent migration attempts.

### Existing Secret

Current template expects these keys in `existingSecret`:

- `postgres-dsn` (required when using the Secret for PostgreSQL)
- `openai-api-key`
- `api-keys` (optional)
- `qdrant-api-key` (optional)

Do not place credentials directly in committed `values.yaml`.

### Ingress and HPA

`values.yaml` ingress and autoscaling settings now have actual templates (`templates/ingress.yaml`, `templates/hpa.yaml`). Enable them only after the shared-store requirement above is satisfied.

## 4. Health and readiness

- `/admin/health`: liveness only; answers when the API process is alive.
- `/admin/ready`: checks configured repository and vector-store readiness and returns `503` when a dependency is unavailable/misconfigured.

Kubernetes uses `/admin/health` for liveness and `/admin/ready` for readiness. This prevents a process from receiving traffic merely because the Python event loop is alive while PostgreSQL/Qdrant are unavailable.

## 5. Embedding/Qdrant upgrades

`QDRANT_DIM` must equal the active embedder dimension and the existing collection vector size. Ragbot validates an existing collection and fails fast on a detectable mismatch.

Changing embedding model or vector dimension is a data migration, not a config-only deployment. Recommended procedure:

1. provision a new collection with the target dimension;
2. deploy an ingestion/reindex process using the new embedder;
3. validate retrieval quality and coverage;
4. cut query traffic to the new collection;
5. retain the old collection for rollback until the change is accepted.

Do not point a new-dimension embedder at an old collection.

## 6. Source data mounts

Compose mounts `RAGBOT_DATA_DIR` read-only at `/data`. `local_fs`, PDF and local repository Source configs should reference paths visible inside the API container (for example `/data/knowledge`). Keep source data outside the image and do not commit customer/private documents to the repository.

## 7. Upgrade procedure

For a normal release:

1. back up PostgreSQL and confirm Qdrant snapshot/backup policy;
2. build/pull the new immutable application image;
3. inspect release notes for schema or embedding changes;
4. run/apply migrations before API readiness;
5. roll out API instances;
6. confirm `/admin/ready` and CI-equivalent smoke tests;
7. trigger re-ingestion only when a connector/embedding/index change requires it;
8. monitor retrieval/tool error metrics and rollback if needed.

## 8. Production checklist

- non-empty service authentication or a trusted gateway;
- tenant/user identity bound upstream rather than trusted from arbitrary client JSON;
- external PostgreSQL and Qdrant for any multi-replica deployment;
- TLS at ingress/load balancer;
- PostgreSQL backup/restore test;
- Qdrant snapshot/restore plan;
- pinned container image versions/digests for controlled releases;
- resource limits and HPA tuned from load tests;
- structured logs/traces exported to a durable backend;
- `RAGBOT_TRACING_ENABLED` only when an OTLP collector is reachable;
- embedding collection/model/dimension recorded as release metadata;
- durable ingestion worker/queue before relying on ingestion across rolling restarts.
