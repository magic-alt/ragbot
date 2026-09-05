# Ragbot Deployment Guide

## 1. Deployment modes

### Development

Without `POSTGRES_DSN`/`QDRANT_URL`, Ragbot can use in-memory stores and inline ingestion. This mode is for tests and local single-process development only.

### Docker Compose

Root `docker-compose.yml` and `infra/docker/docker-compose.yml` start:

- API;
- independent ingestion worker;
- migration service;
- PostgreSQL 16;
- Qdrant v1.19.0;
- optional Ollama and Jaeger profiles.

```bash
cp .env.example .env
mkdir -p data
docker compose up -d --build
```

Both Compose variants set `RAGBOT_INGESTION_MODE=worker` and now expose the same durable retry/reconcile/provider retry settings.

### macOS Docker Desktop with host Ollama

When Ollama runs on the Mac, set the following in `.env` (the model must already
be installed in Ollama):

```dotenv
RAGBOT_LLM_PROVIDER=ollama
RAGBOT_DOCKER_OLLAMA_BASE_URL=http://host.docker.internal:11434
EMBEDDING_MODEL=qwen3-embedding:8b
EMBEDDING_BASE_URL=http://host.docker.internal:11434
EMBEDDING_TIMEOUT_SECONDS=300
EMBEDDING_BATCH_SIZE=8
QDRANT_DIM=4096
QDRANT_COLLECTION=rag_chunks_qwen3_8b_4096
```

Set `OLLAMA_MODEL` separately to your installed chat model. The Docker Ollama
URL defaults to `http://ollama:11434` for the optional container profile;
`RAGBOT_DOCKER_OLLAMA_BASE_URL` selects the host installation. Setting only
`OLLAMA_BASE_URL` does not configure the embedding endpoint. API and worker
must use the same embedding model, dimension and collection. A model/dimension
change requires re-ingesting the corpus into a compatible collection; keep the
old collection until that work is verified.

```bash
python3 scripts/ragbot.py restart --mode docker
python3 scripts/ragbot.py ingest data/ --tenant engineering --type pdf
```

Docker Desktop can publish port 8000 while an older local Python API still owns
`127.0.0.1:8000`. In that case the CLI reaches the old API even after rebuilding
Docker, and may report a missing `/ingest/upload/pdf` route. The controller now
stops a verified local API recorded in its PID file before Docker startup and
compares `/admin/runtime` boot IDs through the container and host addresses
before recording a successful Docker runtime. If an unmanaged process conflicts,
inspect `lsof -nP -iTCP:8000 -sTCP:LISTEN`, stop the conflicting API, or choose
`--port 8001`. Readiness alone does not establish which process the CLI reaches.

The shared image's HTTP healthcheck applies to the API. Worker and migration
services disable that inherited check because they have no HTTP listener;
inspect worker logs and durable job completion to verify ingestion.

PDF uploads default to a 25 MiB limit. For larger documents, set
`RAGBOT_PDF_MAX_BYTES=67108864` (64 MiB) in `.env` and recreate the services with
the controller before retrying the affected file. An HTTP 413 response indicates
this upload limit, rather than an embedding failure.

### Worker-only SaaS credentials

Cloud connector credentials should exist only in the worker. Both Compose files support:

```bash
cp .env.worker.example .env.worker
RAGBOT_WORKER_ENV_FILE="$PWD/.env.worker" docker compose up -d --build
```

The API service does not load this file. Source config contains only `credential_ref=env:VARIABLE`.

## 2. Production identity and RBAC

Production requires API-key principal mappings. Recommended role split:

- `reader`: retrieval/chat/catalog/job read only;
- `operator`: reader plus Source/ingestion/schedule/retry/requeue mutations;
- `owner`: tenant-level operator superset;
- `admin=true`: global operational surfaces and reconciliation.

Do not give a `reader` key to automation that must create or refresh knowledge Sources.

## 3. Database migration lifecycle

Migrations under `infra/migrations/` run in filename order and are tracked in `schema_migrations`. The migration runner uses a PostgreSQL advisory lock.

Rules:

1. released migrations are immutable;
2. schema changes add a new ordered migration;
3. clean-database migration and no-op re-run are CI concerns;
4. back up PostgreSQL before destructive changes.

```bash
POSTGRES_DSN='postgresql://...' python -m services.api.app.storage.migrations
```

Migration 006 introduced durable queue state; 008 introduced Source sync schedule; 009 introduces dead-letter metadata/indexes.

## 4. Durable worker, retries, reconciliation, and scheduler

Recommended defaults:

```bash
RAGBOT_WORKER_POLL_SECONDS=1
RAGBOT_WORKER_LEASE_SECONDS=120
RAGBOT_WORKER_MAX_ATTEMPTS=3
RAGBOT_WORKER_RETRY_BASE_SECONDS=5
RAGBOT_WORKER_RETRY_MAX_SECONDS=300
RAGBOT_RECONCILE_SECONDS=30
RAGBOT_SCHEDULER_SCAN_SECONDS=30
RAGBOT_PROVIDER_MAX_ATTEMPTS=4
RAGBOT_PROVIDER_BACKOFF_BASE_SECONDS=0.5
RAGBOT_PROVIDER_BACKOFF_MAX_SECONDS=30
python -m services.worker.main
```

Execution model:

```text
API / scheduler
      ↓
PostgreSQL pending Job
      ↓ claim (FOR UPDATE SKIP LOCKED)
running + lease + heartbeat
      ↓
provider/connector request
  ├─ transient HTTP/transport failure
  │      → short provider retry/backoff
  └─ still failing
         ↓
whole-ingestion durable attempt
  ├─ retryable + attempts remain
  │      → pending + durable backoff
  └─ permanent/exhausted
         → dead_lettered
```

`Retry-After` is honored for provider responses when present. Permanent authentication/not-found style failures are not blindly retried.

Reconciliation periodically repairs expired leases and stranded failure state. Set `RAGBOT_RECONCILE_SECONDS=0` only when another operational process owns reconciliation.

Scheduled sync uses deterministic Job IDs + atomic insert-if-absent. Missed intervals collapse into one current refresh instead of backfilling every historical interval.

## 5. Docker Compose reliability parity

Both Compose files expose the same worker reliability variables and pin Qdrant to `qdrant/qdrant:v1.19.0`.

This parity matters because local/staging Compose should exercise the same retry/reconcile semantics as Helm rather than silently running weaker defaults.

## 6. Kubernetes / Helm

Chart: `infra/helm/ragbot`.

Minimum production shape:

```yaml
env:
  RAGBOT_ENV: production

worker:
  enabled: true
  replicaCount: 1
  pollSeconds: "1"
  leaseSeconds: "120"
  maxAttempts: "3"
  retryBaseSeconds: "5"
  retryMaxSeconds: "300"
  reconcileSeconds: "30"
  schedulerScanSeconds: "30"
  providerMaxAttempts: "4"
  providerBackoffBaseSeconds: "0.5"
  providerBackoffMaxSeconds: "30"

postgres:
  dsn: postgresql://...

qdrant:
  url: http://qdrant:6333
```

Production rendering rejects missing durable worker, PostgreSQL, or Qdrant.

### Core Secret

`existingSecret` may provide:

- `postgres-dsn`
- `openai-api-key`
- `embedding-api-key`
- `api-keys`
- `api-key-principals`
- `qdrant-api-key`

### Worker-only connector Secrets

Use `worker.extraEnvFrom` or `worker.extraEnv` for SaaS credentials:

```yaml
worker:
  enabled: true
  extraEnvFrom:
    - secretRef:
        name: ragbot-saas-connectors
```

Avoid putting SaaS secrets into `.Values.env`, because API pods do not need them.

### Source mounts

Use `extraVolumes` / `extraVolumeMounts` for local filesystem/PDF/Git Sources and keep them read-only below `RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS` when possible.

## 7. Backlog-driven worker autoscaling

Optional KEDA scaling uses the PostgreSQL queue, not CPU, as the backlog signal:

```yaml
worker:
  autoscaling:
    enabled: true
    minReplicaCount: 1
    maxReplicaCount: 10
    pollingInterval: 15
    cooldownPeriod: 120
    targetJobsPerWorker: "4"
    activationJobs: "1"
```

Size `maxReplicaCount` against upstream provider quotas. More workers can increase 429 pressure if provider quotas are already saturated.

## 8. Queue operations

Useful endpoints:

- `/catalog/overview`: tenant-scoped queue/knowledge summary;
- `/catalog/jobs`: redacted Job list with failure class;
- `/catalog/session`: non-secret principal capability summary;
- `/admin/queue/metrics`: global backlog/DLQ metrics, admin only;
- `/admin/queue/reconcile`: admin repair of expired/stranded queue state;
- `/ingest/jobs/{id}/retry`: retry a `failed` Job using current Source config;
- `/ingest/jobs/{id}/requeue`: requeue a `dead_lettered` Job, defaulting to its immutable connector snapshot.

Production alerting should include:

- oldest pending age;
- pending/running counts;
- stale leases;
- failed and dead-lettered counts;
- provider throttling/retry rates;
- worker process availability.

## 9. Disaster recovery

Ragbot ships:

```bash
bash scripts/backup_ragbot.sh ./backups/<name>
bash scripts/restore_ragbot.sh ./backups/<name>
```

The scripts cover PostgreSQL custom-format dump/restore and Qdrant collection snapshot download/upload with SHA-256 manifest verification.

Full operational procedure, traffic quiescing, post-restore validation, queue reconciliation, RPO/RTO recording, and limitations are documented in [`DISASTER_RECOVERY.md`](DISASTER_RECOVERY.md).

CI includes a real destructive seed → backup → delete → restore → verify smoke against PostgreSQL 16 and Qdrant v1.19.0.

## 10. Cloud/SaaS network boundaries

- secret values live in worker environment/secret stores;
- Drive/Notion use fixed official API hosts;
- Confluence custom host requires production allowlisting;
- private/self-hosted endpoints require explicit private-network opt-in where applicable;
- S3-compatible custom endpoints require host allowlisting;
- provider service identities should be read-only/least privilege;
- egress firewall/service-mesh policy remains the final network boundary.

Credential rotation does not require Source rewrite when the `credential_ref` name remains stable.

## 11. Incremental synchronization

Drive/Notion/Confluence are metadata-first: unchanged remote versions reuse existing chunks and skip body download/embedding; changed/new content is downloaded and indexed; remote deletions are pruned after successful replacement.

The current implementation still enumerates configured remote trees/spaces. It is not yet based on provider delta/change feeds. Persistent provider cursors/tokens are a separate optimization milestone.

## 12. Embedding / Qdrant upgrades

`QDRANT_DIM` must equal the actual embedding dimension. A model/dimension change is a data migration:

1. provision a new collection;
2. deploy the target embedder;
3. re-ingest Sources;
4. validate retrieval/ACL/citations;
5. cut traffic;
6. retain old collection for rollback;
7. remove it only after acceptance.

Do not silently reuse an incompatible collection.

## 13. Staging gates

Core staging uses `.github/workflows/staging-smoke.yml` with production mode, PostgreSQL, Qdrant, independent worker, and real model credentials.

SaaS staging uses the dedicated SaaS smoke workflow when staging service identities are configured. Ordinary fake-server connector tests are necessary but not sufficient for production enablement.

A green PR CI is not a substitute for real credentials/network/provider validation.

## 14. Upgrade checklist

1. capture PostgreSQL + Qdrant backup and verify restore procedure;
2. use an immutable image tag/digest;
3. inspect migration/embedding/connector changes;
4. run migrations;
5. deploy workers/API with compatible secrets;
6. verify readiness and queue consumption;
7. run controlled re-ingestion when representation changed;
8. run core and applicable SaaS staging smoke;
9. monitor queue age, provider throttling, DLQ, retrieval errors, and latency;
10. invoke rollback/reindex/restore runbook if acceptance fails.

## 15. v1 production checklist

- production mode enabled;
- every API key mapped to a scoped principal;
- reader/operator/admin keys separated by actual duties;
- durable worker enabled;
- retry/reconcile parameters reviewed;
- external PostgreSQL/Qdrant;
- TLS/ingress/rate-limit/egress policy;
- worker-only SaaS secrets;
- least-privilege provider identities;
- PostgreSQL + Qdrant restore test;
- immutable image version;
- capacity/retrieval baseline;
- core staging smoke green;
- SaaS staging green for enabled connectors;
- rollback/migration/reindex/DR runbooks reviewed.
