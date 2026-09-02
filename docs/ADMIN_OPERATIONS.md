# Ragbot Admin & Operations Guide

This guide covers the built-in control plane, tenant RBAC, durable ingestion operations, dead-letter recovery, recurring synchronization, connector operations, and worker scaling.

## 1. Open the control plane

```text
http://localhost:8000/admin/ui
```

The UI is served directly by FastAPI with no Node/npm build step. It contains no embedded API credential. The entered `X-API-Key` is stored only in browser `sessionStorage`.

The UI displays the current principal role/capability using `GET /catalog/session` and applies the same backend RBAC contract to its controls:

- `reader`: read-only catalog/queue visibility;
- `operator`: Source, ingestion, retry/requeue, pause/resume, and schedule operations;
- `owner`: tenant-level operator superset;
- `admin=true`: global admin operations such as reconciliation.

UI hiding/disablement is usability only. Authorization is always enforced again by the API.

## 2. Dashboard signals

The control plane shows:

- Source count;
- indexed document/chunk count;
- pending/running Jobs;
- failed Jobs;
- **Dead Lettered** Jobs;
- scheduled Sources;
- oldest pending age;
- stale running leases;
- completed/failed/dead-lettered counts over 24 hours;
- next scheduled sync.

Job rows show `failure_class`, bounded error text, attempts, completion/DLQ timestamps, and the appropriate recovery action.

## 3. Reader vs operator behavior

A reader key can inspect tenant-scoped knowledge/catalog state but cannot mutate it. Quick Import and Source/Job mutation controls are disabled in the UI and backend calls return `403`.

An operator/owner key can:

- Quick Import;
- trigger ingestion;
- pause/resume Sources;
- configure schedules;
- retry failed Jobs;
- requeue dead-lettered Jobs.

Use a dedicated global admin key only for global operations. Avoid giving admin credentials to routine ingestion automation.

## 4. Queue state model

```text
pending
  ↓ worker claim
running
  ├─ completed
  ├─ retryable failure → pending after durable backoff
  └─ permanent/exhausted failure → dead_lettered
```

Provider HTTP calls have a short retry layer for 408/425/429/5xx and transport errors. If the whole ingestion attempt still fails, the Job-level durable retry/backoff contract applies.

Important worker settings:

```bash
RAGBOT_WORKER_MAX_ATTEMPTS=3
RAGBOT_WORKER_RETRY_BASE_SECONDS=5
RAGBOT_WORKER_RETRY_MAX_SECONDS=300
RAGBOT_RECONCILE_SECONDS=30
RAGBOT_PROVIDER_MAX_ATTEMPTS=4
RAGBOT_PROVIDER_BACKOFF_BASE_SECONDS=0.5
RAGBOT_PROVIDER_BACKOFF_MAX_SECONDS=30
```

`Retry-After` is honored when provider responses supply it.

## 5. Failed Retry vs DLQ Requeue

### Failed Job

For a `failed` Job, the UI exposes **Retry**. The endpoint is:

```http
POST /ingest/jobs/{job_id}/retry
```

This creates a fresh Job using the Source's **current** connector configuration.

### Dead-lettered Job

For `dead_lettered`, the UI exposes **Requeue snapshot**:

```http
POST /ingest/jobs/{job_id}/requeue
Content-Type: application/json

{"use_current_source_config": false}
```

Default behavior replays the immutable connector snapshot captured by the dead-letter Job. This is safer for incident analysis and reproducibility.

Use:

```json
{"use_current_source_config": true}
```

only when an operator deliberately repaired/changed the Source and wants the next attempt to use that new configuration.

Do not repeatedly requeue a permanent authentication/configuration failure without fixing the underlying cause.

## 6. Reconciliation

The worker periodically reconciles queue state according to `RAGBOT_RECONCILE_SECONDS`.

Global admins can also invoke:

```bash
curl -X POST 'https://ragbot.example.com/admin/queue/reconcile?max_attempts=3' \
  -H "X-API-Key: $RAGBOT_ADMIN_KEY"
```

Reconciliation is for queue-state repair, including expired leases and stranded failures. It is not a substitute for fixing bad credentials, network policy, malformed Sources, or provider outages.

## 7. Control-plane APIs

Tenant-scoped:

```text
GET /catalog/session
GET /catalog/overview
GET /catalog/sources
GET /catalog/jobs
```

Global admin:

```text
GET  /admin/overview
GET  /admin/queue/metrics
POST /admin/queue/reconcile
```

Catalog responses intentionally redact full Source/Job connector config and secret references.

## 8. Recurring Source sync

Configure:

```bash
curl -X PUT http://localhost:8000/sources/<SOURCE_ID>/sync \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $RAGBOT_OPERATOR_KEY" \
  -d '{"enabled":true,"interval_seconds":3600,"run_immediately":false}'
```

Every durable worker may scan due Sources. Duplicate schedule windows are prevented using deterministic Job IDs plus atomic insert-if-absent.

If Ragbot misses multiple schedule intervals, it collapses them into one current refresh rather than replaying every missed run. If another Job for the same Source is active, scheduled sync waits.

## 9. Cloud and object-store Sources

Supported product Sources include S3/MinIO, Google Drive, Notion, and Confluence.

Secrets stay outside Source config. Example:

```json
{"credential_ref":"env:RAGBOT_NOTION_TOKEN"}
```

Docker Compose loads arbitrary SaaS credentials from the worker-only `RAGBOT_WORKER_ENV_FILE`. Helm uses `worker.extraEnv` / `worker.extraEnvFrom`.

Drive/Notion/Confluence currently use metadata-first synchronization: unchanged remote versions reuse chunks and skip body download/embedding. This is not yet a persistent provider delta/change-feed implementation.

## 10. Backlog-driven autoscaling

Helm can create a KEDA PostgreSQL `ScaledObject`:

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

The scaler counts ready pending Jobs plus expired running leases. Size maximum replicas against provider quotas; aggressive scaling can amplify 429 throttling.

## 11. Operational interpretation

| Signal | Meaning | Response |
| --- | --- | --- |
| pending grows | arrival rate > ingestion capacity | scale workers; inspect embedding/provider latency |
| oldest pending grows | freshness SLA degrading | scale/reduce schedule frequency |
| stale leases > 0 | worker interruption | inspect worker health; reconcile/reclaim |
| failed grows | transient/legacy failure state | inspect class/error, repair, Retry |
| dead_lettered grows | permanent/exhausted failures | fix root cause, then Requeue |
| provider 429 grows | upstream quota pressure | lower concurrency/worker max; increase backoff |
| scheduled high, queue stable | expected recurring workload | no action |

## 12. Disaster recovery

Use `scripts/backup_ragbot.sh` and `scripts/restore_ragbot.sh` for PostgreSQL + Qdrant recovery. A real backup/restore CI smoke validates the mechanism.

See [`DISASTER_RECOVERY.md`](DISASTER_RECOVERY.md) before running a production restore. Restore is destructive and should be performed with API write traffic and workers quiesced.

## 13. Incident triage order

For a rising failure/DLQ rate:

1. inspect `failure_class` and Job error;
2. determine whether it is auth/config, provider throttle/outage, network policy, parsing, embedding, or storage;
3. check worker/provider retry metrics/logs;
4. fix credentials/config/network/provider issue;
5. reconcile only if queue state is stranded;
6. Retry failed or Requeue DLQ work deliberately;
7. watch oldest pending age while backlog drains;
8. avoid scaling workers if the bottleneck is an upstream rate limit.
