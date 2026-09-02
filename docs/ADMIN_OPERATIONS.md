# Ragbot Admin & Operations Guide

This guide covers the product control plane introduced after Quick Import: the built-in Web UI, Source Catalog, ingestion queue operations, recurring Source sync, S3/MinIO onboarding and worker backlog autoscaling.

## 1. Open the built-in control plane

After the API is running:

```text
http://localhost:8000/admin/ui
```

The UI is served directly by FastAPI and has no Node/npm build step. It does not embed an API credential. Operators enter `X-API-Key` in the browser; the value is stored only in browser `sessionStorage` and is sent to Ragbot APIs.

The page provides:

- Quick Import;
- tenant-scoped Source Catalog;
- last successful indexed document/chunk counts;
- latest ingestion status;
- failed Job retry;
- Source pause/resume;
- recurring sync configuration;
- pending/running/failed queue metrics;
- oldest pending age and stale lease count.

The UI is an operations console, not an IAM boundary. In production, keep the existing API-key principal / ingress / TLS controls in front of it.

## 2. Control-plane APIs

### Tenant-scoped overview

```http
GET /catalog/overview?tenant_id=engineering
```

Returns Source totals, scheduled Sources, queue backlog, 24-hour completion/failure activity and the size of the most recent successful knowledge view.

### Source Catalog

```http
GET /catalog/sources?tenant_id=engineering&status=active&limit=100
```

Optional filters:

- `tenant_id`
- `status`
- `source_type`
- `q`
- `limit` (`1..500`)

Catalog responses intentionally expose a sanitized location rather than the complete connector configuration. This is important for connectors that use secret references.

### Job Catalog

```http
GET /catalog/jobs?tenant_id=engineering&status=failed
```

`source_config` is deliberately removed from control-plane Job responses.

### Global queue metrics

Admin principals can query:

```http
GET /admin/queue/metrics
```

Metrics include:

- pending Jobs;
- running Jobs;
- failed Jobs;
- oldest pending Job age;
- expired/stale running leases;
- completions and failures over the previous 24 hours;
- scheduled Source count and next scheduled run.

## 3. Recurring Source sync

Recurring synchronization is first-class Source state, not connector configuration.

Enable hourly sync:

```bash
curl -X PUT http://localhost:8000/sources/<SOURCE_ID>/sync \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $RAGBOT_API_KEY" \
  -d '{"enabled":true,"interval_seconds":3600,"run_immediately":false}'
```

Run one due window immediately and then continue hourly:

```json
{"enabled":true,"interval_seconds":3600,"run_immediately":true}
```

Disable:

```json
{"enabled":false}
```

The minimum interval is 60 seconds.

### Scheduler semantics

Every durable worker may scan due Sources. This does **not** create duplicate scheduled Jobs because:

1. each Source + schedule window derives a deterministic Job ID;
2. PostgreSQL uses atomic `INSERT ... ON CONFLICT DO NOTHING`;
3. the Job stores an immutable connector-config snapshot;
4. normal worker lease/heartbeat/retry semantics execute the Job.

If a manual or different Job for the Source is already pending/running, scheduled sync waits instead of running concurrent replacement ingestion.

If Ragbot is unavailable for multiple schedule intervals, it collapses missed intervals into one current refresh and advances the next schedule into the future. It does not replay every missed interval and cause an ingestion storm.

Worker scan frequency:

```bash
RAGBOT_SCHEDULER_SCAN_SECONDS=30
```

Set to `0` to disable recurring scheduling while retaining ordinary durable ingestion.

## 4. S3 and MinIO knowledge Sources

Quick Import accepts an S3 URI directly:

```bash
rag --server http://localhost:8000 \
  --tenant engineering \
  ingest s3://engineering-manuals/servo/ \
  --wait
```

Or through the API:

```bash
curl -X POST http://localhost:8000/ingest/quick \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $RAGBOT_API_KEY" \
  -d '{
    "tenant_id":"engineering",
    "location":"s3://engineering-manuals/servo/",
    "tags":["manuals"]
  }'
```

The connector currently indexes common text/Markdown/code/config formats and PDF objects. Each S3 object becomes a separate Ragbot Document. Object listing is paginated and individual reads have a configurable hard size limit.

### AWS credentials

For AWS S3, prefer the normal boto3 credential chain: workload identity/IAM role, environment credentials or other deployment-owned mechanisms.

Do **not** write access keys into `Source.config`.

### MinIO / S3-compatible endpoint

A Source can include non-secret connector options:

```json
{
  "endpoint_url":"http://minio.storage.svc:9000",
  "credential_env_prefix":"RAGBOT_MINIO",
  "region_name":"us-east-1"
}
```

The worker resolves:

```text
RAGBOT_MINIO_ACCESS_KEY_ID
RAGBOT_MINIO_SECRET_ACCESS_KEY
RAGBOT_MINIO_SESSION_TOKEN
```

In production, every custom S3-compatible endpoint must be explicitly allowlisted:

```bash
RAGBOT_S3_ALLOWED_HOSTS=minio.storage.svc
```

This permits an intentionally private MinIO endpoint without reopening unrestricted worker-side SSRF. Native AWS S3 usage without a custom `endpoint_url` does not need this application allowlist.

## 5. Backlog-driven worker autoscaling with KEDA

The Helm chart can optionally create a KEDA `ScaledObject` for the ingestion worker. KEDA's PostgreSQL scaler reads `POSTGRES_DSN` from the target worker and queries the actual durable queue.

Example values:

```yaml
worker:
  enabled: true
  schedulerScanSeconds: "30"
  autoscaling:
    enabled: true
    minReplicaCount: 1
    maxReplicaCount: 10
    pollingInterval: 15
    cooldownPeriod: 120
    targetJobsPerWorker: "4"
    activationJobs: "1"
```

The scaling query counts:

- pending Jobs whose `available_at` is ready; and
- running Jobs whose worker lease has expired and is therefore reclaimable.

This makes worker scaling backlog-driven instead of using CPU as an indirect proxy for queue pressure.

KEDA CRDs/operator must already be installed when `worker.autoscaling.enabled=true`. The option is disabled by default, so ordinary Kubernetes/Helm deployments have no KEDA dependency.

## 6. Operational interpretation

Useful signals:

| Signal | Meaning | Typical response |
| --- | --- | --- |
| pending increases steadily | ingestion capacity below arrival rate | increase workers / enable KEDA / inspect embedding provider latency |
| oldest pending age increases | knowledge freshness SLA degrading | scale workers or reduce source frequency |
| stale running leases > 0 | worker crash/network/process interruption | verify workers; reclaim occurs through durable queue |
| failed 24h increases | connector/provider/config regression | inspect Job error, fix config, retry |
| scheduled Sources high but queue stable | expected recurring workload | no action |

## 7. Connector roadmap and secret policy

S3/MinIO is the first remote object-store connector in the product control plane. Google Drive, Notion and Confluence should follow the same rule:

- Source configuration stores resource identifiers and a **secret reference**, never OAuth/access-token material itself;
- worker deployments resolve that secret through environment/Kubernetes/secret-manager integration;
- catalog and Job APIs do not mirror secret-bearing connector config;
- refresh and rate-limit behavior stays inside the connector adapter;
- scheduled sync uses the same durable scheduler rather than connector-specific cron processes.

This contract is intentionally established before adding the SaaS connectors so they do not create incompatible credential models.
