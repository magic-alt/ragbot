# Ragbot API Guide

## Canonical specification

The running FastAPI application is the canonical HTTP contract:

- Swagger UI: `GET /docs`
- OpenAPI JSON: `GET /openapi.json`
- Offline export: `python scripts/export_openapi.py --output build/openapi.json`

This guide explains product and security semantics that are easier to understand in prose.

## Authentication, trusted identity, and RBAC

If `RAGBOT_API_KEYS` is non-empty, protected endpoints require `X-API-Key`.

`RAGBOT_API_KEY_PRINCIPALS` maps each API key to trusted tenant/user/groups/roles/admin state. Production startup requires complete principal coverage for configured API keys. Request payloads and headers cannot expand the tenant/user/ACL scope beyond the principal.

v1 tenant roles:

| Principal | Intended capability |
| --- | --- |
| `reader` | retrieval/chat/catalog/job read |
| `operator` | reader capabilities plus Source, ingestion, retry/requeue and schedule mutations |
| `owner` | tenant-level superset of operator |
| `admin=true` | global operational/admin surfaces; also bypasses tenant role checks |

Development mode without principal mappings remains backward compatible and exposes owner/admin-like local capabilities. Do not use that mode as an authorization model in production.

Probe endpoints `/admin/health` and `/admin/ready` remain available for liveness/readiness checks.

### `GET /catalog/session`

Returns non-secret capability metadata for the authenticated API key. The Admin UI uses this endpoint to render read-only vs operator controls without exposing principal configuration or secret values.

Representative reader response:

```json
{
  "principal_mode": "scoped",
  "admin": false,
  "roles": ["reader"],
  "tenant_ids": ["engineering"],
  "capabilities": {
    "read": true,
    "operate": false,
    "admin": false
  }
}
```

## Source types

Ingestible source types:

- `local_fs`
- `pdf`
- `web`
- `repo`
- `s3`
- `gdrive`
- `notion`
- `confluence`

SQL querying is separate from ingestion and uses `POSTGRES_DSN` plus allowed-schema policy.

## Product ingestion API

### `POST /ingest/quick`

Requires `operator`, `owner`, or global admin when scoped principals are enabled.

Creates/reuses a Source and submits ingestion in one request:

```json
{
  "tenant_id": "engineering",
  "location": "/data/manuals",
  "name": "Engineering manuals",
  "tags": ["manuals"]
}
```

Important fields:

- `tenant_id`: required;
- `location`: local path, URL, or connector product URI;
- `source_type`: optional; inferred when possible;
- `name`, `tags`, `acl_policy_id`: optional Source metadata;
- `config`: non-secret connector configuration;
- `reuse_source`: default `true`;
- `sync_source_metadata`: default `true`;
- `dedupe_active_job`: default `true`;
- `idempotency_key`: optional deterministic request idempotency key.

Default Source identity derives from:

```text
tenant_id + source_type + canonicalized location
```

Same-config pending/running Jobs may be reused as a convenience. If an active Job exists with a different connector config, the request returns `409` rather than mutating the Source and claiming the old Job represents the new request.

`idempotency_key` derives a deterministic Job ID and is the strict replay mechanism across API replicas; it requires `reuse_source=true`.

### `POST /ingest/batch`

Requires operator capability. Submits 1–100 Quick Import specifications for one tenant. Each item has an independent result; inspect `failed` and `items` even when the HTTP response is `202`.

## Cloud/SaaS credential contract

Google Drive, Notion, and Confluence Source configuration stores a reference, never the credential value:

```json
{
  "credential_ref": "env:RAGBOT_NOTION_TOKEN"
}
```

The API validates the `env:VARIABLE` reference but does not resolve the secret. The ingestion worker resolves it at execution time. Inline access/refresh tokens, API keys, passwords, private keys, and client secrets are rejected for SaaS source types.

See `docs/CLOUD_CONNECTORS.md` for connector-specific configuration.

## Low-level Sources API

Source mutation routes require operator capability under scoped principals.

### `POST /sources`

Valid source/config contracts:

| source_type | required config | notes |
| --- | --- | --- |
| `local_fs` | `path` | mounted text/Markdown tree |
| `pdf` | `path` | local or remote PDF |
| `web` | `url` | web content |
| `repo` | `path` | repository path/URL; optional `ref` |
| `s3` | `bucket` | optional `prefix`, endpoint/region options |
| `gdrive` | `folder_id`, `credential_ref` | optional `credential_type=access_token|google_json` |
| `notion` | `page_id`, `credential_ref` | optional recursive traversal/API version |
| `confluence` | `base_url`, `space_key`, `credential_ref` | basic auth also requires email; bearer supported |

### `GET /sources`

Lists non-deleted Sources within authorized tenant scope.

### `GET /sources/{source_id}`

Returns one Source after tenant authorization. This low-level endpoint includes Source config; operator/control-plane UIs should prefer redacted catalog APIs.

### `PUT /sources/{source_id}`

Updates name, connector config, status, ACL policy, or tags. A queued Job retains the immutable `source_type/source_config` snapshot captured when submitted.

### `PUT /sources/{source_id}/sync`

Configures recurring synchronization:

```json
{
  "enabled": true,
  "interval_seconds": 3600,
  "run_immediately": false
}
```

Scheduled Jobs use deterministic IDs plus atomic insert-if-absent; missed intervals collapse into one current refresh. Active ingestion for the same Source delays the scheduled refresh.

### `DELETE /sources/{source_id}`

Purges indexed Qdrant vectors and PostgreSQL Documents/Chunks and tombstones the Source.

## Durable ingestion Jobs

### `POST /ingest/jobs`

Requires operator capability. Queues an active Source. PostgreSQL-backed production persists a pending Job and an independent worker claims it with lease/heartbeat/recovery semantics.

### `GET /ingest/jobs`

Lists Jobs with optional tenant/source filters.

### `GET /ingest/jobs/{job_id}`

Gets a Job after tenant authorization. Low-level responses include the immutable connector snapshot; catalog responses redact it.

Important reliability fields include:

- `status`;
- `attempts`;
- `available_at`;
- lease/heartbeat timestamps;
- `error`;
- `failure_class`;
- `dead_lettered_at`;
- attempt/reuse/write statistics.

### Job state model

```text
pending
  ↓ claim
running
  ├─ success → completed
  ├─ retryable failure + attempts remaining
  │      → pending (durable exponential backoff)
  └─ permanent/exhausted failure
         → dead_lettered
```

Provider HTTP clients perform a separate short retry layer for 408/425/429/5xx and transport errors before the whole ingestion attempt is returned to the durable queue. `Retry-After` is honored when present.

### `POST /ingest/jobs/{job_id}/retry`

Requires operator capability. Creates a fresh Job from the **current Source config** and is valid only for legacy/intermediate `failed` Jobs.

### `POST /ingest/jobs/{job_id}/requeue`

Requires operator capability. Requeues a `dead_lettered` Job.

Default request:

```json
{
  "use_current_source_config": false
}
```

The default replays the immutable dead-letter Job snapshot. Set `use_current_source_config=true` only when an operator deliberately wants the repaired/current Source configuration.

## Product control plane

### `GET /catalog/overview`

Tenant-scoped Source/knowledge/queue summary. Queue fields include `pending`, `running`, `failed`, `dead_lettered`, stale lease count, oldest pending age, and 24-hour completion/failure/DLQ counts.

### `GET /catalog/sources`

Redacted Source Catalog. Full connector config and credential references are not returned.

### `GET /catalog/jobs`

Redacted Job history/progress. `source_config` is removed; failure class and DLQ metadata remain visible for operations.

### `GET /catalog/session`

Returns current principal roles/capabilities for UI behavior. It contains no API key, secret values, or connector config.

### `GET /admin/overview`

Global summary; requires `admin=true` when scoped principals are enabled.

### `GET /admin/queue/metrics`

Global queue/backlog metrics, including DLQ counts; admin required.

### `POST /admin/queue/reconcile`

Admin-only queue repair surface. It reconciles expired running leases and stranded failure state according to the configured max-attempt contract.

Example:

```bash
curl -X POST 'https://ragbot.example.com/admin/queue/reconcile?max_attempts=3' \
  -H "X-API-Key: $RAGBOT_ADMIN_KEY"
```

### `GET /admin/ui`

Built-in zero-build control plane. The API key is stored only in browser `sessionStorage`. The UI displays principal role/capability, disables write controls for readers, shows Dead Lettered counts/failure classes, supports failed Retry and DLQ Requeue, and never requests inline SaaS credential values.

## Incremental cloud synchronization

Drive/Notion/Confluence refreshes are metadata-first:

1. enumerate remote metadata;
2. compare `external_id + remote_version` with previous chunks;
3. reuse unchanged documents without content download/embedding;
4. fetch/chunk/embed changed or new documents;
5. prune remote deletions only after the replacement snapshot succeeds.

Current implementation still enumerates the configured remote tree/space. It is **not yet a provider delta/change-feed implementation**. Drive Changes API / equivalent persistent cursor work belongs to a separate optimization milestone.

## Search and Chat

### `POST /search`

Direct hybrid retrieval without the Agent loop. Production combines Qdrant vector search + PostgreSQL FTS/CJK lexical search + RRF and optional reranking. Tenant/ACL scope comes from the trusted principal.

### `POST /chat`

Agentic RAG route → tool → synthesize → verify flow. Streaming uses SSE; document retrieval supports all ingestible source types.

### `POST /v1/chat/completions`

OpenAI-compatible adapter. Tenant/user context remains constrained by the API-key principal.

## Other admin endpoints

- `GET /admin/health`: process liveness;
- `GET /admin/ready`: repository/vector readiness;
- `GET /admin/metrics`: aggregate quality/runtime metrics;
- `GET /admin/metrics/history`: recent metric history;
- `POST /admin/feedback`: feedback for a known request;
- `GET /admin/cost`: model-router cost summary;
- `GET /admin/cache`: cache statistics.

## Source/network security boundary

Production connector policy includes:

- Web/remote PDF/remote Git block non-public destinations by default and revalidate redirects;
- local Sources must stay below `RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS`;
- S3-compatible custom endpoints require explicit allowlisting;
- Confluence production base URLs require `RAGBOT_CONFLUENCE_ALLOWED_HOSTS`;
- private/self-hosted endpoints require explicit private-network opt-in when applicable;
- remote downloads use hard byte limits;
- SaaS secret values live in worker environment/secret stores, not Source config.

Application validation complements rather than replaces VPC/firewall/service-mesh egress policy.
