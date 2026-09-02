# Ragbot API Guide

## Canonical specification

The running FastAPI application is the canonical HTTP contract:

- Swagger UI: `GET /docs`
- OpenAPI JSON: `GET /openapi.json`
- Offline export: `python scripts/export_openapi.py --output build/openapi.json`

Do not hand-maintain a second OpenAPI schema. This guide explains product semantics that are easier to understand in prose.

## Authentication and trusted identity

If `RAGBOT_API_KEYS` is non-empty, protected endpoints require `X-API-Key`.

When `RAGBOT_API_KEY_PRINCIPALS` is configured, an API key maps to a trusted principal containing allowed tenant IDs, stable user ID, groups, roles and optional admin status. Routes authorize the requested tenant against that mapping; `/search`, `/chat` and the OpenAI-compatible adapter derive ACL scope from the trusted principal rather than allowing request fields to expand access.

Production startup requires principal coverage for configured API keys. `/admin/health` and `/admin/ready` remain probe endpoints. Global operational endpoints require an admin principal when scoped principals are enabled.

## Source types

Ingestible source types are:

- `local_fs`
- `pdf`
- `web`
- `repo`
- `s3`
- `gdrive`
- `notion`
- `confluence`

SQL database querying is separate from ingestion and is configured through `POSTGRES_DSN`.

## Product ingestion API

### `POST /ingest/quick`

Creates/reuses a Source and submits ingestion in one request.

```json
{
  "tenant_id": "engineering",
  "location": "/data/manuals",
  "name": "Engineering manuals",
  "tags": ["manuals"]
}
```

Fields:

- `tenant_id`: required.
- `location`: local path, URL or connector product URI.
- `source_type`: optional; inferred when possible.
- `name`, `tags`, `acl_policy_id`: optional Source metadata.
- `config`: connector configuration.
- `reuse_source`: default `true`.
- `sync_source_metadata`: default `true`.
- `dedupe_active_job`: default `true`.
- `idempotency_key`: optional explicit request idempotency key.

Common inference:

| location | inferred type |
| --- | --- |
| local `*.pdf` | `pdf` |
| local `*.git` | `repo` |
| other local path | `local_fs` |
| HTTP(S) `*.pdf` | `pdf` |
| GitHub/GitLab/Bitbucket or `*.git` URL | `repo` |
| `s3://bucket/prefix` | `s3` |
| `gdrive://folder-id` or Drive folder URL | `gdrive` |
| `notion://page-id` or Notion page URL | `notion` |
| `confluence://host/SPACE` or Atlassian space URL | `confluence` |
| other HTTP(S) URL | `web` |

Default Source identity is derived from:

```text
tenant_id + source_type + canonicalized location
```

If a same-config pending/running Job already exists, convenience dedupe can return it. If an active Job exists with a different connector config, the new request returns `409` rather than rewriting the Source and pretending the old Job represents the new request.

`idempotency_key` derives a deterministic Job ID and is the strict repeat-request mechanism across replicas. It requires `reuse_source=true`.

Representative response:

```json
{
  "status": "accepted",
  "source_id": "...",
  "source_type": "gdrive",
  "source_reused": false,
  "job_id": "...",
  "job_status": "pending",
  "job_reused": false
}
```

### SaaS credential contract

Google Drive, Notion and Confluence never accept credential values as product configuration. Store only a reference:

```json
{
  "credential_ref": "env:RAGBOT_NOTION_TOKEN"
}
```

The API validates `env:VARIABLE` syntax without resolving it. The worker resolves the secret at execution time. This lets SaaS credentials exist only in worker pods/containers.

For SaaS source types Ragbot rejects common inline secret fields including access/refresh tokens, API keys, passwords, private keys and client secrets.

Connector examples:

Google Drive:

```json
{
  "tenant_id": "engineering",
  "location": "gdrive://1AbCdEfFolder",
  "config": {
    "credential_ref": "env:RAGBOT_DRIVE_CREDENTIALS_JSON",
    "credential_type": "google_json"
  }
}
```

Notion:

```json
{
  "tenant_id": "engineering",
  "location": "notion://0123456789abcdef0123456789abcdef",
  "config": {
    "credential_ref": "env:RAGBOT_NOTION_TOKEN"
  }
}
```

Confluence:

```json
{
  "tenant_id": "engineering",
  "location": "confluence://acme.atlassian.net/ENG",
  "config": {
    "credential_ref": "env:RAGBOT_CONFLUENCE_TOKEN",
    "email": "ragbot@example.com",
    "auth_type": "basic"
  }
}
```

Full connector semantics are documented in `docs/CLOUD_CONNECTORS.md`.

### `POST /ingest/batch`

Submits 1–100 Quick Import specifications under one tenant. Each item gets an independent result. A `202` batch response does not imply every item was accepted; inspect `failed` and `items`.

## Low-level Sources API

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
| `confluence` | `base_url`, `space_key`, `credential_ref` | basic auth also requires `email`; bearer supported |

Invalid/missing config is rejected before Source persistence.

### `GET /sources`

Lists non-deleted sources within authorized tenant scope.

### `GET /sources/{source_id}`

Returns one Source after tenant authorization. This low-level endpoint includes Source config, so use it only with appropriately authorized callers. Product catalog endpoints provide a safer redacted view for operator UIs.

### `PUT /sources/{source_id}`

Updates name, connector config, status, ACL policy or tags. Replacement config is revalidated. A queued durable Job keeps the immutable `source_type/source_config` snapshot captured when it was submitted.

### `PUT /sources/{source_id}/sync`

Enables/disables periodic synchronization.

```json
{
  "enabled": true,
  "interval_seconds": 3600,
  "run_immediately": false
}
```

- minimum interval is 60 seconds;
- due Sources enter the normal durable queue;
- deterministic scheduled Job IDs plus atomic insert-if-absent make concurrent scheduler scans safe;
- missed historical windows collapse to one current refresh instead of backfilling an ingestion storm;
- active ingestion for the same Source delays scheduled refresh.

For Drive/Notion/Confluence, a scheduled refresh is metadata-first: unchanged remote versions reuse prior chunks and skip content download/embedding.

### `DELETE /sources/{source_id}`

Purges indexed Qdrant vectors and PostgreSQL Documents/Chunks, then tombstones the Source.

## Durable ingestion jobs

### `POST /ingest/jobs`

Queues ingestion for an active Source. With PostgreSQL, the API persists a pending Job and an independent worker claims it with lease/heartbeat/recovery semantics. Production rejects inline ingestion.

The Job records immutable connector config. Credential references may be present in that snapshot, but secret values are resolved only when a worker executes the Job.

### `GET /ingest/jobs`

Lists Jobs with optional tenant/source filters.

### `GET /ingest/jobs/{job_id}`

Gets one Job after tenant authorization. Low-level Job responses include `source_config`; product catalog endpoints deliberately remove it.

Important fields include status, counts, error, attempts, timing/lease data and `stats` such as chunks written/reused/removed.

### `POST /ingest/jobs/{job_id}/retry`

Creates a fresh Job for a failed ingestion if the Source still exists and is active. The new Job snapshots the Source's current connector config.

## Product control plane

### `GET /catalog/overview`

Tenant-scoped summary of Sources, indexed documents/chunks, queue state and schedules.

### `GET /catalog/sources`

Redacted Source Catalog. Supports tenant/status/source-type/search/limit filters. The response exposes a safe location such as `gdrive://...`, `notion://...` or `confluence://host/SPACE`, not full connector config or secret references.

### `GET /catalog/jobs`

Redacted ingestion progress/history. `source_config` is removed from product Job responses.

### `GET /admin/overview`

Global control-plane summary; admin principal required.

### `GET /admin/queue/metrics`

Admin queue/backlog metrics including pending/running/failed counts, oldest pending age, stale running leases, recent completion/failure activity and scheduled Source state.

### `GET /admin/ui`

Built-in zero-build operator UI. API key is kept in browser `sessionStorage`. Cloud Quick Import accepts a `credential_ref` and non-secret connector JSON; it explicitly warns operators not to paste token/private-key values.

## Incremental cloud synchronization semantics

The cloud connector result is still a complete replacement snapshot, so deletion/pruning semantics remain consistent with other Sources. The optimization happens before content download:

1. enumerate remote metadata;
2. compare `external_id + remote_version` against previous chunks;
3. unchanged document -> return reusable chunks;
4. changed/new document -> fetch content, chunk, embed and upsert;
5. absent remote document -> previous chunks/doc are pruned after the new snapshot succeeds.

Current remote version signals:

- Google Drive: `modifiedTime + version + md5Checksum`;
- Notion: page `last_edited_time`;
- Confluence: page version number + last-updated time.

## Search

### `POST /search`

Direct hybrid retrieval without the Agent loop. Supports tenant/ACL scope plus source type, document ID, tag, path/URL prefix and time-range filters.

Production retrieval combines Qdrant vector search and PostgreSQL FTS/CJK lexical search, RRF and optional reranking.

## Chat

### `POST /chat`

Core Agentic RAG endpoint. Required: `query`, `tenant_id`, `user_id`; optional session/stream/constraints/client context.

Document retrieval `source_types` may include all ingestible source types listed above.

With `stream=true`, SSE events include `tool_call`, `tool_result`, `token`, `final`, and `error`.

## OpenAI-compatible endpoint

### `POST /v1/chat/completions`

Accepts an OpenAI-style messages payload and supports streaming. Tenant/user context remains subject to trusted-principal authorization.

## Other admin endpoints

- `GET /admin/health`: process liveness.
- `GET /admin/ready`: storage/vector readiness.
- `GET /admin/metrics`: aggregate quality/runtime metrics.
- `GET /admin/metrics/history`: recent metric history.
- `POST /admin/feedback`: feedback for a known request.
- `GET /admin/cost`: model-router cost summary.
- `GET /admin/cache`: retrieval/embedding cache statistics.

## Source security boundary

Production connector policy is part of the contract:

- Web/remote PDF/remote Git block non-public destinations by default and revalidate redirects;
- local sources must stay below `RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS`;
- S3-compatible custom endpoints require explicit production host allowlisting;
- Confluence production base URLs require `RAGBOT_CONFLUENCE_ALLOWED_HOSTS`;
- private/self-hosted endpoints additionally require the explicit private-network opt-in when applicable;
- remote downloads use hard byte limits;
- SaaS secret values live in worker environment/secret stores, not Source config.

Application validation is not a replacement for VPC/firewall/service-mesh egress policy.
