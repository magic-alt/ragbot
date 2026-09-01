# Ragbot API Guide

## Canonical specification

The running FastAPI application is the canonical HTTP contract:

- Swagger UI: `GET /docs`
- OpenAPI JSON: `GET /openapi.json`
- Offline export: `python scripts/export_openapi.py --output build/openapi.json`

Do not hand-maintain a second OpenAPI document.

## Authentication and identity

If `RAGBOT_API_KEYS` is non-empty, protected endpoints require `X-API-Key`. The current key model is service-level authentication. `tenant_id` and `user_id` are supplied by the caller and are not cryptographically bound to the key. Put Ragbot behind a trusted gateway/identity layer for externally reachable multi-tenant deployments.

`/admin/health` and `/admin/ready` are probe endpoints and do not require the application API key.

## Chat

### `POST /chat`

Core agent endpoint.

Required fields:

- `query`: non-empty user query.
- `tenant_id`: non-empty tenant identifier.
- `user_id`: non-empty user identifier.

Optional fields:

- `session_id`
- `stream` (default `false`)
- `constraints`
- `client_context`

Supported document retrieval `source_types` are:

- `pdf`
- `web`
- `repo`
- `local_fs`

Database querying is a separate SQL tool configured by `POSTGRES_DSN`; `database`/`db_doc` are not ingestible document source types.

### Streaming

With `stream=true`, response media type is `text/event-stream`. Events include:

- `tool_call`
- `tool_result`
- `token`
- `final`
- `error`

`final` contains `request_id`, final answer, citations, confidence and followups. Agent failures emit a sanitized `error` event and always terminate the callback stream. `client_context` is processed identically in streaming and non-streaming requests.

## Search

### `POST /search`

Direct hybrid retrieval without running the full agent loop. Supports tenant, ACL/security scope, source type, document ID, tag, path/URL prefix and time-range filters. Production lexical search is executed in PostgreSQL and vector search in Qdrant, then merged/reranked by the retrieval service.

## Sources

### `POST /sources`

Creates a source. Valid source/config pairs:

| source_type | required config | purpose |
| --- | --- | --- |
| `local_fs` | `path` | local mounted text/Markdown tree |
| `pdf` | `path` | one PDF path |
| `web` | `url` | web content |
| `repo` | `path` | repository path/URL; optional `ref` |

Invalid or missing required config fails before a Source record is persisted.

### `GET /sources`

Lists non-deleted sources. Optional `tenant_id` filter. Without a tenant filter this is an administrative/service-level operation; do not expose it directly to untrusted tenant clients.

### `GET /sources/{source_id}`

Gets one active/paused source.

### `PUT /sources/{source_id}`

Updates name, config, status (`active` or `paused`), ACL policy or tags. Replacement config is validated against the existing source type.

### `DELETE /sources/{source_id}`

Purges the Source's indexed Qdrant vectors and PostgreSQL/in-memory Documents/Chunks, then tombstones the Source. Cleanup occurs before the status transition so a cleanup failure leaves the source retryable rather than silently searchable after deletion.

## Ingestion jobs

### `POST /ingest/jobs`

Queues an ingestion run for an **active** source. Paused sources return `409`; deleted/missing sources return `404`; tenant mismatch returns `403`.

The current executor is process-local. A `202 Accepted` response means the API process accepted the job, not that a durable external queue owns it. Deploy/restart can interrupt an in-flight job; failed/partial ingestion is designed to be reconciled on retry.

### `GET /ingest/jobs`

Lists jobs with optional tenant/source filters.

### `GET /ingest/jobs/{job_id}`

Gets one job.

### `POST /ingest/jobs/{job_id}/retry`

Retries a failed job if its Source still exists and is active.

Job stats include document IDs, total current chunks, newly ingested chunks, reused unchanged chunks and stale-data cleanup counts.

## OpenAI-compatible endpoint

### `POST /v1/chat/completions`

Accepts an OpenAI-style messages payload and supports streaming. Tenant/user context can be supplied with `X-Tenant-ID` and `X-User-ID`; otherwise `default`/`anonymous` are used.

Ragbot intentionally does not fabricate OpenAI `usage` token counts from character lengths. Provider-level token accounting should be exposed only when a model provider returns authoritative usage data.

## Admin endpoints

- `GET /admin/health`: process liveness.
- `GET /admin/ready`: repository/vector dependency readiness; returns 503 if not ready.
- `GET /admin/metrics`: aggregate request/retrieval/tool metrics.
- `GET /admin/metrics/history`: recent request metric history.
- `POST /admin/feedback`: positive/negative feedback for a known request.
- `GET /admin/cost`: cost tracker summary for the current router implementation.
- `GET /admin/cache`: retrieval/embedding cache status and statistics.

Except health/readiness, admin endpoints use the same API-key dependency as application routes.
