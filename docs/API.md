# Ragbot API Guide

## Canonical specification

The running FastAPI application is the canonical HTTP contract:

- Swagger UI: `GET /docs`
- OpenAPI JSON: `GET /openapi.json`
- Offline export: `python scripts/export_openapi.py --output build/openapi.json`

Do not hand-maintain a second OpenAPI schema. This guide explains product semantics that are easier to understand in prose.

## Authentication and trusted identity

If `RAGBOT_API_KEYS` is non-empty, protected endpoints require `X-API-Key`.

When `RAGBOT_API_KEY_PRINCIPALS` is configured, an API key maps to a trusted principal containing allowed tenant IDs, stable user ID, groups, roles and optional admin status. Application routes authorize the requested tenant against that mapping; `/search`, `/chat` and the OpenAI-compatible adapter derive ACL scope from the trusted principal rather than allowing request fields to expand access.

Production startup requires principal coverage for configured API keys. Development mode can run without principals for local testing.

`/admin/health` and `/admin/ready` are probe endpoints and do not require the application API key. Global operational endpoints such as metrics/cost/cache require an admin principal when scoped principals are enabled.

## Product ingestion API

The recommended onboarding surface is Quick Import. The lower-level Source and Job APIs remain available when callers need explicit lifecycle orchestration.

### `POST /ingest/quick`

Creates or reuses a Source and submits ingestion in one request.

Example:

```json
{
  "tenant_id": "engineering",
  "location": "/data/manuals",
  "name": "Engineering manuals",
  "tags": ["manuals"]
}
```

Fields:

- `tenant_id`: required tenant.
- `location`: required local path or remote URL.
- `source_type`: optional `local_fs`, `pdf`, `web` or `repo`; inferred when omitted.
- `name`, `tags`, `acl_policy_id`: optional Source metadata.
- `config`: optional connector configuration such as repository `ref` or chunking options.
- `reuse_source`: default `true`.
- `sync_source_metadata`: default `true`.
- `dedupe_active_job`: default `true`.
- `idempotency_key`: optional explicit request idempotency key.

Source type inference:

| location | inferred type |
| --- | --- |
| local `*.pdf` | `pdf` |
| local `*.git` | `repo` |
| other local path | `local_fs` |
| HTTP(S) `*.pdf` | `pdf` |
| GitHub/GitLab/Bitbucket or `*.git` URL | `repo` |
| other HTTP(S) URL | `web` |

Default Source identity is derived from:

```text
tenant_id + source_type + canonicalized location
```

This allows deployment/bootstrap scripts to submit the same knowledge source repeatedly without creating an unbounded number of Source rows.

If a matching Source already exists and no incompatible ingestion is active, Quick Import can reuse it and synchronize the connector config plus explicitly supplied name/tags/ACL metadata.

If a pending/running Job already exists with the **same** connector configuration, the default convenience dedupe returns that Job. If an active Job exists with a **different** connector configuration, such as another Git `ref`, path or chunking configuration, the new request returns `409` instead of mutating the Source and pretending the old Job represents the new request.

Representative response:

```json
{
  "status": "accepted",
  "source_id": "...",
  "source_type": "local_fs",
  "source_reused": false,
  "job_id": "...",
  "job_status": "pending",
  "job_reused": false
}
```

Possible submission status values include:

- `accepted`: a Job was submitted.
- `already_queued`: an active same-config Job was reused.
- `idempotent_replay`: the exact Job associated with the explicit idempotency key already exists.

#### Strict idempotency

`idempotency_key` derives a deterministic Job ID from stable Source identity + caller key. A replay is checked before Source metadata mutation and returns the exact existing Job even after completion.

`idempotency_key` requires `reuse_source=true`; the incompatible combination is rejected with `422` before persisting Source/Job state.

The ordinary active-Job lookup is a duplicate-prevention convenience, not a distributed uniqueness guarantee. If callers require strict repeat-request behavior across multiple API replicas or genuinely concurrent requests, they should provide an `idempotency_key`.

### `POST /ingest/batch`

Submits 1–100 Quick Import specifications under one tenant.

```json
{
  "tenant_id": "engineering",
  "sources": [
    {"location": "/data/manuals"},
    {"location": "https://example.com/guide.pdf"},
    {
      "location": "https://github.com/magic-alt/ragbot",
      "config": {"ref": "main"}
    }
  ]
}
```

Response contains `total`, `accepted`, `failed` and one result per input item. Expected request/config errors are isolated per item so callers can see which sources were accepted. Unexpected internal submission failures are logged server-side and returned as a generic error rather than exposing internal exception details.

A `202` batch response does not imply every item succeeded; inspect `failed` and `items`. For large catalogs, send multiple bounded batches instead of bypassing the 100-source validation limit.

## Low-level Sources API

### `POST /sources`

Creates a Source directly. Valid source/config pairs:

| source_type | required config | purpose |
| --- | --- | --- |
| `local_fs` | `path` | local mounted text/Markdown tree |
| `pdf` | `path` | local or remote PDF path/URL |
| `web` | `url` | web content |
| `repo` | `path` | repository path/URL; optional `ref` |

Invalid or missing required config fails before a Source is persisted.

### `GET /sources`

Lists non-deleted sources. Optional `tenant_id` filter. Principal-enabled deployments restrict results to authorized tenant scope.

### `GET /sources/{source_id}`

Gets one active/paused Source after tenant authorization.

### `PUT /sources/{source_id}`

Updates name, config, status (`active` or `paused`), ACL policy or tags. Replacement config is validated against the existing source type.

A queued durable Job does **not** read mutable connector config at execution time. Its `source_type` and `source_config` are captured at submission and executed from that snapshot. Therefore changing a Source after a Job is queued affects future Jobs, not the connector configuration of the already queued Job.

Current Source metadata/ACL state is still resolved at worker execution, so security-policy changes can apply to work that has not yet completed.

### `DELETE /sources/{source_id}`

Purges indexed Qdrant vectors and PostgreSQL/in-memory Documents/Chunks, then tombstones the Source. Cleanup occurs before the status transition so a cleanup failure leaves the Source retryable rather than silently searchable after deletion.

## Durable ingestion jobs

### `POST /ingest/jobs`

Queues an ingestion run for an **active** Source. This is the lower-level alternative to `/ingest/quick`.

- paused Source → `409`
- deleted/missing Source → `404`
- tenant mismatch/unauthorized tenant → `403`

With PostgreSQL, the API persists a `pending` Job and returns `202`; an independent worker owns execution. Workers claim jobs atomically with `FOR UPDATE SKIP LOCKED`, maintain lease/heartbeat state, reclaim expired leases after crashes and stop retrying after the configured maximum attempt count.

At submission the Job records `source_type` and `source_config`. The worker reconstructs connector execution from this snapshot instead of silently switching to a later mutable Source config.

Development/in-memory mode can execute inline for convenience. Production mode rejects inline ingestion.

### `GET /ingest/jobs`

Lists Jobs with optional tenant/source filters and principal tenant isolation.

### `GET /ingest/jobs/{job_id}`

Gets one Job after tenant authorization. Useful for CLI/UI progress polling.

Important fields include:

- `status`: `pending`, `running`, `completed`, `failed`
- `source_type`, `source_config`
- `doc_count`, `chunk_count`
- `error`
- `attempts`
- `started_at`, `completed_at`, `created_at`
- `available_at`, `lease_owner`, `lease_expires_at`, `heartbeat_at`
- `stats`

Job stats include current document IDs/chunks, new writes, reused unchanged chunks and stale-data cleanup counts where applicable.

### `POST /ingest/jobs/{job_id}/retry`

Creates a fresh Job for a failed ingestion if the Source still exists and is active. The retry captures the Source's current connector configuration as the new Job snapshot.

## Search

### `POST /search`

Direct hybrid retrieval without running the Agent loop. Supports tenant/ACL scope plus source type, document ID, tag, path/URL prefix and time-range filters.

Production retrieval combines Qdrant vector search and PostgreSQL FTS/CJK lexical search, merges rankings with RRF and can optionally apply a cross-encoder reranker. Reranker provider failure falls back to RRF rather than making retrieval unavailable.

## Chat

### `POST /chat`

Core Agentic RAG endpoint.

Required fields:

- `query`
- `tenant_id`
- `user_id`

Optional fields:

- `session_id`
- `stream` (default `false`)
- `constraints`
- `client_context`

Supported document retrieval `source_types` are `pdf`, `web`, `repo`, `local_fs`. Database querying is a separate SQL tool configured by `POSTGRES_DSN`.

### Streaming

With `stream=true`, media type is `text/event-stream`. Events include:

- `tool_call`
- `tool_result`
- `token`
- `final`
- `error`

`final` contains the request ID, answer, citations, confidence and followups. Agent failures emit a sanitized `error` event and terminate the callback stream.

## OpenAI-compatible endpoint

### `POST /v1/chat/completions`

Accepts an OpenAI-style messages payload and supports streaming. Tenant/user context can be supplied through the adapter's supported headers/fields and is checked against the trusted principal when principal mode is active.

Usage fields that are not backed by authoritative provider token accounting must remain clearly identified as estimates; Ragbot should not present character-count approximations as provider token truth.

## Admin endpoints

- `GET /admin/health`: process liveness.
- `GET /admin/ready`: repository/vector dependency readiness; returns 503 if dependencies are not ready.
- `GET /admin/metrics`: aggregate request/retrieval/tool metrics.
- `GET /admin/metrics/history`: recent request metric history.
- `POST /admin/feedback`: positive/negative feedback for a known request.
- `GET /admin/cost`: router cost tracker summary.
- `GET /admin/cache`: retrieval/embedding cache status and statistics.

## Source security boundary

Production connector policy is part of the API contract:

- Web/remote PDF/remote Git reject loopback, private, link-local and reserved destinations by default;
- redirects are revalidated;
- optional per-connector hostname allowlists can narrow egress;
- local sources must be below `RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS`;
- Web/PDF downloads have byte limits;
- production remote Git uses HTTPS.

Application validation is not a replacement for VPC/firewall/service-mesh egress policy.
