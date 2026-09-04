# Ragbot API Guide

## Canonical specification

The running FastAPI application is the canonical HTTP contract:

- Swagger UI: `GET /docs`
- OpenAPI JSON: `GET /openapi.json`
- Offline export: `python scripts/export_openapi.py --output build/openapi.json`

This guide documents security and operational semantics that are easier to understand in prose.

## Authentication, trusted identity, and RBAC

If `RAGBOT_API_KEYS` is non-empty, protected endpoints require `X-API-Key`.

`RAGBOT_API_KEY_PRINCIPALS` maps each key to trusted tenant/user/groups/roles/admin state. Production startup requires complete principal coverage and every non-admin production principal must carry at least one platform role: `reader`, `operator`, or `owner`. Request payloads/headers cannot expand tenant or user identity beyond that principal.

Platform capabilities are explicit and hierarchical:

| Capability | reader | operator | owner | `admin=true` |
|---|:---:|:---:|:---:|:---:|
| `knowledge.query` | ✓ | ✓ | ✓ | ✓ |
| `catalog.read` | ✓ | ✓ | ✓ | ✓ |
| `feedback.write` | ✓ | ✓ | ✓ | ✓ |
| `source.create` |  | ✓ | ✓ | ✓ |
| `source.update` |  | ✓ | ✓ | ✓ |
| `source.sync` |  | ✓ | ✓ | ✓ |
| `ingestion.run` |  | ✓ | ✓ | ✓ |
| `ingestion.retry` |  | ✓ | ✓ | ✓ |
| `source.delete` |  |  | ✓ | ✓ |
| global admin/metrics/reconcile |  |  |  | ✓ |

`admin=true` is a global operational identity, not a synonym for tenant owner. Additional role strings may still be used by document ACL policies; custom ACL roles do **not** grant platform capabilities.

Development mode without principal mappings remains backward compatible and has unrestricted local tenant capabilities. Do not use that mode as a production authorization model.

### `GET /catalog/session`

Returns non-secret RBAC metadata for the authenticated principal. The response preserves the legacy UI summary and adds the authoritative capability set:

```json
{
  "principal_mode": "scoped",
  "admin": false,
  "roles": ["operator"],
  "tenant_ids": ["engineering"],
  "capabilities": {
    "read": true,
    "operate": true,
    "admin": false
  },
  "effective_capabilities": [
    "catalog.read",
    "feedback.write",
    "ingestion.retry",
    "ingestion.run",
    "knowledge.query",
    "source.create",
    "source.sync",
    "source.update"
  ],
  "role_capability_matrix": {
    "reader": ["catalog.read", "feedback.write", "knowledge.query"],
    "operator": ["... reader + non-destructive source/ingestion capabilities ..."],
    "owner": ["... operator + source.delete ..."]
  }
}
```

## Source types

Ingestible source types: `local_fs`, `pdf`, `web`, `repo`, `s3`, `gdrive`, `notion`, `confluence`.

Agent SQL is a separate, fail-closed capability. It uses an isolated `RAGBOT_SQL_DSN`, never the internal `POSTGRES_DSN` control-plane database.

## Product ingestion API

### `POST /ingest/quick`

Requires operator-or-owner ingestion capability. Creates/reuses a Source and submits ingestion in one request:

```json
{
  "tenant_id": "engineering",
  "location": "/data/manuals",
  "name": "Engineering manuals",
  "tags": ["manuals"]
}
```

Important fields: `tenant_id`, `location`, optional `source_type`, Source metadata, non-secret connector `config`, `reuse_source`, `sync_source_metadata`, `dedupe_active_job`, and optional `idempotency_key`.

Default Source identity derives from:

```text
tenant_id + source_type + canonicalized location
```

Same-config pending/running Jobs may be reused. A conflicting active connector config returns `409`. `idempotency_key` is the strict replay mechanism across API replicas and requires `reuse_source=true`.

### `POST /ingest/batch`

Requires operator-or-owner ingestion capability. Submits 1–100 Quick Import specifications for one tenant. Each item has an independent result; inspect `failed` and `items` even when HTTP status is `202`.

## Cloud/SaaS credential contract

Google Drive, Notion, and Confluence Source configuration stores a reference, never the secret value:

```json
{
  "credential_ref": "env:RAGBOT_NOTION_TOKEN"
}
```

The API validates the reference; the worker resolves it at execution time. Inline access/refresh tokens, API keys, passwords, private keys, and client secrets are rejected for SaaS source types.

## Low-level Sources API

### `POST /sources`

Requires `source.create` (operator/owner). Valid contracts:

| source_type | required config |
|---|---|
| `local_fs` | `path` |
| `pdf` | `path` |
| `web` | `url` |
| `repo` | `path` |
| `s3` | `bucket` |
| `gdrive` | `folder_id`, `credential_ref` |
| `notion` | `page_id`, `credential_ref` |
| `confluence` | `base_url`, `space_key`, `credential_ref` |

### `GET /sources`

Requires `catalog.read`. Lists non-deleted Sources in authorized tenant scope.

### `GET /sources/{source_id}`

Requires `catalog.read` after tenant authorization. The low-level endpoint includes Source config; product UIs should prefer redacted catalog APIs.

### `PUT /sources/{source_id}`

Requires `source.update`. Updates name/config/status/ACL/tags. Queued Jobs retain the immutable connector snapshot captured at submission.

### `PUT /sources/{source_id}/sync`

Requires `source.sync`. Configures recurring synchronization.

### `DELETE /sources/{source_id}`

Requires **`source.delete`**, therefore tenant `owner` or global admin. Operator is intentionally insufficient for destructive deletion.

Deletion first tombstones/advances Source generation, then purges PostgreSQL/Qdrant knowledge. Running or queued stale Jobs are fenced and cannot republish into the deleted lifecycle.

## Durable ingestion Jobs

### `POST /ingest/jobs`

Requires `ingestion.run`. PostgreSQL production persists a pending Job; an independent worker claims it with lease/heartbeat/recovery semantics.

### `GET /ingest/jobs` and `GET /ingest/jobs/{job_id}`

Require `catalog.read` plus tenant authorization.

Important reliability fields include `status`, `attempts`, `available_at`, lease/heartbeat timestamps, `error`, `failure_class`, `dead_lettered_at`, immutable connector snapshot, and `stats.source_generation`.

State model:

```text
pending
  ↓ claim
running
  ├─ success → completed
  ├─ retryable failure + attempts remaining → pending(backoff)
  └─ permanent/exhausted/source-generation failure → dead_lettered
```

### `POST /ingest/jobs/{job_id}/retry`

Requires `ingestion.retry`. Creates a fresh Job from current Source config and applies only to `failed` Jobs.

### `POST /ingest/jobs/{job_id}/requeue`

Requires `ingestion.retry`. Requeues a `dead_lettered` Job. Default behavior replays the dead-letter snapshot; `use_current_source_config=true` intentionally adopts repaired current Source config.

The production queue implementation is the repository/PostgreSQL lease contract. The old `services/worker/queue.py` abstraction has been removed.

## Product control plane

The following tenant catalog endpoints require `catalog.read`:

- `GET /catalog/overview`
- `GET /catalog/sources`
- `GET /catalog/jobs`
- `GET /catalog/session`

Global surfaces require `admin=true`:

- `GET /admin/overview`
- `GET /admin/queue/metrics`
- `POST /admin/queue/reconcile`
- `GET /metrics`
- `GET /admin/metrics`
- `GET /admin/metrics/history`
- `GET /admin/cost`

`POST /admin/feedback` is allowed by `feedback.write`, but its request-id lookup is intentionally process-local diagnostic history. If the request was handled by another replica the endpoint may return `404`; production feedback persistence remains a separate data-model concern.

## Search and Chat

The following require `knowledge.query`:

- `POST /search`
- `POST /chat`
- `POST /v1/chat/completions`

Search uses tenant/ACL scope derived from the trusted principal. The normal default is adaptive hybrid retrieval over Qdrant vector search + PostgreSQL lexical/CJK candidates, followed by optional reranking.

### `POST /search` retrieval-quality controls

```json
{
  "query": "What techniques lower VRAM consumption?",
  "tenant_id": "engineering",
  "user_id": "researcher",
  "top_k": 10,
  "mode": "hybrid",
  "candidate_pool": 50,
  "rerank": false,
  "explain": true
}
```

Fields added for controlled retrieval experiments:

| Field | Default | Semantics |
|---|---|---|
| `mode` | `hybrid` | `vector`, `lexical`, or `hybrid`; enables true first-stage ablation |
| `candidate_pool` | configured/default | pre-rerank recall budget, 1–200; does not change final `top_k` |
| `rerank` | `true` | apply the configured reranker; set `false` to isolate vector/lexical/fusion behavior |
| `explain` | `false` | diagnostic intent flag; structured retrieval traces remain backward compatible |

Hybrid fusion uses an observable adaptive RRF policy. Weak lexical evidence shifts weight toward semantic vectors. When a CJK query retrieves an English corpus and lexical evidence effectively comes from residual ASCII terms such as `GPU`, the lexical branch is capped at 10% authority instead of receiving unconditional 50/50 fusion weight.

Per-result `metadata._retrieval` includes vector and lexical ranks/raw scores, pre-rerank/RRF score, optional reranker score, embedding model, fusion method and request context. Existing `vector.score`, `lexical.score`, and `rrf_score` keys remain available for compatible evaluators.

Top-level `diagnostics` includes embedding backend/model/dimension, whether semantic embeddings are active, candidate counts, resolved candidate pool, fusion weights/reason, and whether the reranker is configured/requested/applied.

The OpenAI adapter preserves system/history context; the last non-empty user turn remains the active retrieval query. `temperature` and `max_tokens` are propagated to synthesis. Current SSE is final-answer chunk streaming, not provider-native token streaming; token usage is estimated.

## Metrics and diagnostics

### `GET /metrics`

Admin-protected Prometheus exposition endpoint. Production Agent metrics are emitted at event time as counters/histograms rather than reconstructed from one process's rolling history. Representative metrics:

- `ragbot_agent_requests_total`
- `ragbot_agent_request_duration_seconds`
- `ragbot_agent_retrieval_duration_seconds`
- `ragbot_agent_tool_calls_total`
- `ragbot_agent_tool_duration_seconds`
- `ragbot_agent_feedback_total`
- `ragbot_http_requests_total`
- `ragbot_ingestion_jobs`
- `ragbot_ingestion_oldest_pending_age_seconds`
- `ragbot_ingestion_stale_running_leases`

Prometheus aggregates counters/histograms across replicas. Queue/Source gauges are refreshed from the shared repository during scrape.

### OpenTelemetry metrics

Set:

```dotenv
RAGBOT_OTEL_METRICS_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
```

Agent request/tool/latency/feedback metrics are exported via OTLP using the OpenTelemetry SDK.

### `/admin/metrics` and `/admin/metrics/history`

These are explicitly **process-local diagnostics**, retained for recent request inspection and feedback correlation. They are not the production metrics backend.

`GET /admin/cache` is retained only as an admin-protected deprecated compatibility tombstone. It returns `enabled=false` / `retired=true`, exposes no runtime cache state, and cannot enable or clear caching. The local cache primitives in `services/api/app/cache/` are experimental/test utilities and are not connected to retrieval.

## CLI ownership

`cli/rag.py` is the single product CLI implementation behind both `rag` and `python -m cli.rag`. `scripts/ragbot.py` is a bootstrap/deployment controller and delegates product commands to `cli.rag`. The old `cli/rag_impl.py` and `scripts/ragbot_impl.py` indirection files have been removed.

## Source/network security boundary

Production connector policy blocks non-public remote destinations by default, revalidates redirects, constrains local paths to configured roots, applies hard download byte limits, and keeps SaaS secrets in worker environment/secret stores rather than Source config. Application validation complements rather than replaces VPC/firewall/service-mesh egress policy.
