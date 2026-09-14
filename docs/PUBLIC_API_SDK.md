# Ragbot v1 public API and SDK contract

Issue #58 establishes a versioned developer boundary over Ragbot's internal
services. Internal Python classes are free to evolve; supported clients should
bind to `/v1` plus the maintained Python/Node SDKs.

## Versioning model

Phase 1 keeps existing endpoints for compatibility and mounts the same semantic
routers under `/v1`:

```text
legacy                    supported v1
/search                    /v1/search
/chat                      /v1/chat
/sources                   /v1/sources
/ingest/jobs               /v1/ingest/jobs
/catalog/jobs              /v1/catalog/jobs
...
```

The same router objects are mounted at both locations, so RBAC, tenant scoping,
keyset pagination, ingestion semantics and retrieval behavior remain
single-sourced. `/v1` is a contract façade, not a fork of the application.

New SDK code should use `/v1`. Legacy routes remain available until an explicit
deprecation/removal policy says otherwise.

Successful `/v1` responses include:

```text
X-Request-ID: <request correlation id>
X-Ragbot-API-Version: v1
```

## Stable error envelope

Versioned endpoints use one machine-readable shape:

```json
{
  "error": {
    "code": "not_found",
    "message": "Source not found",
    "request_id": "...",
    "retryable": false,
    "details": null
  }
}
```

Stable fields:

- `code`: SDK-switchable error class;
- `message`: human-readable summary;
- `request_id`: support/trace correlation identifier;
- `retryable`: whether generic SDK retry policy may retry the operation;
- `details`: optional structured information.

Validation errors use `validation_error`. Typical transient codes include
`rate_limited`, `upstream_error`, `service_unavailable`, and
`deadline_exceeded`.

Legacy endpoints retain the historical FastAPI `detail` response shape.

## Frozen Phase-1 operations

The machine-readable compatibility manifest lives at:

```text
contracts/public_api_v1_contract.json
```

It currently freezes the minimum supported surface used by the SDKs:

- `POST /v1/search`;
- `POST /v1/chat`;
- `GET /v1/sources`;
- `GET /v1/sources/{source_id}`;
- `GET /v1/catalog/jobs`;
- `POST /v1/ingest/jobs`.

`scripts/check_public_api_contract.py` resolves the generated OpenAPI schema and
fails CI if a required operation, request field, path/query parameter, or frozen
success-response field disappears.

This is intentionally a *minimum compatibility manifest*. Adding optional
fields or new operations remains compatible. Phase 2 can evolve this into a
full semantic-versioning diff policy before public package releases are broadly
advertised.

## Pagination

Collection APIs use existing keyset cursors rather than offset pagination.
Callers should treat cursors as opaque strings.

Example:

```text
GET /v1/sources?tenant_id=t1&limit=100
  -> next_cursor = "..."
GET /v1/sources?tenant_id=t1&limit=100&cursor=...
```

Both maintained SDKs expose cursor helpers so applications do not have to build
pagination loops manually.

## Python SDK

Package source:

```text
packages/python-client
```

Package name:

```text
ragbot-client
```

The SDK provides:

- `RagbotClient` synchronous client;
- `AsyncRagbotClient` asynchronous client;
- `RagbotApiError`;
- typed search/chat/source/job structures;
- search and chat;
- source get/list/iteration;
- job listing;
- SSE chat streaming;
- configurable timeout and bounded retry behavior.

Example:

```python
from ragbot_client import RagbotClient

with RagbotClient("https://ragbot.example", api_key="...") as client:
    response = client.search({
        "query": "EtherCAT distributed clocks",
        "tenant_id": "engineering",
        "user_id": "alice",
        "plan": "hybrid_rrf",
        "top_k": 10,
    })
    print(response["request_id"])
```

For asyncio applications, cancelling the task using `AsyncRagbotClient`
propagates cancellation through `httpx` instead of converting it into a retry.

## Node.js SDK

Package source:

```text
packages/node-client
```

Package name:

```text
@ragbot/client
```

The former private/typecheck-only package is now package-build capable. The
primary entry point is `RagbotClient`; historical free functions `chat()` and
`chatStream()` remain during migration.

Example:

```ts
import { RagbotClient } from "@ragbot/client";

const client = new RagbotClient("https://ragbot.example", {
  apiKey: process.env.RAGBOT_API_KEY,
});

const result = await client.search({
  query: "EtherCAT distributed clocks",
  tenant_id: "engineering",
  user_id: "alice",
  plan: "hybrid_rrf",
});
```

The Node client supports `AbortSignal`, per-call timeouts, typed errors,
keyset-pagination iteration, SSE parsing and retryable-query handling.

## Retry semantics

Generic SDK retry policy is deliberately conservative.

Safe query/read operations may retry transient transport failures and HTTP
429/502/503/504. The server's `Retry-After` value is honored when numeric.
Retries use bounded backoff.

Phase 1 does **not** claim generic mutation idempotency. Source creation,
ingestion mutation and upload mutation APIs therefore should not be blindly
replayed by application code after an ambiguous network failure. A first-class
`Idempotency-Key` server contract remains Phase-2 work.

## Cancellation and deadlines

There are two distinct budgets:

1. HTTP client timeout/cancellation — controls how long the caller waits;
2. retrieval `deadline_ms` — controls server-side retrieval stages.

For search requests, use a retrieval deadline shorter than the outer HTTP
client timeout when the application needs deterministic server-side cancellation
and a durable `deadline_exceeded` lineage record.

## Streaming

`/v1/chat` with `stream=true` uses SSE. SDK parsers support standard `event:`
and multiline `data:` fields. Clients should treat unknown future event names
as forward-compatible rather than crashing the stream.

Typical events include:

- `tool_call`;
- `tool_result`;
- `token`;
- `final`;
- `error`.

The final event contains the request ID and final answer/citations.

## Relationship to #61

The public `request_id` is also the durable observability key introduced by
#61. This gives SDK callers one identifier that can be used for:

- support/debugging;
- durable feedback;
- retrieval/index/model lineage lookup;
- evaluation and incident correlation.

The SDK does not know how #61 stores runs; it depends only on the stable HTTP
request ID contract.

## Phase-2 boundary

Issue #58 remains open after Phase 1 for:

- server-side idempotency keys for mutation endpoints;
- full Python/Node lifecycle parity for create/update/delete/sync/upload APIs;
- generated or validated SDK models from the published OpenAPI contract;
- formal semver/deprecation checks for additive vs breaking schema changes;
- scoped MCP/tool surface built over the same `/v1` contracts;
- automated PyPI/npm release workflow and signed provenance;
- end-to-end staging tests against a deployed multi-replica Ragbot service.
