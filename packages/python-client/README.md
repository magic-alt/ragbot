# ragbot-client

Typed Python client for Ragbot's versioned `/v1` public API.

```python
from ragbot_client import RagbotClient

with RagbotClient("http://localhost:8000", api_key="...") as client:
    result = client.search({
        "query": "EtherCAT distributed clocks",
        "tenant_id": "engineering",
        "user_id": "alice",
        "plan": "hybrid_rrf",
        "top_k": 10,
    })
    print(result["request_id"], result["chunks"])
```

The SDK exposes sync and async clients, typed API errors, per-call timeouts,
retry handling for retryable read/query operations, keyset-pagination helpers,
and SSE chat streaming. Cancelling an asyncio task using `AsyncRagbotClient`
propagates cancellation to the underlying HTTP request.

`RagbotApiError` exposes `status_code`, stable `code`, `request_id`,
`retryable`, and structured `details` from the `/v1` error envelope.
