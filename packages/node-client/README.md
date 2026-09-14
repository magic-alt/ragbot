# @ragbot/client

Typed Node.js client for Ragbot's versioned `/v1` public API.

```ts
import { RagbotClient } from "@ragbot/client";

const client = new RagbotClient("http://localhost:8000", {
  apiKey: process.env.RAGBOT_API_KEY,
});

const result = await client.search({
  query: "EtherCAT distributed clocks",
  tenant_id: "engineering",
  user_id: "alice",
  plan: "hybrid_rrf",
  top_k: 10,
});

console.log(result.request_id, result.chunks);
```

The client supports `AbortSignal`, per-call timeouts, typed `RagbotApiError`,
keyset pagination helpers, retry handling for retryable query/read operations,
and SSE chat streaming. Legacy free functions `chat()` and `chatStream()` remain
available while applications migrate to `RagbotClient`.
