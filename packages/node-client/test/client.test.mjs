import assert from "node:assert/strict";
import test from "node:test";

import { RagbotApiError, RagbotClient } from "../dist/index.js";

function jsonResponse(status, payload, headers = {}) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

test("typed API error preserves request id and retryability", async () => {
  const client = new RagbotClient("http://ragbot.test", {
    fetchImpl: async () =>
      jsonResponse(409, {
        error: {
          code: "conflict",
          message: "plan unavailable",
          request_id: "req-1",
          retryable: false,
          details: { plan: "qdrant_dense_sparse" },
        },
      }),
  });
  await assert.rejects(
    () => client.search({ query: "x", tenant_id: "t", user_id: "u" }),
    (error) => {
      assert.ok(error instanceof RagbotApiError);
      assert.equal(error.statusCode, 409);
      assert.equal(error.code, "conflict");
      assert.equal(error.requestId, "req-1");
      return true;
    },
  );
});

test("source iterator follows keyset cursors", async () => {
  const urls = [];
  const client = new RagbotClient("http://ragbot.test", {
    fetchImpl: async (input) => {
      const url = new URL(String(input));
      urls.push(url.href);
      if (!url.searchParams.has("cursor")) {
        return jsonResponse(200, {
          total: 2,
          next_cursor: "cursor-2",
          sources: [{ source_id: "s1", tenant_id: "t", source_type: "pdf", name: "one", config: {} }],
        });
      }
      assert.equal(url.searchParams.get("cursor"), "cursor-2");
      return jsonResponse(200, {
        total: 2,
        next_cursor: null,
        sources: [{ source_id: "s2", tenant_id: "t", source_type: "pdf", name: "two", config: {} }],
      });
    },
  });

  const ids = [];
  for await (const source of client.iterateSources({ tenantId: "t", pageSize: 1 })) {
    ids.push(source.source_id);
  }
  assert.deepEqual(ids, ["s1", "s2"]);
  assert.equal(urls.length, 2);
});

test("AbortSignal cancels an in-flight request", async () => {
  const controller = new AbortController();
  const client = new RagbotClient("http://ragbot.test", {
    timeoutMs: 10_000,
    fetchImpl: async (_input, init) =>
      new Promise((_resolve, reject) => {
        init.signal.addEventListener("abort", () => reject(init.signal.reason), { once: true });
      }),
  });
  const pending = client.search(
    { query: "x", tenant_id: "t", user_id: "u" },
    { signal: controller.signal },
  );
  controller.abort(new Error("cancelled by caller"));
  await assert.rejects(pending, /cancelled by caller/);
});
