# Async Retrieval Plane

Issue #57 establishes retrieval as one typed asynchronous pipeline instead of a synchronous method with per-request thread-pool fan-out.

## Public plan contract

The stable plan identifiers are:

| Plan | Candidate path | Status |
| --- | --- | --- |
| `dense` | embedding -> active vector index | control/ablation |
| `lexical` | PostgreSQL FTS/CJK | control/ablation |
| `hybrid_rrf` | dense + lexical concurrently -> adaptive RRF -> optional reranker | default control |
| `qdrant_dense_sparse` | named dense + sparse vectors in Qdrant | experimental, capability-gated |

The older `mode=vector|lexical|hybrid` API remains a compatibility alias for `dense|lexical|hybrid_rrf`.

`qdrant_dense_sparse` is intentionally fail-fast until the active IndexVersion and vector backend expose a real named dense+sparse schema. Ragbot must not silently synthesize a sparse representation or replace the control plan without an evaluation gate.

## Async execution

`Retriever` owns one bounded, long-lived executor for synchronous PostgreSQL/Qdrant/reranker adapters. A request no longer creates a `ThreadPoolExecutor`.

The normal hybrid path is:

```text
RetrievalRequest
      |
      +---- dense candidate task ---- embedding -> vector search -> visibility fence
      |
      +---- lexical candidate task -- PostgreSQL FTS/CJK
      |
      v
 adaptive RRF
      |
 optional reranker
      |
 optional diversity stage
      |
RetrievalResponse + RetrievalTrace
```

API and Agent callers use `await Retriever.query()` / `await Retriever.aretrieve()`. The historical synchronous `retrieve()` facade exists for CLI/tests and refuses to block an already-running event loop.

## Deadline and cancellation semantics

`deadline_ms` is one end-to-end retrieval budget. Every candidate/rerank stage consumes the same monotonic budget; it is not reset per backend call.

A deadline failure raises `RetrievalDeadlineExceeded` and records:

- plan;
- deadline;
- stage timings collected so far;
- candidate counts collected so far;
- `timed_out=true`;
- `error_stage`.

For HTTP `/search`, this becomes a `504` with the non-secret retrieval trace. If a parallel hybrid branch times out or the request is cancelled, outstanding asyncio tasks are cancelled. Blocking library calls already executing in a worker thread cannot be force-killed by Python; the backend clients therefore still need their own transport/database timeouts.

## Trace contract

Each returned chunk records `_retrieval` metadata with the evidence required to diagnose ranking changes:

- dense/vector rank and raw score when present;
- lexical rank and raw score when present;
- fusion score and fusion policy;
- rerank score when enabled;
- final score and final rank;
- embedding model;
- request-level stage timings and candidate counts.

`vector` remains a compatibility alias of the new `dense` trace source for existing workbench/evaluation consumers.

## Quality promotion rule

The current adaptive `hybrid_rrf` plan remains the control. Sparse/multi-vector work is not promoted because it is technically newer or returns more candidates.

Before promoting another plan, run the repository retrieval-quality/evaluation gates on the same corpus and compare at minimum:

- Recall@K / Hit@K;
- MRR/NDCG where labels support it;
- exact, paraphrase and cross-lingual slices;
- p50/p95 end-to-end retrieval latency;
- candidate counts and reranker cost;
- citation/ACL correctness.

A candidate plan should be rejected if quality regresses outside the agreed tolerance or if the latency/cost increase is not justified by quality gain.

## Phase-2 boundary

Phase 1 fixes the async/composable control plane and the explicit experimental port. Remaining Issue #57 work includes:

1. define the sparse encoder contract and immutable IndexVersion schema for named dense+sparse vectors;
2. implement Qdrant native hybrid/prefetch without changing active-index semantics;
3. evaluate sparse + dense and multi-vector representations against `hybrid_rrf`;
4. add semantic MMR/diversity only if corpus evidence shows value over the lightweight optional duplicate suppression stage;
5. persist the most useful stage latency/fusion evidence into the durable evaluation/trace plane tracked by #61.
