# Async Retrieval Plane

Issue #57 establishes retrieval as one typed asynchronous pipeline instead of a synchronous method with per-request thread-pool fan-out.

## Public plan contract

The stable plan identifiers are:

| Plan | Candidate path | Status |
| --- | --- | --- |
| `dense` | embedding -> active vector index | control/ablation |
| `lexical` | PostgreSQL FTS/CJK | control/ablation |
| `hybrid_rrf` | dense + lexical concurrently -> adaptive RRF -> optional reranker | default control |
| `qdrant_dense_sparse` | named dense + sparse -> Qdrant Prefetch/RRF -> optional reranker | Phase-2 candidate |

The older `mode=vector|lexical|hybrid` API remains a compatibility alias for `dense|lexical|hybrid_rrf`.

`hybrid_rrf` remains the production control. `qdrant_dense_sparse` is capability- and contract-gated: it requires an IndexVersion with an immutable sparse representation contract and never silently falls back to the control plan.

## Async execution

`Retriever` owns one bounded, long-lived executor for synchronous PostgreSQL/Qdrant/reranker adapters. A request no longer creates a `ThreadPoolExecutor`.

The normal control path is:

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

The Phase-2 candidate is:

```text
same dense embedding contract
        +
SparseEmbeddingSpec
        |
        v
Qdrant named dense+sparse IndexVersion
        |
        v
Prefetch(dense) + Prefetch(sparse)
        |
        v
Qdrant native RRF
        |
optional existing reranker
```

API and Agent callers use `await Retriever.query()` / `await Retriever.aretrieve()`. The historical synchronous `retrieve()` facade exists for CLI/tests and refuses to block an already-running event loop.

## Candidate IndexVersion before activation

`RetrievalRequest.index_version_id` is an internal evaluation selector and is not exposed by `/v1/search`. It allows a validating dense+sparse physical collection to be queried before the active Qdrant alias changes.

The candidate plan validates before I/O that:

- the selected IndexVersion is `validating`, `ready` or `active`;
- its dense embedding contract equals the runtime dense contract;
- a sparse contract exists in `vector_schema`;
- the configured sparse encoder has the exact same contract ID;
- the Qdrant backend supports the native hybrid Query API.

This makes Golden Dataset comparison possible without routing production search traffic to the candidate.

## Deadline and cancellation semantics

`deadline_ms` is one end-to-end retrieval budget. Every candidate/rerank stage consumes the same monotonic budget; it is not reset per backend call.

A deadline failure raises `RetrievalDeadlineExceeded` and records plan, deadline, stage timings/candidate counts collected so far, `timed_out=true`, and `error_stage`.

For HTTP `/search`, this becomes a `504` with the non-secret retrieval trace. If a parallel hybrid branch times out or the request is cancelled, outstanding asyncio tasks are cancelled. Blocking library calls already executing in a worker thread cannot be force-killed by Python; backend clients therefore still need their own transport/database timeouts.

## Trace contract

Each returned chunk records `_retrieval` metadata with the evidence required to diagnose ranking changes:

- dense/vector rank and raw score when present;
- lexical or native dense+sparse rank and raw score when present;
- fusion score and fusion policy;
- rerank score when enabled;
- final score and final rank;
- embedding model;
- request-level stage timings and candidate counts.

Dense+sparse traces additionally record `index_version_id`, dense and sparse representation contract IDs, dense embedding latency, sparse embedding latency, and native Qdrant search latency. These fields feed the durable #61 quality plane.

`vector` remains a compatibility alias of the new `dense` trace source for existing workbench/evaluation consumers.

## Quality promotion rule

The adaptive `hybrid_rrf` plan remains the control. `benchmarks.retrieval_plan_promotion` evaluates both plans on the same Golden Dataset and persists:

```text
hybrid_rrf -> EvaluationRun(baseline)
qdrant_dense_sparse -> EvaluationRun(candidate)
                         |
                         v
                  PromotionDecision
```

The gate compares Recall@10, MRR@10, nDCG@10, p95 latency and cost. Missing required evidence rejects promotion. The runner does not activate an IndexVersion automatically; an accepted candidate may optionally be marked `ready`, after which the existing explicit activate/rollback operations remain the deployment boundary.

See [`DENSE_SPARSE_PROMOTION.md`](DENSE_SPARSE_PROMOTION.md) for configuration, build, evaluation, promotion and rollback details.

## Phase-2 scope boundary

This phase intentionally implements only the dense+sparse candidate. ColBERT/late interaction, multi-query expansion and semantic MMR remain future experiments. They should be introduced one at a time only after this evidence pipeline proves that a new representation can be compared and promoted without changing multiple ranking variables at once.
