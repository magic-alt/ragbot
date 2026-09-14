# Durable quality / observability control plane

Issue #61 turns Ragbot's runtime traces and benchmark outputs into durable,
comparable evidence. The goal is not to store more user content. It is to make
model/index/retrieval changes reproducible enough that promotion decisions can
be based on measured quality, latency and cost.

## Runtime lineage

Every durable RAG run is keyed by the same `request_id` returned to the caller.
A run can include:

- tenant/user identity from the trusted request path;
- route and terminal status;
- retrieval plan and deterministic retrieval contract ID;
- embedding contract ID;
- active IndexVersion ID;
- reranker contract ID;
- model provider/model contract summaries;
- per-stage latency;
- retrieved chunk/document IDs, rank and score;
- citation identifiers;
- provider token usage and estimated cost;
- sampled structured trace details.

The control-plane record deliberately excludes evidence/chunk text.

## Privacy defaults

`RAGBOT_TRACE_STORE_CONTENT=false` is the default. With that default the raw
query is not stored; a SHA-256 query hash is retained for deduplication and
correlation. Detailed trace payloads are sampled separately from the mandatory
minimal lineage row.

| Variable | Default | Meaning |
| --- | --- | --- |
| `RAGBOT_TRACE_STORE_CONTENT` | `false` | opt in to raw query storage |
| `RAGBOT_TRACE_DETAIL_SAMPLE_RATE` | `1.0` | fraction of requests retaining detailed structured trace |
| `RAGBOT_TRACE_RETENTION_DAYS` | `30` | expiry assigned to durable request traces; `0` disables automatic expiry assignment |

Use `POST /admin/quality/retention/purge` from trusted operations to remove
expired request traces. Production deployments should choose a retention period
that matches privacy, incident-response and compliance requirements.

## Tables

Migration `015_quality_observability.sql` adds:

- `rag_runs`: durable online request lineage;
- `evaluation_runs`: immutable offline/staging evaluation evidence;
- `promotion_decisions`: baseline/candidate decision evidence.

The pre-existing `feedback` table is extended with `citation_id`, `rating` and
metadata instead of creating a second feedback system.

PostgreSQL remains authoritative. In-memory implementations exist only for
local development/tests.

## Search request lifecycle

The search API allocates its request ID **before** retrieval begins:

```text
request_id
   |
   +--> typed RetrievalRequest
   |       |
   |       +--> dense / lexical / fusion / rerank
   |       |
   |       +--> RetrievalTrace
   |
   +--> completed / failed / deadline_exceeded / cancelled
   |
   +--> rag_runs
```

This means a timeout can be correlated to the exact retrieval/embedding/index
contracts instead of disappearing before a request ID exists.

Durable observability is fail-open for serving: a PostgreSQL trace-write failure
is logged but does not turn an otherwise successful search into a user-facing
outage.

## Agent model usage

`ModelRouter` records model usage with the Agent request ID. `CostTracker` can
therefore return only records for one request. A durable Agent run can identify
which provider/model handled each task and aggregate input/output/cached/
reasoning tokens and estimated cost.

The request ID is carried in a ContextVar together with the existing task scope;
provider configuration is not mutated.

## EvaluationRun

`EvaluationRun` is immutable. It contains:

- dataset name/version;
- code revision;
- candidate/baseline refs;
- runtime contract IDs;
- benchmark configuration;
- quality metrics;
- latency metrics;
- cost metrics;
- artifact metadata.

`evaluation_contract_id` is content-derived from the dataset/config/runtime
contract identity; `evaluation_run_id` uniquely identifies one execution.
Repeated runs of one contract therefore remain independently auditable.

## Existing benchmark integration

The existing `benchmarks.rag_native_compare` remains the benchmark owner. The
quality layer provides `evaluation_from_native_report()` to normalize its JSON
report into an `EvaluationRun`.

Canonical promotion metrics are:

```text
metrics.recall
metrics.mrr
metrics.ndcg
latency.p95_latency_ms
cost.cost_usd
```

The adapter also retains category/slice metrics and corpus/version metadata as
evaluation artifacts, so exact/paraphrase/cross-lingual analysis can remain
visible even when the top-level promotion policy is concise.

## Promotion gate

A promotion compares one immutable baseline EvaluationRun with one immutable
candidate EvaluationRun. The default policy rejects:

- any Recall regression;
- any MRR regression;
- any nDCG regression;
- p95 latency increase over 15%;
- cost increase over 25%.

Thresholds are explicit inputs and are stored with the decision. Missing
required evidence is a rejection, not an implicit pass.

This makes the intended #57 Phase-2 flow:

```text
hybrid_rrf baseline
        |
        +--> EvaluationRun

qdrant_dense_sparse candidate
        |
        +--> EvaluationRun
                 |
                 v
          PromotionPolicy
                 |
         accept / reject
                 |
        PromotionDecision
```

A retrieval plan should not become the production default merely because a new
backend supports it.

## Admin APIs

Trusted admin surfaces:

- `GET /admin/quality/runs/{request_id}`
- `GET /admin/quality/runs`
- `POST /admin/quality/evaluations`
- `GET /admin/quality/evaluations/{evaluation_run_id}`
- `GET /admin/quality/evaluations`
- `POST /admin/quality/promotions/evaluate`
- `POST /admin/quality/retention/purge`

Existing `/admin/feedback` can now attach durable feedback to a request (and
optionally a citation ID) instead of requiring that request to remain in the
same API replica's process-local history.

## What remains under #61

Phase 1 establishes the evidence boundary. Follow-up work should include:

- parser/chunker/prompt contract IDs in all relevant online runs;
- first-token latency where provider transports expose it reliably;
- dashboards and alerting for p95/p99 latency, failure rate, retrieval quality
  drift and model cost;
- scheduled Golden Dataset evaluation ingestion;
- automated promotion workflows that consume `PromotionDecision` without
  bypassing IndexVersion activation/rollback controls;
- richer feedback analytics by query slice, citation and contract version.
