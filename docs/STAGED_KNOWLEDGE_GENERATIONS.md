# Staged Knowledge Generations

Ragbot does not attempt to create a distributed transaction between PostgreSQL and Qdrant. PostgreSQL is the authoritative visibility boundary; Qdrant is a prepared semantic index whose physical points are admitted only when the active PostgreSQL manifest references them.

## Publication protocol

```text
Source / durable Job / lifecycle fence
        ↓
connector → Parser Port → Chunker Port
        ↓
complete candidate snapshot
        ↓
knowledge_generations(status=staging)
        ↓
staged_documents + staged_chunks
        ↓
changed/new embeddings
        ↓
generation-specific Qdrant points
        ↓
knowledge_generations(status=prepared)
        ↓
PostgreSQL activation transaction
  ├─ lock Source and validate lifecycle generation
  ├─ replace active documents/chunks
  ├─ swap source_active_generations
  ├─ mark previous generation retired
  └─ insert publication_outbox cleanup events
        ↓
COMMIT = new knowledge becomes visible
        ↓
existing durable worker deletes retired Qdrant points
```

There is no interval in which staged PostgreSQL rows are exposed to lexical retrieval. Qdrant writes may happen before activation, but those points are rejected by retrieval until the active PostgreSQL chunk manifest references their physical point IDs.

## Logical chunks and physical vector points

A logical Ragbot chunk has a stable `chunk_id` when its content/transformation/index contract is reusable. The physical Qdrant point is recorded in the active PostgreSQL `chunks.qdrant_point_id` field.

For changed/new chunks Ragbot creates a deterministic generation-specific physical point ID:

```text
point_id = UUIDv5(generation_id + chunk_id)
```

Unchanged chunks reuse the prior physical point instead of paying embedding cost again. Therefore a current logical generation can reference physical points originally created by several prior generations.

This is why vector visibility is defined by the active manifest rather than by a simple `payload.generation_id == active_generation_id` predicate.

## Query visibility

Vector retrieval is two-stage:

1. Qdrant returns semantic candidates.
2. Ragbot batches their logical chunk IDs through PostgreSQL and resolves the current active `chunk_id -> qdrant_point_id` mapping.

A Qdrant hit participates in fusion/reranking only when its physical point ID exactly matches the active mapping.

Consequences:

- a staged point is query-invisible before activation;
- a retired point is query-invisible immediately after activation even if cleanup has not run yet;
- a reused old physical point remains valid when the active manifest still references it;
- PostgreSQL FTS sees only the active manifest because staged rows live in separate tables.

The Retriever intentionally overfetches vector candidates before active-manifest filtering. The publication outbox should normally remove stale points within seconds, so stale generations do not accumulate in the candidate pool.

## Source lifecycle fencing

Every ingestion Job carries the Source lifecycle token captured when it was submitted. The candidate `knowledge_generations` row persists that token.

Activation locks the Source row in the same PostgreSQL transaction that swaps the active generation and checks:

- Source still exists;
- tenant identity is unchanged;
- Source is not deleted;
- current Source lifecycle token equals the durable Job/generation token.

A Source edit/delete/pause that changes `updated_at` therefore prevents an older prepared generation from being activated.

## Failure matrix

| Failure | Active knowledge | Recovery |
|---|---|---|
| connector/parser/chunker fails | previous generation remains active | normal Job retry/DLQ |
| embedding fails before any vector write | previous generation remains active | generation marked failed |
| Qdrant fails after writing some staged points | previous generation remains active | failed generation enqueues point cleanup |
| PostgreSQL staging fails | previous generation remains active | retry Job; staged transaction rolls back |
| activation transaction fails | previous pointer/active rows remain intact | retry/fail generation |
| worker dies during retired-vector delete | new generation remains active | outbox lease expires and is reclaimed |
| Qdrant delete is temporarily unavailable | new generation remains active | bounded outbox backoff/retry |

Deletion is idempotent, so an outbox event may safely retry a point that was already removed.

## Durable outbox

`publication_outbox` currently owns `delete_qdrant_points` events. Point IDs are batched (up to 500 per event) so large replacement generations do not create unbounded row payloads.

Workers claim events with leases using the same operational model as ingestion Jobs:

```text
pending → running → completed
              └→ pending (retry)
              └→ failed (attempts exhausted)
```

Expired running events are reconciled back to pending unless their attempt budget is exhausted.

Configuration:

```dotenv
RAGBOT_PUBLICATION_OUTBOX_SCAN_SECONDS=5
RAGBOT_PUBLICATION_OUTBOX_MAX_ATTEMPTS=10
```

The existing ingestion worker owns outbox reconciliation; Ragbot does not introduce a second queue or a second worker service.

## Schema

Migration `010_staged_knowledge_generations.sql` introduces:

- `knowledge_generations` — lifecycle/audit record for every candidate generation;
- `source_active_generations` — authoritative Source → active generation pointer;
- `staged_documents` — candidate document manifest;
- `staged_chunks` — candidate lexical/vector manifest;
- `publication_outbox` — durable post-commit external cleanup;
- `source_id` / `generation_id` ownership columns on active documents/chunks.

Existing active knowledge is best-effort bootstrapped as `legacy:<source_id>` using the Source's configured/default document identity. The first normal post-upgrade ingestion fully establishes explicit ownership even for unusual legacy custom document IDs.

## Compatibility boundary

The generation protocol is an additive storage capability. Built-in InMemory and PostgreSQL repositories support it. A custom repository that does not implement staged publication stays on the legacy direct-publication path and is reported in ingestion stats as:

```json
{"publication_mode": "legacy-direct"}
```

Built-in production PostgreSQL reports:

```json
{
  "publication_mode": "staged-generation",
  "knowledge_generation_id": "gen-...",
  "previous_knowledge_generation_id": "gen-...",
  "vector_cleanup_enqueued": 42
}
```

Custom repositories should implement the full generation/outbox capability before claiming equivalent atomic cutover semantics.

## Operational checks

For an ingestion Job, inspect:

- `knowledge_generation_id`;
- `previous_knowledge_generation_id`;
- `chunks_ingested` versus `chunks_reused`;
- `vector_cleanup_enqueued`;
- failed/running `publication_outbox` rows.

A growing pending/failed outbox indicates Qdrant cleanup or worker-health problems. It does not make retired vectors visible, but excessive stale points can increase vector overfetch work and should be repaired promptly.

## Backup and restore

The migration tables are normal PostgreSQL durable state and are included by Ragbot's PostgreSQL backup. Qdrant remains separately snapshotted. Because PostgreSQL is authoritative for visibility, restoring both stores from inconsistent points in time can still require reconciliation/reindexing; coordinated snapshots remain preferred for strict disaster-recovery objectives.
