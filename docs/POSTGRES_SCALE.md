# PostgreSQL Scale and Operations

Ragbot treats PostgreSQL as the authoritative control-plane and knowledge-manifest store. Qdrant is a derived vector index. This document defines the PostgreSQL performance contract introduced by Issue #56.

## Hot-path rules

1. User-facing list/catalog APIs are bounded and use keyset pagination. `limit` is capped at 500; opaque cursors encode only the ordered timestamp/id tuple.
2. Tenant/source/status ownership filtering executes in SQL. Do not load a tenant or repository-wide catalog and filter it in Python on request paths.
3. Worker queue claims perform only the `FOR UPDATE SKIP LOCKED` claim transaction. Expired-lease/DLQ repair belongs to the periodic `RAGBOT_RECONCILE_SECONDS` pass and explicit admin reconciliation.
4. Small writes retain the compatibility `executemany` path. Chunk/generation batches at or above `RAGBOT_PG_COPY_MIN_ROWS` use PostgreSQL `COPY` plus staging/upsert semantics.
5. Query-shape indexes are migration-owned. Any changed hot query must update the EXPLAIN plan gate in `benchmarks/postgres_hotpaths.py`.

## Runtime policy

| Setting | Default | Purpose |
| --- | ---: | --- |
| `RAGBOT_PG_POOL_MIN` | 2 | Minimum open connections per process |
| `RAGBOT_PG_POOL_MAX` | 10 | Maximum pooled connections per API/worker process |
| `RAGBOT_PG_CONNECT_TIMEOUT_SECONDS` | 5 | libpq connection timeout |
| `RAGBOT_PG_STATEMENT_TIMEOUT_MS` | 30000 | Per-session statement timeout |
| `RAGBOT_PG_LOCK_TIMEOUT_MS` | 5000 | Fail boundedly when waiting for locks |
| `RAGBOT_PG_IDLE_TRANSACTION_TIMEOUT_MS` | 30000 | Kill leaked idle transactions |
| `RAGBOT_PG_COPY_MIN_ROWS` | 256 | Minimum batch size for COPY/staging |

Pool sizing is **per process**. When API and worker replicas scale horizontally, keep the aggregate maximum below the database connection budget or place PgBouncer/RDS Proxy-equivalent pooling in front of PostgreSQL. Do not increase `POOL_MAX` to hide saturation; inspect `/admin/database/metrics` first.

## Pagination contract

The bounded catalog surfaces are:

- `GET /sources`
- `GET /ingest/jobs`
- `GET /catalog/sources`
- `GET /catalog/jobs`
- `GET /catalog/documents`
- `GET /catalog/generations`

Responses preserve their resource arrays and add `total` plus `next_cursor`. Cursors are opaque to clients. Callers should pass `next_cursor` back unchanged.

Keysets use deterministic descending order:

- Sources: `(created_at, source_id)`
- Jobs: `(created_at, job_id)`
- Documents: `(ingested_at, doc_id)`
- Knowledge generations: `(created_at, generation_id)`

Offset pagination is intentionally not the production contract: deep offsets force PostgreSQL to visit/discard an ever-growing prefix and are unstable under concurrent inserts.

## Queue and reconciliation

Normal worker loop:

```text
periodic reconcile (default every 30s)
        ↓
claim one ready row via SKIP LOCKED
        ↓
execute / heartbeat / terminal transition
```

`claim_next_job()` must not perform queue-wide reconciliation. This makes empty-queue polling and N-worker contention proportional to claim traffic instead of multiplying repair writes by worker count.

IndexVersion cutover still wraps claims with the shared publication advisory lock from Issue #53. The lock wraps the optimized claim transaction; it does not reintroduce reconciliation.

## COPY semantics

`add_chunks()` uses a transaction-local temporary table followed by `INSERT ... ON CONFLICT DO UPDATE`. This preserves the historical chunk upsert behavior while avoiding per-row round trips.

Generation staging uses direct `COPY` into `staged_documents` and `staged_chunks` after deleting the previous candidate rows for that generation. Publication/activation semantics are unchanged.

The fallback below `RAGBOT_PG_COPY_MIN_ROWS` is deliberate: for small batches, creating a staging table can cost more than `executemany`.

## Performance gate

PRs touching PostgreSQL/storage/control-plane paths run `.github/workflows/postgres-performance.yml`.

Default PR gate:

```bash
python -m benchmarks.postgres_hotpaths \
  --chunks 3000 \
  --plan-jobs 3000 \
  --claim-jobs 64 \
  --workers 1,8
```

The gate records:

- legacy `executemany` rows/s;
- COPY rows/s and speedup;
- `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` node types and timings;
- claim throughput and p50/p95/max at each worker count;
- connection-pool and Ragbot hot-path counters.

The plan assertions disable sequential scans only for the EXPLAIN session. If a query has no usable index, PostgreSQL still emits a `Seq Scan` with prohibitive cost and the gate fails. This removes small-CI-dataset planner variance without hiding a missing index.

### Full promotion matrix

Before changing bulk-write, queue or pagination architecture, run the manual workflow at:

| Scale | Chunks per legacy/COPY run | Worker matrix | Purpose |
| --- | ---: | --- | --- |
| CI | 3,000 | 1,8 | Fast regression signal |
| Medium | 10,000 / 100,000 | 1,8,32 | Throughput/claim-contention trend |
| Promotion | 1,000,000 | 1,8,32 | Bulk-ingest production evidence |

Manual workflow inputs for the full gate:

```text
chunks=1000000
plan_jobs=100000
claim_jobs=1024
workers=1,8,32
min_copy_speedup=<baseline promotion threshold>
```

Do not copy a speedup threshold from another PostgreSQL instance. Establish the baseline on the intended production class, then version the threshold with the benchmark evidence.

## Retention and partitioning policy

Ragbot does not partition tables merely because partitioning exists. Partitioning adds operational complexity and only becomes mandatory after measured table/index maintenance pressure.

Recommended retention classes:

- `ingestion_jobs`: keep active/nonterminal rows indefinitely; retain terminal rows long enough for audit/debugging (default recommendation 30–90 days), then archive/delete according to product policy.
- `publication_outbox`: completed events may be deleted after the recovery/audit window; dead-letter/retryable events stay until resolved.
- `knowledge_generations`: keep active + rollback-relevant generation metadata; retire historical metadata according to audit requirements. Do not delete a generation still referenced by active chunks/documents.
- `vector_index_versions`: retention is governed by IndexVersion rollback policy from `docs/INDEX_LIFECYCLE.md`.
- `documents/chunks`: product knowledge, not operational logs; lifecycle deletion is Source/Generation-owned rather than time-based retention.

Partitioning trigger review should happen when one of these is observed in production evidence:

- terminal `ingestion_jobs` history reaches tens of millions of rows and vacuum/index maintenance becomes material;
- retention deletes cause sustained table/index bloat;
- time-bounded operational queries dominate and partition pruning materially lowers buffers/latency;
- backup/restore or maintenance windows exceed SLOs because of operational-history tables.

If triggered, partition **operational history first** (for example terminal jobs by `created_at` month). Do not partition `chunks` by time unless retrieval/storage measurements prove a benefit; tenant/source/generation access patterns are more important there.

## Observability

`GET /admin/database/metrics` returns non-secret:

- psycopg pool statistics;
- claim attempts/hits/empty polls;
- claim mean/max latency accumulated in-process;
- COPY batch/row counters;
- effective pool/timeout/COPY threshold settings.

Use database-native monitoring for authoritative server latency, lock waits, vacuum/bloat and connection saturation. Ragbot runtime metrics diagnose the application side of the boundary; they do not replace `pg_stat_activity`, `pg_stat_statements`, managed-database metrics or PostgreSQL logs.
