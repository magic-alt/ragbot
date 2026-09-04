# Ingestion Count Semantics

Ragbot distinguishes the resulting knowledge snapshot from work performed by one ingestion Job.

| Field | Meaning |
| --- | --- |
| `job.doc_count` | Documents present in the completed Source snapshot |
| `job.chunk_count` | Chunks written/embedded by this Job (legacy API field) |
| `job.stats.chunks_total` | Chunks present in the completed Source snapshot |
| `job.stats.chunks_ingested` | Chunks written/embedded by this Job |
| `job.stats.chunks_reused` | Unchanged chunks reused without re-embedding |
| `job.stats.chunks_removed` | Stale PostgreSQL chunks removed during reconciliation |

Therefore a successful incremental re-ingestion can legitimately have:

```text
docs=1, chunks=287, written=0, reused=287
```

This means the Source has 287 searchable chunks and all of them were reused. It does **not** mean the knowledge base contains zero chunks.

The human-facing CLI reports `chunks=<snapshot total>, written=<new writes>, reused=<unchanged reuse>`. JSON/API responses keep the existing Job field semantics for compatibility.
