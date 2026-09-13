# Vector Index Lifecycle

Ragbot treats an embedding/vector-schema change as a **data migration**, not an environment-variable edit.

The production contract is:

```text
EmbeddingSpec / embedding_contract_id
        │
        ▼
IndexVersion
        │
        ├─ physical Qdrant collection A (active)
        ├─ physical Qdrant collection B (candidate)
        │
        ▼
stable Qdrant alias: <QDRANT_COLLECTION>_active
```

The lifecycle is:

```text
building -> validating -> ready -> active -> retired -> deleted
                  \-> failed -----------^
```

A candidate is built and validated independently. Promotion changes the stable Qdrant alias atomically; the old collection remains available for the configured retention window.

## 1. First deployment: adopt the existing collection in place

Existing installations do **not** need to copy vectors when this feature is first deployed.

Given:

```dotenv
QDRANT_COLLECTION=rag_chunks
QDRANT_DIM=1536
```

Ragbot derives the logical alias:

```text
rag_chunks_active
```

or uses an explicit override:

```dotenv
QDRANT_INDEX_ALIAS=production_knowledge
```

On first startup after migration `012_vector_index_versions.sql`:

1. the existing physical `rag_chunks` collection remains unchanged;
2. Ragbot creates the alias `rag_chunks_active -> rag_chunks` if it does not already exist;
3. PostgreSQL records that physical collection as the bootstrap active `IndexVersion`;
4. subsequent search/upsert calls use the stable alias.

API and worker may start concurrently. Alias creation and legacy IndexVersion registration are therefore race-tolerant and converge on the same physical collection instead of depending on startup order.

No initial re-embedding is required.

## 2. Register a standby embedding contract

Do not replace the currently serving `EMBEDDING_MODEL` before a candidate index exists. Keep the current model as the default contract and load candidate contracts beside it.

`RAGBOT_EMBEDDING_PROFILES_JSON` is intentionally non-secret. A profile may reference an API-key environment variable **by name**, but must not contain the key itself.

Example for a second local Ollama embedding model:

```dotenv
RAGBOT_EMBEDDING_PROFILES_JSON={"qwen4b":{"provider_id":"openai-compatible","model":"qwen3-embedding:4b","dimension":2560,"base_url":"http://host.docker.internal:11434","api_key_env":""}}
```

The same profile set must be present on every API and ingestion-worker replica before activation. Keep both the old and new contracts loaded throughout the rollback-retention window.

List the resolved contract IDs:

```bash
python scripts/rag_index.py contracts
```

or use the global-admin endpoint:

```text
GET /admin/indexes/contracts
```

## 3. Create a candidate physical collection

```bash
python scripts/rag_index.py create \
  --contract emb-<candidate-contract-id>
```

This creates:

- one `vector_index_versions` row with `status=building`;
- one independent physical Qdrant collection with the candidate dimension/distance;
- no query-visible change.

The collection name is generated from alias + contract + version ID unless `--collection` is supplied.

## 4. Build outside the API request path

Run the build as a separate operator/job process:

```bash
python scripts/rag_index.py build idx-... --batch-size 100
```

The build:

- scans the active PostgreSQL chunk manifest;
- embeds each active chunk using the candidate contract;
- preserves each chunk's current `qdrant_point_id`, so Source-generation visibility rules remain valid after cutover;
- writes only to the candidate physical collection;
- persists `vectors_written`, vectors/s, elapsed time and estimated remaining seconds after every batch;
- records parser/chunker contracts observed in the active corpus;
- records a catalog fingerprint over active `chunk_id + qdrant_point_id + checksum`.

If the catalog changes while the build runs, the candidate fails instead of publishing a mixed snapshot.

A failed build can be retried. The candidate physical collection is recreated first so partial/stale points from the previous attempt cannot leak into the retry.

## 5. Shadow validation

A simple machine-readable case file can compare the active and candidate dense indexes:

```json
[
  {
    "query": "How does source generation fencing work?",
    "expected_chunk_ids": ["chunk-..."]
  },
  {
    "query": "关节模组 EtherCAT 相关说明"
  }
]
```

Run:

```bash
python scripts/rag_index.py shadow idx-... \
  --cases eval/my_index_cases.json \
  --top-k 10 \
  --output reports/index-shadow.json
```

Baseline and candidate queries use their **own** embedding contracts independently. Comparing a 1536D active index with a 2560D candidate is therefore valid.

The built-in result includes:

- per-query top-K chunk IDs;
- overlap@K;
- baseline/candidate latency;
- p95 latency;
- Hit@K and MRR when exact `expected_chunk_ids` are supplied.

For production promotion, use the project's Golden Dataset/evaluation artifacts rather than relying only on this compact shadow helper.

## 6. Mark ready only with evidence

```bash
python scripts/rag_index.py ready idx-... \
  --evidence reports/index-shadow.json \
  --approve
```

`ready` requires both:

- non-empty validation evidence;
- explicit approval.

Ragbot rechecks the catalog fingerprint before moving to `ready`.

## 7. Atomic activation

```bash
python scripts/rag_index.py activate idx-... \
  --retention-seconds 604800
```

Activation uses a short PostgreSQL publication barrier. Worker job claims take the **shared** form of one advisory lock, so workers remain concurrent during normal ingestion. `activate`/`rollback` take the **exclusive** form only for the cutover attempt:

```text
1. take exclusive publication barrier -> new durable worker claims wait
2. require running ingestion job count == 0
3. validate ready + embedding contract + catalog fingerprint
4. atomically switch Qdrant alias -> candidate physical collection
5. commit PostgreSQL active/retired IndexVersion state
6. update active chunk metadata to the new embedding contract/model/dimension
7. release barrier -> pending jobs claim and naturally use the new index
```

If a job is already running, activation fails fast with an explicit quiescent-boundary error; it does not interrupt that ingestion. Retry after current jobs reach terminal state. Long parser/chunker/embedding work is never executed under the exclusive cutover lock.

The chunk metadata update is part of the same PostgreSQL IndexVersion activation transaction. It preserves chunk IDs and `qdrant_point_id` but changes the effective embedding identity, so the next ordinary Source sync can reuse unchanged candidate vectors rather than re-embedding the entire corpus simply because the old chunk metadata still named the previous model.

Qdrant alias state is the **query-visible authority**. `ActiveIndexEmbedder` resolves its embedding contract from the alias-visible physical collection. Therefore the next request after the alias switch uses:

```text
new query embedding contract + new physical vector index
```

as one serving state.

If PostgreSQL commit fails, Ragbot first attempts a compensating alias switch. If a process crashes after the alias switch and before the PostgreSQL commit, query traffic still follows the visible alias consistently and `reconcile` repairs PostgreSQL, including the active chunk embedding identity.

```bash
python scripts/rag_index.py reconcile
```

The publication barrier opens an independent PostgreSQL session using the original configured DSN. Production fails closed if that reconnect credential is unavailable; it never silently degrades a multi-replica cutover to a process-local lock.

## 8. Rollback

```bash
python scripts/rag_index.py rollback idx-<previous> \
  --retention-seconds 604800
```

Rollback does not re-embed: it atomically points the alias at the retained physical collection, restores that version's embedding contract and updates the PostgreSQL active chunk embedding identity in the same control-plane transaction.

However, retired collections are immutable snapshots. Ragbot deliberately rejects rollback when the PostgreSQL knowledge catalog has changed since that IndexVersion was built. Serving a known-stale knowledge snapshot is considered worse than refusing rollback.

If continuous rollback across ongoing ingestion is required later, it should be implemented as explicit multi-index write/mirroring rather than weakening the catalog fence.

## 9. Retention and deletion

Retired versions receive `delete_after` during activation/rollback. Prune only after the rollback window:

```bash
python scripts/rag_index.py prune
```

The alias-visible active physical collection is never deleted by retention cleanup.

## 10. Backup / restore

`scripts/backup_ragbot.sh` resolves the stable alias and snapshots the **active physical collection**, then stores both physical collection and alias in `manifest.json`.

Qdrant collection snapshots do not contain alias metadata. `scripts/restore_ragbot.sh` therefore:

1. verifies artifact checksums;
2. restores PostgreSQL;
3. restores the physical Qdrant collection recorded in the manifest;
4. explicitly creates/replaces the logical alias in a Qdrant alias update;
5. polls `/aliases` and fails the restore unless the expected alias target is observed.

After restore:

```bash
python scripts/rag_index.py reconcile
python scripts/ragbot.py doctor
```

Do not reopen traffic until readiness and retrieval smoke tests pass.

The existing recovery bundle snapshots the active physical index. Retired rollback collections are retention assets rather than the primary disaster-recovery payload; if a deployment requires rollback history to survive total Qdrant loss, those retained collections need an additional snapshot/retention policy.

## 11. Admin API

Global-admin-only endpoints:

```text
GET  /admin/indexes
GET  /admin/indexes/contracts
POST /admin/indexes
GET  /admin/indexes/{id}
POST /admin/indexes/{id}/shadow
POST /admin/indexes/{id}/ready
POST /admin/indexes/{id}/activate
POST /admin/indexes/{id}/rollback
POST /admin/indexes/reconcile/run
POST /admin/indexes/retention/prune
```

Large `build` work is intentionally not exposed as a blocking HTTP request; use the operator CLI/job path.

## Invariants

1. One IndexVersion has one immutable embedding contract and vector schema.
2. Incompatible embedding changes never mutate the query-visible collection in place.
3. The Qdrant alias is the serving cutover primitive; PostgreSQL is the durable lifecycle/history control plane.
4. Source-generation visibility remains enforced inside every IndexVersion.
5. Promotion/rollback must use a catalog snapshot that still matches PostgreSQL.
6. No already-running ingestion may straddle an IndexVersion cutover; new worker claims are gated only for the short cutover attempt.
7. Active chunk embedding identity changes transactionally with the IndexVersion so incremental reuse remains correct.
8. A candidate never becomes active solely because its build succeeded; validation evidence and explicit promotion are separate states.
9. Old physical indexes are retained explicitly and deleted only by retention policy.
10. Retrieval-plan changes such as sparse/multi-vector fusion remain owned by the retrieval roadmap (#57); this lifecycle is the index-version foundation beneath them.
