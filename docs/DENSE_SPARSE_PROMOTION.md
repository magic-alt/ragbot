# Dense + Sparse Retrieval Promotion

Issue #57 Phase 2 introduces one deliberately narrow retrieval experiment:
`qdrant_dense_sparse` versus the current `hybrid_rrf` control.

The goal is not to accumulate retrieval algorithms. The goal is to make one
native Qdrant dense+sparse representation measurable, rejectable, promotable
and reversible through the platform contracts established by #53 and #61.

## Controlled variables

The first candidate **must preserve the active dense embedding contract**.
Only the sparse representation and fusion location change.

Control:

```text
active dense IndexVersion
  -> dense candidate + PostgreSQL lexical candidate
  -> adaptive RRF
  -> optional existing reranker
```

Candidate:

```text
same dense embedding contract
  + SparseEmbeddingSpec
  -> named `dense` + named `sparse` Qdrant IndexVersion
  -> Qdrant Query API Prefetch
  -> native RRF
  -> same optional existing reranker
```

ColBERT, late interaction, multi-query, semantic MMR and additional reranker
variants are explicitly outside this phase. Changing more than one major
representation variable would make the evaluation result difficult to
attribute.

## Sparse contract

Enable the production sparse encoder with:

```bash
RAGBOT_SPARSE_ENABLED=true
RAGBOT_SPARSE_PROVIDER=fastembed
RAGBOT_SPARSE_MODEL=Qdrant/bm25
RAGBOT_SPARSE_VECTOR_NAME=sparse
RAGBOT_SPARSE_MODIFIER=idf
```

Optional:

```bash
RAGBOT_SPARSE_REVISION=
RAGBOT_SPARSE_TOKENIZER=
RAGBOT_SPARSE_LANGUAGE=
RAGBOT_SPARSE_BATCH_SIZE=32
```

`SparseEmbeddingSpec` is immutable and produces a content-derived
`sparse-...` contract ID. Changing model, revision, vector name, modifier,
tokenizer or language creates a new contract.

FastEmbed is an optional dependency:

```bash
pip install -e ".[sparse]"
```

The runtime lazily loads sparse model weights. Merely inspecting contracts does
not download or initialize a model.

## Build a candidate IndexVersion

List dense and sparse contracts:

```http
GET /admin/indexes/contracts
```

The response preserves the historical `items` field for dense embedding
contracts and adds `sparse_items`.

Create the candidate using the **currently active dense embedding contract**:

```json
POST /admin/indexes
{
  "embedding_contract_id": "<active dense contract>",
  "sparse_contract_id": "<configured sparse contract>"
}
```

The resulting IndexVersion records:

```json
{
  "vector_schema": {
    "dense": {
      "name": "dense",
      "dimension": 1536,
      "distance": "cosine"
    },
    "sparse": {
      "name": "sparse",
      "contract_id": "sparse-...",
      "provider_id": "fastembed",
      "model": "Qdrant/bm25",
      "modifier": "idf"
    },
    "fusion": {
      "provider": "qdrant",
      "method": "rrf"
    }
  }
}
```

Build it through the existing IndexVersion build operation. The builder embeds
every active chunk with the existing dense encoder and the selected sparse
encoder and writes both named vectors into the candidate physical collection.
The candidate remains `validating`; the active Qdrant alias is unchanged.

## Query a candidate before activation

`RetrievalRequest.index_version_id` is an internal evaluation selector. It is
not exposed by `/v1/search`.

For `qdrant_dense_sparse`, the engine validates all of the following before any
query executes:

- the IndexVersion exists and is `validating`, `ready` or `active`;
- its dense contract equals the runtime active dense contract;
- it declares a sparse contract;
- the configured sparse encoder has the exact same contract ID;
- the vector backend exposes the native hybrid Query API.

The trace records:

- `index_version_id`;
- dense and sparse representation contract IDs;
- dense embedding latency;
- sparse embedding latency;
- native Qdrant search latency;
- candidate count, fusion, reranking and final ranking evidence.

## Qdrant Query API

The candidate uses named vectors and Qdrant Query API semantics:

```text
Prefetch(dense query, using="dense")
Prefetch(sparse query, using="sparse")
            |
            v
        Fusion(RRF)
            |
            v
          top-k
```

Tenant/ACL/filter constraints are applied before candidate fusion. There is no
fallback from `qdrant_dense_sparse` to `hybrid_rrf`; missing capability or
contract mismatches fail fast.

## Golden Dataset evaluation

Run the baseline and candidate over exactly the same Golden Dataset:

```bash
python -m benchmarks.retrieval_plan_promotion \
  --golden eval/datasets/deepseek_in_action_retrieval.json \
  --candidate-index <index-version-id> \
  --tenant <tenant-id> \
  --repetitions 3 \
  --output reports/dense-sparse-promotion.json
```

The runner performs a warm-up outside the measured window, then persists two
immutable `EvaluationRun` rows:

```text
hybrid_rrf / active IndexVersion
        -> EvaluationRun(baseline)

qdrant_dense_sparse / candidate IndexVersion
        -> EvaluationRun(candidate)
```

Both runs contain:

- Golden Dataset content hash/version;
- code revision;
- retrieval plan;
- IndexVersion ID;
- dense embedding contract;
- candidate sparse contract when applicable;
- vector schema;
- Recall@10, MRR@10, nDCG@10, Hit@5/10 and pass rate;
- p50/p95/mean query latency and stage latency;
- category/query-slice evidence;
- explicit retrieval cost field.

Queries are stored in artifacts only as SHA-256 hashes; the runner does not
need raw query text in durable `RagRun` storage.

## Promotion gate

The same run creates and persists a `PromotionDecision` using #61's
`PromotionPolicy`.

Defaults reject any Recall/MRR/nDCG regression and allow at most 15% p95
latency increase and 25% cost increase. Missing required evidence is a reject,
not a pass.

The process is:

```text
EvaluationRun(baseline)
          +
EvaluationRun(candidate)
          |
          v
  PromotionDecision
      /        \
 reject       accept
                |
                v
          optional mark-ready
```

The benchmark **never activates the alias automatically**. `--mark-ready` may
transition an accepted validating IndexVersion to `ready`, attaching the
promotion evidence. Activation is still the existing explicit operation:

```http
POST /admin/indexes/{index_version_id}/activate
```

Rollback remains:

```http
POST /admin/indexes/{index_version_id}/rollback
```

This preserves the existing quiescence/publication barriers and makes an
accepted evaluation necessary evidence rather than an implicit deployment.

## Incremental ingestion after promotion

Once a dense+sparse IndexVersion is active, every worker/API replica must start
with the exact sparse encoder contract. Startup fails if the active index has a
sparse schema but the runtime has no sparse encoder or a different contract.

The Qdrant adapter inspects the active physical schema. Normal ingestion still
uses the established worker call path, but an upsert into a sparse-active
collection automatically generates and writes both named vectors. This avoids a
state where the initial full build has sparse vectors but later generations are
dense-only.

## CI gate

`Dense Sparse Retrieval Gate` runs against Qdrant `v1.19.0` and verifies:

1. sparse contract identity and mismatch rejection;
2. candidate creation with the same dense contract;
3. named dense+sparse IndexVersion build;
4. candidate query before alias activation;
5. real Qdrant named-vector Query API / Prefetch / RRF;
6. active hybrid incremental upserts;
7. Golden baseline/candidate EvaluationRuns and PromotionDecision.

The existing CI, PostgreSQL, Quality Observability and API/SDK gates continue to
run normally. Sparse retrieval is not promoted merely because this feature gate
passes; a real Golden Dataset `PromotionDecision=accept` remains the quality
evidence required for a deployment decision.
