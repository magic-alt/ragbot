# Weighted RRF promotion experiment

Issue #57 Phase 2.1 tunes only the Qdrant RRF fusion contract. It deliberately reuses the already-built named dense+sparse `IndexVersion`; dense embeddings, sparse embeddings, vector schema, reranker setting and document corpus stay fixed.

## Why this is a query contract, not an IndexVersion

The validated Phase-2 candidate stores:

```text
named dense vector  -> Qwen3 embedding contract
named sparse vector -> Qdrant/bm25 contract
```

Changing RRF `weights` or `k` does not alter those stored vectors. A new fusion configuration therefore gets a new `fusion-*` contract and `EvaluationRun`, but **must not rebuild the physical collection**.

## Qdrant request

For an explicit fusion experiment Ragbot uses Qdrant's native weighted RRF query:

```python
models.RrfQuery(
    rrf=models.Rrf(k=2, weights=[dense_weight, sparse_weight])
)
```

Prefetch order is always:

```text
[dense, sparse]
```

so `[3.0, 1.0]` means dense:sparse = 3:1. Tenant/ACL filters remain inside both Prefetch branches before fusion.

When no explicit `QdrantRrfFusionSpec` is selected, Ragbot preserves the Phase-2 equal/default native RRF request shape.

## Controlled grid

The default operator grid is:

```text
1:1  current Phase-2 control candidate
2:1  mild dense preference
3:1  primary follow-up candidate
5:1  strong dense preference
```

The baseline is still the production `hybrid_rrf` plan. Every pair receives:

- a content-derived `fusion-*` contract ID;
- its own candidate `EvaluationRun`;
- its own `PromotionDecision` against one shared baseline EvaluationRun;
- exact/paraphrase/cross-lingual and other category metrics;
- p50/p95/stage latency evidence.

The grid never marks an IndexVersion ready and never activates the alias.

## Expanded Golden Dataset

`eval/datasets/deepseek_in_action_retrieval_v2.json` covers:

- exact;
- identifier;
- paraphrase;
- cross-lingual;
- mixed Chinese/English;
- weak lexical overlap;
- strong lexical overlap.

The dataset uses resilient term-based labels, so production promotion requires one exact document scope. Use `--scope-doc-id`; the runner writes that scope into an in-memory dataset copy **before** hashing its dataset version, so the filter is part of immutable evidence.

## Command

With the existing validating candidate preserved from Phase 2:

```bash
python -m benchmarks.weighted_rrf_promotion \
  --golden eval/datasets/deepseek_in_action_retrieval_v2.json \
  --candidate-index idx-real-dense-sparse-20260915 \
  --tenant <TENANT_ID> \
  --scope-doc-id <DEEPSEEK_DOC_ID> \
  --weights 1:1,2:1,3:1,5:1 \
  --rrf-k 2 \
  --repetitions 3 \
  --output reports/weighted-rrf-promotion.json
```

Keep `--rerank` unset when comparing against the locally validated reranker-disabled deployment. If no fusion candidate satisfies the zero-quality-regression policy, the command exits non-zero and the IndexVersion remains validating.

## Promotion rule

The default policy remains fail-closed:

```text
Recall drop = 0
MRR drop    = 0
nDCG drop   = 0
```

Latency improvement cannot compensate for a quality regression. A fusion contract is only a production candidate if its own `PromotionDecision` is `accept`. Even then, choosing it as the deployed default and marking/activating the IndexVersion remain explicit operator actions.
