# Retrieval promotion evidence integrity

The #61 promotion gate is fail-closed. A retrieval candidate may be technically valid and faster while its evaluation evidence is not valid enough to support promotion.

## Bounded relevance universe

Recall and nDCG require a known relevance universe. Ragbot accepts the following promotion evidence:

- `relevance.expected_chunk_ids`: exact relevant chunks are enumerated;
- `relevance.doc_ids`: exact relevant documents are enumerated;
- `relevance.relevant_total`: the dataset explicitly declares the relevant entity count;
- heuristic `any_terms`, `all_terms`, `path_contains`, or `pages` only when the merged dataset/case filter is scoped to exactly one `doc_id`.

A heuristic relevance case without that scope is useful for exploratory retrieval debugging, but it is **not promotion eligible**. The promotion runner rejects the Golden Dataset before executing baseline or candidate queries.

This specifically prevents a multi-document corpus from treating a term match as if exactly one relevant document existed. That error can make DCG exceed the assumed ideal DCG and produce an impossible nDCG above `1.0`.

## Metric-domain invariant

`PromotionPolicy` independently validates normalized quality metrics:

```text
0 <= Recall <= 1
0 <= MRR    <= 1
0 <= nDCG   <= 1
```

Missing or out-of-range values produce `PromotionDecision=reject`. They are evidence-integrity failures, not candidate wins or regressions.

## Production experiment rule

For resilient term-based datasets such as the DeepSeek retrieval suite, a real multi-document tenant must add a precise document scope before it can be used for promotion evidence. For example:

```json
{
  "defaults": {
    "top_k": 10,
    "filters": {
      "doc_ids": ["<the DeepSeek document id in this deployment>"]
    }
  }
}
```

The scope becomes part of the dataset content hash and therefore part of the immutable `EvaluationRun` evidence. An unfiltered report from a heterogeneous tenant should be retained only as a debug artifact and must not be used for `mark-ready` or alias activation.

## Separation from ranking experiments

Evidence-integrity fixes do not change retrieval ranking. Weighted RRF, sparse weights, candidate pool size, reranker configuration, and other ranking changes belong to distinct retrieval/fusion contracts and must be evaluated only after the evidence gate passes.
