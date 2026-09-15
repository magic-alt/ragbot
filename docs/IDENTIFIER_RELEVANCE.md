# Identifier relevance canonicalization

Ragbot keeps identifier relevance separate from ordinary full-text term matching.

## Why this exists

PDF extraction can preserve typographic separators that are visually equivalent to ASCII query text. A confirmed real example is `DeepSeek‑V3`, where the PDF contains U+2011 NON-BREAKING HYPHEN while the Golden Dataset query uses ASCII `-`.

Using ordinary `any_terms` for that case created a false negative even though the visible identifier was present in the source document.

## Contract

Golden Dataset cases that test exact technical/product identifiers may use:

```json
{
  "relevance": {
    "identifiers": ["DeepSeek-V3"],
    "max_rank": 5
  }
}
```

Identifier canonicalization applies only to `relevance.identifiers`. It does **not** change `any_terms`, `all_terms`, answer-text checks, or path matching.

The canonical form:

- applies Unicode NFKC normalization and case folding;
- removes Unicode dash punctuation, including U+2011 NON-BREAKING HYPHEN;
- removes ASCII hyphen, underscore, Unicode minus, soft hyphen, and whitespace;
- therefore treats `DeepSeek-V3`, `DeepSeek‑V3`, `DeepSeek_V3`, `DeepSeek V3`, and `DeepSeek-\nV3` as the same identifier;
- preserves unrelated punctuation rather than turning identifier matching into broad fuzzy text matching.

## Promotion evidence

`identifiers` is a heuristic relevance selector. It identifies matching evidence but does not, by itself, define the complete relevance universe.

For promotion runs it therefore follows the same fail-closed scope rule as term/path/page selectors: the evaluation must be scoped to exactly one document, or use an explicit relevance cardinality / exact chunk or document labels.

## DeepSeek v2 Golden Dataset

The two identifier cases in `eval/datasets/deepseek_in_action_retrieval_v2.json` use `identifiers` instead of `any_terms`:

- `identifier-deepseek-v3`
- `identifier-fp8`

This keeps ordinary term matching semantics unchanged while allowing PDF typography and line-wrap variants to compare correctly.

## What this does not prove

Fixing identifier relevance changes evaluator correctness only. It does not change the indexed text, embedding vectors, sparse vectors, Qdrant collection, retrieval fusion, or active alias. A previously failed identifier case must be rerun against the same IndexVersion/fusion contract before its retrieval status is considered resolved.
