# Promotion release gates

Ragbot distinguishes **relative candidate acceptance** from **release readiness**.

The original promotion policy compares a candidate with a baseline and rejects regressions in Recall, MRR, nDCG, p95 latency and cost. That remains the default behavior for backward compatibility.

A candidate can nevertheless improve relative to a baseline while both share the same absolute blind spot. Release policy can therefore opt into two additional gates:

- absolute candidate quality floors;
- critical Golden Dataset cases that must pass.

## Policy fields

`PromotionPolicy` supports:

```text
min_recall
min_mrr
min_ndcg
critical_case_ids
```

All are disabled by default (`None` / empty), so existing promotion calls retain their previous semantics.

When enabled:

- `min_recall`, `min_mrr`, and `min_ndcg` must be in `[0,1]`;
- the candidate metric must exist, be normalized, and meet the configured minimum;
- every critical case ID must exist in `EvaluationRun.artifacts.cases`;
- every critical case must have `retrieval_pass == true`;
- missing or duplicate critical-case evidence fails closed.

The release gates supplement rather than replace the existing relative regression and latency/cost gates.

## Why this matters

A real DeepSeek Weighted RRF experiment produced a relative acceptance for the 2:1 fusion candidate while baseline and candidate both missed the same identifier case. Relative acceptance was correct—it showed no regression and better ranking elsewhere—but it was not sufficient evidence for release readiness.

Critical-case gating lets the operator require that known high-value cases pass before mark-ready without forcing every future Golden Dataset to have global Recall=1.0.

## CLI examples

Weighted RRF:

```bash
python -m benchmarks.weighted_rrf_promotion \
  --golden eval/datasets/deepseek_in_action_retrieval_v2.json \
  --candidate-index <index-version-id> \
  --tenant <tenant-id> \
  --scope-doc-id <deepseek-doc-id> \
  --weights 2:1 \
  --min-recall 0.95 \
  --min-mrr 0.95 \
  --min-ndcg 0.95 \
  --critical-case identifier-deepseek-v3 \
  --repetitions 3
```

Dense+sparse promotion runner:

```bash
python -m benchmarks.retrieval_plan_promotion \
  --golden <dataset.json> \
  --candidate-index <index-version-id> \
  --tenant <tenant-id> \
  --min-recall 0.95 \
  --critical-case identifier-deepseek-v3
```

`--critical-case` is repeatable.

The same fields are available through the admin quality promotion API. API input trims and de-duplicates critical IDs and rejects blank IDs at request validation time before `PromotionPolicy` is constructed.

## Safety boundary

This policy change does not mark an IndexVersion ready and does not activate a Qdrant alias. Existing explicit IndexVersion lifecycle operations remain the deployment boundary.

Absolute floors should be chosen for the release dataset rather than hard-coded globally. In particular, Ragbot does not make `min_recall=1.0` a repository default because larger, more diverse Golden Datasets may intentionally use a lower validated floor plus critical-case requirements.
