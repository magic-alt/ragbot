# Ragbot live RAG evaluation

Ragbot now has two complementary retrieval test surfaces:

- `scripts/search_test.py` is an interactive/explainable smoke test for one or more queries.
- `scripts/rag_eval.py` runs a repeatable Golden Dataset against the live `/search` API and writes JSON, Markdown and HTML reports.

The live evaluator deliberately uses the running API instead of creating a fresh in-memory service. This means the report measures the actual index that was produced by your PDF ingestion jobs, together with the currently configured embedding model, vector store, lexical retrieval, RRF fusion and optional reranker.

## 1. Ingest the corpus

macOS/Linux:

```bash
python3 scripts/ragbot.py ingest /Users/you/ragbot/data \
  --tenant engineering \
  --tag pdf
```

Windows PowerShell:

```powershell
python .\scripts\ragbot.py ingest .\data `
  --tenant engineering `
  --tag pdf
```

Confirm that ingestion finishes with non-zero document/chunk counts.

## 2. Run the starter evaluation

macOS/Linux:

```bash
python3 scripts/rag_eval.py eval/datasets/pdf_retrieval_smoke.json \
  --tenant engineering \
  --open
```

Windows PowerShell:

```powershell
python .\scripts\rag_eval.py .\eval\datasets\pdf_retrieval_smoke.json `
  --tenant engineering `
  --open
```

The command writes:

```text
reports/rag-eval/
├── pdf-retrieval-smoke-YYYYMMDD-HHMMSS.json
├── pdf-retrieval-smoke-YYYYMMDD-HHMMSS.md
├── pdf-retrieval-smoke-YYYYMMDD-HHMMSS.html
└── latest.html
```

`latest.html` is convenient for repeatedly tuning chunk size, embedding model or reranking and refreshing one stable report path.

## 3. What the report measures

For labeled retrieval cases the evaluator reports:

- **Hit@1 / Hit@3 / Hit@5 / Hit@10** — fraction of queries with at least one relevant chunk in the first K results.
- **MRR@10** — reciprocal rank of the first relevant chunk, averaged over labeled cases.
- **Recall@10** — available when the Golden Dataset contains exact `expected_chunk_ids` or expected `pages`.
- **Pass rate** — per-case pass/fail based on each case's `max_rank` and optional answer checks.
- **Search p50 / p95 latency** — end-to-end `/search` HTTP latency.
- **Runtime diagnostics** — embedding backend/model/dimension, whether the embedding is semantic, vector store, repository and reranker state.
- **Per-result retrieval trace** — vector rank/score, lexical rank/score, RRF score and reranker score when available.

If `--with-answers` is enabled, the evaluator also calls `/chat` and records answer latency, confidence, citation count and optional answer expectations.

## 4. Starter dataset

`eval/datasets/pdf_retrieval_smoke.json` contains the three technical queries used while debugging the DeepSeek/LLM PDF corpus:

1. LoRA vs QLoRA;
2. quantization and GPU memory usage;
3. Chinese-to-English cross-lingual GPU-memory retrieval.

The starter relevance rules are keyword based. They are intentionally useful before you have manually labeled exact pages/chunks, but they are not a substitute for a mature Golden Dataset.

## 5. Golden Dataset schema

Example:

```json
{
  "name": "my-pdf-golden",
  "defaults": {
    "top_k": 10,
    "filters": {"source_types": ["pdf"]}
  },
  "thresholds": {
    "hit_at_5_min": 0.85,
    "mrr_at_10_min": 0.65,
    "recall_at_10_min": 0.80,
    "p95_search_ms_max": 2000,
    "semantic_embedding_required": true
  },
  "cases": [
    {
      "id": "qlora-definition",
      "category": "fact",
      "query": "What is QLoRA?",
      "relevance": {
        "expected_chunk_ids": ["known-good-chunk-id"],
        "max_rank": 5
      },
      "answer": {
        "contains_all": ["LoRA", "quantization"],
        "min_citations": 1
      }
    }
  ]
}
```

### Relevance selectors

A case becomes a labeled retrieval case when `relevance` contains at least one selector:

- `expected_chunk_ids`: exact gold chunk IDs. When present this is authoritative.
- `doc_ids`: restrict a match to one of the listed document IDs.
- `pages`: expected PDF pages; also enables page-based Recall@10.
- `path_contains`: expected source path/URL substring.
- `all_terms`: every term must appear in the retrieved chunk.
- `any_terms`: at least one term must appear.
- `max_rank`: highest acceptable first-relevant rank; default is 5.

When exact chunk IDs are not yet available, a practical progression is:

```text
keyword relevance
    ↓
manual review of top results
    ↓
label expected pages
    ↓
label exact chunk IDs
    ↓
CI-quality MRR / Recall gate
```

## 6. Answer evaluation

Add an `answer` block to a case:

```json
{
  "answer": {
    "contains_all": ["LoRA", "quantization"],
    "contains_any": ["4-bit", "NF4"],
    "min_citations": 1
  }
}
```

A case with an `answer` block automatically calls `/chat`. `--with-answers` calls `/chat` for every case even when it has no explicit answer expectations.

This is a deterministic smoke gate, not a complete faithfulness judge. For high-stakes evaluation, add a reviewed answer rubric or an optional LLM-as-judge/RAGAS layer after retrieval quality is stable.

## 7. Threshold gates

Supported dataset thresholds:

```text
pass_rate_min
hit_at_1_min
hit_at_3_min
hit_at_5_min
hit_at_10_min
mrr_at_10_min
recall_at_10_min
p95_search_ms_max
p95_answer_ms_max
semantic_embedding_required
```

By default failed thresholds are shown in the report but the process still exits successfully. Use:

```bash
python3 scripts/rag_eval.py eval/datasets/my_golden.json \
  --tenant engineering \
  --fail-on-threshold
```

for CI-style exit code `1` when a gate fails.

## 8. Interpreting the current local development mode

If the report says:

```text
embedding_backend = HashEmbedder
semantic_embedding = false
```

then the index is suitable for pipeline development, not for judging production semantic or Chinese-to-English retrieval quality. Configure a real embedding backend and re-ingest the PDFs before treating Hit/MRR as model-quality measurements.

A real embedding index and a HashEmbedder index must not be mixed. Changing embedding model/dimension is an index-contract change and requires re-indexing the corpus.

## 9. Recommended acceptance ladder

For an initial 50-100 question technical PDF Golden Dataset, a reasonable engineering workflow is:

```text
Stage A — smoke
Hit@5 >= 0.70
MRR@10 >= 0.45

Stage B — usable
Hit@5 >= 0.85
MRR@10 >= 0.65
Recall@10 >= 0.80 (exact/page labels)

Stage C — tuned
Hit@5 >= 0.92
MRR@10 >= 0.75
Recall@10 >= 0.90
```

These are project gates rather than universal RAG benchmarks. Set them from your own corpus, query distribution and failure cost.

## 10. A/B tuning workflow

Run the same Golden Dataset after changing one variable at a time:

```text
embedding model A vs B
chunk size 600 vs 800 vs 1200
chunk overlap 50 vs 100 vs 150
Top K 5 vs 10
reranker off vs on
```

Keep the JSON reports so changes can be compared quantitatively rather than by subjective answer inspection.
