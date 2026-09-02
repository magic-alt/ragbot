# 1000-PDF Scale Benchmark

This document records the deterministic 1000-PDF integration and retrieval benchmark used to validate Ragbot's PDF ingestion path against real PostgreSQL and Qdrant services.

The benchmark is an **offline integration/capacity baseline**, not a claim about production semantic-embedding quality. It intentionally uses Ragbot's deterministic `HashEmbedder` so the result is reproducible in GitHub Actions without external model or API dependencies.

## Validated baseline

| Item | Value |
| --- | --- |
| Repository | `magic-alt/ragbot` |
| Pull request | `#6` |
| Validated code SHA | `5a4468014f4f9ce8fc8942e23c0d734128764ac4` |
| Workflow | `PDF Scale Benchmark` |
| Successful run | `33491876902` |
| Job | `99805015665` |
| Runner | GitHub-hosted `ubuntu-latest` |
| Python | 3.12 |
| PostgreSQL | `postgres:16-alpine` |
| Qdrant | `qdrant/qdrant:v1.19.0` |
| Embedding baseline | `hash-128` |

The same validated code SHA also passed the repository's ordinary CI workflow before this document was added.

## Workload

The benchmark generates and ingests a deterministic engineering corpus through the real PDF parser, source pipeline, PostgreSQL repository, Qdrant adapter, and hybrid retriever.

| Parameter | Value |
| --- | ---: |
| PDFs | 1,000 |
| Pages per PDF | 2 |
| Retrieval queries | 100 |
| Unchanged PDFs re-ingested | 50 |
| Chunk size | 1,000 |
| Chunk overlap | 120 |
| Embedding dimension | 128 |
| Deterministic identity terms | 4 per document |

Each generated document contains four deterministic identity terms in addition to technical prose. This is deliberate: a single identity token is not a valid oracle for a 1,000-document benchmark when a 128-bucket hash embedding is used, because token-to-bucket collisions are then unavoidable. The four-term identity remains fully offline and deterministic while allowing the benchmark to exercise both vector and lexical retrieval without making a single hash bucket the document identity.

## Results

The successful run uploaded `pdf-scale-results.json` with the following measurements.

### Ingestion

| Metric | Result |
| --- | ---: |
| PDFs succeeded | **1,000 / 1,000** |
| PDFs failed | **0** |
| Chunks persisted/indexed | **24,128** |
| PDF generation | 8.466 s |
| Ingestion | 207.780 s |
| Documents / second | 4.813 |
| Chunks / second | 116.123 |

### Retrieval

| Metric | Gate | Result | Status |
| --- | ---: | ---: | --- |
| Recall@5 | >= 0.980 | **1.000** | PASS |
| MRR | >= 0.950 | **1.000** | PASS |
| Latency P50 | informational | 8.051 ms | PASS |
| Latency P95 | informational | 8.756 ms | PASS |
| Latency mean | informational | 8.191 ms | PASS |

### Unchanged re-ingestion

| Metric | Gate | Result | Status |
| --- | ---: | ---: | --- |
| Reuse rate | >= 0.990 | **1.000** | PASS |
| Chunks reused | informational | 1,206 | PASS |
| Chunks rewritten | informational | **0** | PASS |
| Re-ingestion time | informational | 6.882 s | PASS |

### Storage and process footprint

| Metric | Result |
| --- | ---: |
| Qdrant points | 24,128 |
| PostgreSQL database | 35,240,863 bytes (~33.61 MiB) |
| Maximum RSS reported by runner | 435,452 KiB (~425.25 MiB) |
| End-to-end benchmark time | 227.297 s |
| Recorded failures | 0 |

The benchmark also asserts that the Qdrant point count exactly matches the number of produced chunks.

## Failure that led to the retrieval fix

The preceding 1000-PDF run (`33484276209`, job `99780651277`) successfully ingested all 1,000 PDFs with zero ingestion failures, but failed the retrieval gate:

```text
Recall@5 0.190 < 0.980
```

Two independent issues were identified.

### 1. Hybrid RRF modality crowd-out

The hybrid retriever fuses vector and PostgreSQL full-text rankings with reciprocal-rank fusion (RRF). The previous defaults weighted the vector branch at `0.6` and the lexical branch at `0.4`, with `k=60` and ten candidates fetched per branch.

For disjoint result sets this means:

```text
vector rank 10  = 0.6 / (60 + 10) = 0.00857
lexical rank 1  = 0.4 / (60 + 1)  = 0.00656
```

Therefore every one of the ten vector candidates could outrank even the best lexical candidate. A nominally hybrid top-5 result could become vector-only even when PostgreSQL FTS had the exact lexical match.

The default fusion weights are now balanced at `0.5 / 0.5`. Explicit caller-supplied weights remain supported. A regression test verifies that a disjoint lexical ranking can enter the final retrieval window instead of being structurally suppressed.

### 2. Single-token hash identity collisions

The original synthetic corpus assigned one unique marker token to each document while using a 128-dimensional hash embedding. Mapping 1,000 unique document-marker tokens into 128 hash buckets necessarily creates collisions, so the benchmark was partly measuring an artificial identity collision rather than end-to-end retrieval behavior.

The corpus now uses four independent deterministic marker terms per document. The benchmark remains offline, deterministic, and uses the same 128-dimensional hash embedder; the change only removes the invalid assumption that one 128-way hash bucket can uniquely identify 1,000 documents. A regression test verifies all 4,000 generated marker terms are unique before hashing.

No quality gate was lowered as part of either fix.

## Reproduction

The GitHub Actions workflow starts real PostgreSQL and Qdrant service containers, installs the benchmark dependencies, applies the full migration chain, waits for Qdrant readiness, and executes:

```bash
python -m benchmarks.pdf_scale \
  --documents 1000 \
  --pages-per-document 2 \
  --queries 100 \
  --collection "ragbot_pdf_scale_${GITHUB_RUN_ID}_${GITHUB_RUN_ATTEMPT}" \
  --output pdf-scale-results.json
```

For a local reproduction, provide the same service endpoints used by the benchmark code:

```bash
export POSTGRES_TEST_DSN='postgresql://ragbot:ragbot@127.0.0.1:5432/ragbot'
export QDRANT_URL='http://127.0.0.1:6333'
python -m services.api.app.storage.migrations
python -m benchmarks.pdf_scale \
  --documents 1000 \
  --pages-per-document 2 \
  --queries 100 \
  --output pdf-scale-results.json
```

The default gates remain:

- `Recall@5 >= 0.98`
- `MRR >= 0.95`
- unchanged re-ingestion reuse `>= 0.99`
- all requested documents must ingest successfully
- Qdrant point count must equal the produced chunk count

## Interpretation and limitations

This benchmark demonstrates that the tested Ragbot revision can ingest 1,000 generated PDFs end-to-end, persist/index 24,128 chunks in real PostgreSQL and Qdrant services, satisfy the deterministic corpus retrieval gates, and reuse unchanged chunks on re-ingestion.

It does **not** establish recall for arbitrary natural-language corpora, production-scale concurrency, multi-tenant saturation limits, or the quality of a production embedding/reranking model. Those require separate representative datasets and model-specific evaluations.

Because this document itself creates a new commit SHA, PR #6 performs one additional full 1000-PDF workflow run on the final documentation-containing SHA before the PR is marked ready for review. That final validation run is intentionally a PR check rather than embedded back into this file, avoiding a self-referential documentation-commit loop.
