# Factorial ingestion benchmark

This benchmark is the promotion gate between the current PDF ingestion default and a structured-parser / sentence-aware candidate pipeline.

It was added after the DeepSeek in Action component benchmark showed two separate effects:

- LlamaIndex sentence splitting preserved retrieval quality while producing substantially cleaner chunk boundaries than Ragbot fixed-window splitting.
- PyMuPDF extracted the Golden Dataset content much faster than PyPDF2 and preserved page/bbox provenance, but its fine-grained blocks exploded chunk cardinality when every parser block was chunked independently.

The experiment therefore tests parser choice, parser-block coalescing and splitter choice together while keeping the embedding model, cosine retrieval implementation, Golden Dataset, chunk budget and top-k fixed.

## Production bridge behavior

Block coalescing is implemented in `services.worker.parsing` and is **disabled by default**. Existing Source contracts keep their current chunk identities until an explicit configuration is promoted.

Example opt-in chunking configuration:

```json
{
  "provider": "llamaindex",
  "strategy": "sentence",
  "block_coalescing": {
    "enabled": true,
    "target_chars": 3200,
    "respect_page": true,
    "respect_section": true
  }
}
```

The bridge groups compatible parser blocks into chunking windows and then runs the configured chunker. It does not replace the chunker and it does not let windows cross page boundaries by default. Source block indices, source block kinds and bbox provenance are retained in metadata.

Tables, code, figures and image blocks remain structural boundaries rather than being merged into surrounding prose.

## Default compact design

With the default options the CLI evaluates six cells:

| Pipeline level | Ragbot fixed | LlamaIndex sentence |
| --- | --- | --- |
| PyPDF2 / raw page blocks | ✓ | ✓ |
| PyMuPDF / raw text blocks | ✓ | ✓ |
| PyMuPDF / coalesced text blocks | ✓ | ✓ |

PyPDF2/coalesced is omitted from the compact design because the legacy parser already emits page-scale blocks, making that row mostly redundant. Use `--design full` to run every parser × raw/coalesced × splitter combination.

## Install

For the current two-parser/two-splitter experiment:

```bash
./.venv/bin/python -m pip install -e \
  ".[worker,benchmark-frameworks,parser-pymupdf]"
```

Configure the same semantic embedding model used by the live Ragbot deployment. For the current local Qwen3 setup:

```bash
export EMBEDDING_MODEL="qwen3-embedding:8b"
export EMBEDDING_BASE_URL="http://127.0.0.1:11434"
export QDRANT_DIM="4096"
export EMBEDDING_TIMEOUT_SECONDS="300"
export EMBEDDING_BATCH_SIZE="8"
```

## Run the six-cell benchmark

```bash
./.venv/bin/python scripts/rag_factorial_benchmark.py \
  --dataset eval/datasets/deepseek_in_action_retrieval.json \
  --corpus "/Users/kaermax/ragbot/data/DeepSeek in Action.pdf" \
  --parsers pypdf2,pymupdf \
  --splitters ragbot,llamaindex \
  --design compact \
  --embedding env \
  --chunk-size 800 \
  --chunk-overlap 100 \
  --coalesce-target-multiplier 4 \
  --top-k 10 \
  --repetitions 3
```

Reports are written under `reports/rag-factorial/` as timestamped JSON/Markdown plus `latest.json` and `latest.md`.

## Metrics to use for promotion

Do not select a pipeline from Hit@1 alone. Read the following together:

- retrieval: Hit@1/3/5/10, MRR@10, nDCG@10, Recall@10 and category-level metrics;
- parser fidelity: extracted characters, label text coverage, page metadata coverage, bbox/table/section coverage;
- fragmentation: raw parser blocks → bridge blocks → final chunks;
- chunk shape: non-boundary end rate, character inflation and chunk length distribution;
- cost: parser seconds, segment seconds, embedding seconds, query latency and peak Python allocation.

A successful PyMuPDF coalescing result should materially reduce `raw_blocks→bridge_blocks` and final chunk count without losing `label_text_coverage` or retrieval quality.

## Promotion rule

The DeepSeek dataset is a development dataset, so it is suitable for regression and candidate screening but not sufficient to change production defaults on its own.

A production default should only be changed after the candidate also passes a reviewed corpus that includes at least:

- ordinary technical books/manuals;
- multi-column papers;
- datasheets/specifications;
- table-heavy reports;
- figure/caption content;
- numeric and page-specific questions;
- Chinese/English mixed content;
- hard paraphrases and hard negatives.

Recommended promotion sequence:

1. DeepSeek six-cell factorial benchmark: eliminate obviously bad pipeline cells.
2. Diverse PDF Golden Dataset: verify retrieval and structure fidelity.
3. Isolated ingestion benchmark: compare parse/chunk/embed/index wall time and storage cardinality.
4. Live Ragbot A/B collection: validate hybrid retrieval, reranker, citation and ACL behavior.
5. Only then change the Source default parser/splitter configuration.
