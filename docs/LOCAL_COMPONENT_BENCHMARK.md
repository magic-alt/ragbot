# Local splitter and parser benchmark

Use this benchmark after the Level 2 Golden Dataset gate and the Level 3 native framework comparison. Its purpose is attribution: determine whether a quality difference comes from segmentation or parsing before changing Ragbot's production architecture.

## Why this benchmark exists

Whole-framework comparisons change too many variables at once. LangChain and LlamaIndex can differ in splitter semantics, native vector-store behavior, retriever wrappers and metadata handling. A whole-framework win therefore does not prove that replacing Ragbot's retrieval stack is the right change.

This benchmark runs two controlled experiments:

| Experiment | Changed variable | Fixed variables |
| --- | --- | --- |
| Splitter-only | Ragbot fixed-window vs LangChain recursive character vs LlamaIndex sentence splitting | parsed page text, embedding model, cosine search, Golden Dataset, top-k |
| Parser-only | PyPDF2 vs PyMuPDF vs optional Docling/Unstructured | Ragbot chunking, embedding model, cosine search, Golden Dataset, top-k |

Both experiments reuse `benchmarks.rag_native_compare` scoring, so concept labels used by development datasets and page/doc/path labels used by mature datasets have the same semantics as Level 3.

## Install

The DeepSeek experiment already needs the framework benchmark extras:

```bash
./.venv/bin/python -m pip install -e ".[worker,benchmark-frameworks]"
```

For PyMuPDF:

```bash
./.venv/bin/python -m pip install -e ".[parser-pymupdf]"
```

Optional heavier parser experiments:

```bash
./.venv/bin/python -m pip install -e ".[parser-docling]"
./.venv/bin/python -m pip install -e ".[parser-unstructured]"
```

Do not install Docling/Unstructured merely to increase the number of backends. Add them when page structure, tables or difficult PDFs are part of the target workload.

## DeepSeek in Action: recommended run

Use the same Qwen3 embedding configuration used by the live Ragbot deployment:

```bash
export EMBEDDING_MODEL="qwen3-embedding:8b"
export EMBEDDING_BASE_URL="http://127.0.0.1:11434"
export QDRANT_DIM="4096"
export EMBEDDING_TIMEOUT_SECONDS="300"
export EMBEDDING_BATCH_SIZE="8"
```

Then run both local experiments:

```bash
./.venv/bin/python scripts/rag_component_benchmark.py \
  --dataset eval/datasets/deepseek_in_action_retrieval.json \
  --corpus "/absolute/path/to/DeepSeek in Action.pdf" \
  --components splitter,parser \
  --splitters ragbot,langchain,llamaindex \
  --parsers pypdf2,pymupdf \
  --embedding env \
  --chunk-size 800 \
  --chunk-overlap 100 \
  --top-k 10 \
  --repetitions 3
```

Reports are written to `reports/rag-components/` as JSON and Markdown, with `latest.json` and `latest.md` pointers.

## Splitter metrics

Quality metrics are the same family as the native framework comparison:

- Hit@1/3/5/10
- MRR@10
- Recall@10
- nDCG@10
- category-level metrics
- query p50/p95/mean latency

The controlled splitter experiment adds segmentation diagnostics:

- total chunks;
- mean/p50/p95/max chunk characters;
- `non_boundary_end_rate`: fraction of chunks ending away from punctuation/newline boundaries;
- `character_inflation_ratio`: total chunk characters divided by source characters, exposing overlap/duplication cost;
- `chunks_per_page`;
- split time and Python peak allocation.

Interpret quality and chunk economics together. A splitter that improves MRR by producing twice as many overlapping chunks may increase embedding cost and index size enough to be a poor production trade-off.

## Parser metrics

Parser comparison keeps Ragbot's chunking kernel fixed. In addition to retrieval metrics it records:

- parser time and documents/second;
- extracted characters and characters/page;
- `page_metadata_coverage`: fraction of physical PDF pages represented by parser page metadata;
- `label_text_coverage`: fraction of Golden Dataset concept-label cases whose answer-bearing terms survive parsing before chunking/retrieval;
- page/bbox/table/section block rates;
- block count, chunk count and Python peak allocation.

`label_text_coverage` is particularly useful for separating parser loss from retrieval loss. If a Golden case fails because its relevant terms never appear in parser output, changing the vector retriever cannot repair that failure.

## Decision rules

Do not promote a component because it wins one easy 12-case dataset. A useful promotion threshold is:

1. no material regression in Hit@5/Recall@10;
2. repeatable MRR/nDCG improvement on hard cases;
3. acceptable chunk inflation/index cost for splitters;
4. page/label fidelity improvement for parsers;
5. acceptable parsing/indexing latency and memory;
6. repeated result on at least one second corpus;
7. production Golden Dataset with reviewed stable labels before changing defaults.

The desired architecture is component-selective: if LlamaIndex splitting wins, use that splitter behind Ragbot's framework-neutral chunking interface; if PyMuPDF/Docling parsing wins, use that parser behind the normalized-document contract. Keep Ragbot's durable ingestion, ACL, multi-tenant, Qdrant/PostgreSQL retrieval and operational semantics unless a separate benchmark demonstrates a reason to replace them.

## Relationship to corpus-scope parity

This local component benchmark has no live-index scope mismatch because every backend consumes the same local corpus bytes. The native Level 3 benchmark still needs explicit live Ragbot corpus filtering when a tenant contains unrelated documents. Keep that as a separate methodology guardrail: local component attribution and live corpus-scope parity solve different sources of benchmark bias.
