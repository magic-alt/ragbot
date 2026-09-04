# Ragbot vs LangChain vs LlamaIndex RAG benchmark

This benchmark answers a narrower and more useful question than "which RAG framework is faster?": **when the corpus, query set, embedding model and similarity search are held constant, does changing Ragbot's chunking strategy improve retrieval quality or indexing cost?**

The repository already has production-specific behavior that a framework benchmark must not accidentally change: durable Source/Job lifecycle, incremental reuse, tenant/ACL metadata, PostgreSQL lexical retrieval, Qdrant payloads, adaptive fusion, optional reranking and citations. A whole-framework benchmark that changes all of these at once cannot attribute a quality or latency difference to LangChain or LlamaIndex.

## 1. Controlled chunking benchmark

Install the optional benchmark dependencies:

```bash
pip install -e ".[dev,worker,benchmark-frameworks]"
```

Run the deterministic smoke benchmark:

```bash
python -m benchmarks.rag_framework_compare \
  --backends ragbot,langchain,llamaindex \
  --synthetic-documents 60 \
  --synthetic-queries 30 \
  --output rag-framework-benchmark.json
```

This mode uses `HashEmbedder`. It is useful for dependency, CPU, memory and regression smoke tests. It is **not** evidence about semantic RAG quality.

For quality comparison, configure the same real Ragbot embedding endpoint used in production and run:

```bash
export EMBEDDING_MODEL=qwen3-embedding:4b
export EMBEDDING_BASE_URL=http://127.0.0.1:11434
export QDRANT_DIM=2560

python -m benchmarks.rag_framework_compare \
  --corpus-dir ./data/benchmark-corpus \
  --golden ./eval/datasets/framework_golden.json \
  --embedding env \
  --chunk-size 800 \
  --chunk-overlap 100 \
  --top-k 10 \
  --output rag-framework-semantic.json
```

The Golden Dataset uses the existing Ragbot evaluation shape. This benchmark needs document-level relevance labels through `relevance.doc_ids` or `relevance.path_contains` so the same labels remain valid even when different splitters create different chunk IDs.

Example:

```json
{
  "name": "framework-golden",
  "cases": [
    {
      "id": "retry-budget",
      "query": "What retry budget does the ingestion worker use?",
      "relevance": {
        "doc_ids": ["operations/worker.md"]
      }
    }
  ]
}
```

### Controlled variables

All tested backends use the same:

- source documents;
- queries and relevance labels;
- Ragbot `Embedder` instance;
- embedding dimension;
- cosine search implementation;
- `top_k`;
- character-equivalent chunk budget.

Only the splitter changes:

| Backend | Splitter |
| --- | --- |
| `ragbot` | current fixed character window |
| `langchain` | `RecursiveCharacterTextSplitter` |
| `llamaindex` | `SentenceSplitter` |

For the LlamaIndex comparison the benchmark passes a character tokenizer so `chunk_size=800` means approximately the same budget as the current Ragbot and LangChain runs. A second token-budget experiment should be run separately when selecting the final production strategy.

### Metrics

The JSON report records:

- split wall time and chunks/second;
- embedding wall time;
- total wall time;
- query p50/p95/mean latency;
- Hit@1/3/5/10 and MRR@10;
- chunk count and length distribution;
- rate of chunks ending away from a sentence/paragraph boundary;
- Python peak allocation observed by `tracemalloc`.

Run at least five repetitions for performance conclusions and compare medians/p95 rather than one run. Keep the embedding endpoint, model, hardware and dependency versions fixed.

## 2. Production end-to-end benchmark

The controlled benchmark isolates chunking. It does **not** measure PostgreSQL, Qdrant, ACL filtering, hybrid retrieval, reranking, Source reconciliation or HTTP overhead. After selecting candidate splitters, run a second A/B/C test through the real Ragbot pipeline:

```text
A: Ragbot parser + Ragbot fixed splitter + Ragbot embedder + Ragbot PG/Qdrant retrieval
B: Ragbot parser + LangChain splitter      + Ragbot embedder + Ragbot PG/Qdrant retrieval
C: Ragbot parser + LlamaIndex splitter     + Ragbot embedder + Ragbot PG/Qdrant retrieval
```

Use separate Source/index generations or separate Qdrant collections. Do not mix chunks created by different splitter contracts in one knowledge generation.

For each variant:

1. ingest the exact same corpus;
2. record documents/s, chunks/s, embedding calls/batches, total wall time and peak RSS;
3. run the existing `scripts/rag_eval.py` Golden Dataset;
4. compare Hit@K, MRR, Recall, `/search` p50/p95 and optional answer/citation checks;
5. re-ingest unchanged content and compare reuse rate;
6. repeat with reranking disabled and enabled to separate first-stage retrieval from reranker gains.

Recommended benchmark matrix:

| Dimension | Values |
| --- | --- |
| splitter | Ragbot / LangChain recursive / LlamaIndex sentence |
| chunk size | 400 / 800 / 1200 |
| overlap | 0 / 80 / 160 |
| retrieval | vector / lexical / hybrid |
| reranker | off / on |
| query language | English / Chinese / cross-language |
| corpus type | prose PDF / technical PDF / Markdown / source code |

Do not cross every dimension in the first run. Start with splitter x chunk size using hybrid retrieval and the production embedding model, then perform focused ablations around the best two configurations.

## 3. Recommended integration boundary

Ragbot should remain the owner of lifecycle and security semantics. Frameworks should be optional adapters behind stable Ragbot ports.

```text
Source connector
    -> DocumentParser
    -> Chunker                         <- Ragbot / LangChain / LlamaIndex adapters
    -> normalized Ragbot Chunk
    -> Embedder                        <- Ragbot API adapter; optional framework adapters
    -> Ragbot PostgreSQL + Qdrant
    -> Ragbot hybrid Retriever
    -> optional reranker
    -> Agent + citations
```

Introduce a `Chunker` protocol and registry instead of embedding split logic in each connector/job. Suggested configuration:

```yaml
chunking:
  provider: langchain      # ragbot | langchain | llamaindex
  strategy: recursive
  chunk_size: 800
  chunk_overlap: 100
  version: 1
```

Persist at least these fields on every chunk:

```text
chunker_provider
chunker_strategy
chunker_version
chunker_config_hash
embedding_model
embedding_dimension
```

Include them in the incremental-reuse identity. Changing any of them is an index-contract change and must cause safe reprocessing/reindexing.

Do **not** allow a third-party framework to become the canonical owner of Ragbot chunk IDs, tenant IDs, ACL payloads, Source generations or deletion semantics.

## 4. Where LangChain can help

The highest-value LangChain use is the splitter package, not replacing the Ragbot runtime:

- generic text: recursive paragraph/newline/word-aware splitting;
- Markdown/code: language-aware separator sets;
- experimental Qdrant dense/sparse/hybrid retrieval as an isolated benchmark backend.

The last item should use a separate collection. Ragbot currently has a deliberate PostgreSQL FTS/CJK + Qdrant semantic + adaptive RRF design; replacing it with framework-default hybrid retrieval changes both quality semantics and the ACL/filter contract.

## 5. Where LlamaIndex can help

Useful candidates are:

- `SentenceSplitter` for prose-heavy PDF/document corpora;
- semantic splitting for selected long heterogeneous documents, benchmarked against its extra embedding cost;
- metadata transformations as experiments;
- ideas from `IngestionPipeline` such as transformation caching, async execution and parallel processing.

Ragbot already has durable Job retries, Source fencing and content/version-aware reuse. Those production semantics should not be replaced by an in-process framework ingestion pipeline. Instead, copy or adapt the useful transformation-level ideas inside the Ragbot worker boundary.

## 6. Architecture work that matters more than framework adoption

The code review identified several likely higher-ROI performance and reliability improvements:

1. run Qdrant vector search and PostgreSQL lexical search concurrently in hybrid mode; they are independent first-stage branches today;
2. eliminate reranker candidate N+1 SQL reads by adding a batch `get_chunks(ids)` repository operation or by reusing stored payload text;
3. add async/concurrent embedding batches with bounded rate limiting and retry/backoff;
4. avoid materializing an entire large Source twice (`previous_chunks` plus `candidate_chunks`) by moving toward bounded staged-generation processing;
5. preserve PDF page/block metadata before chunking instead of flattening every page into one string;
6. formalize `VectorStore`/`Chunker` protocols instead of `object` duck typing and source-type `if/elif` dispatch;
7. implement staged generation activation/outbox/reconciliation across PostgreSQL and Qdrant to close the current non-atomic publish window;
8. record embedding/splitter throughput, batch count, retry count, token count and cost as first-class benchmark/production metrics.

Framework adoption should be judged against these changes with the same Golden Dataset and capacity tests.
