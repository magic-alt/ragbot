# Level 2 + Level 3 RAG benchmark

Ragbot now has two complementary quality layers above the synthetic system gate:

- **Level 2 — real-corpus Golden Dataset**: score the live Ragbot deployment on real indexed documents and real questions.
- **Level 3 — native framework comparison**: run Ragbot, LangChain and LlamaIndex against the same Golden Dataset and embedding model, then compare retrieval quality and latency.

Use `scripts/verify_ragbot.py` first. That command answers whether the deployed pipeline is healthy. Use this benchmark after the system gate passes to answer whether retrieval is good on your corpus and whether another framework implementation actually improves it.

## 1. Install benchmark dependencies

```bash
python3 -m pip install -e ".[dev,worker,benchmark-frameworks]"
```

The `benchmark-frameworks` extra contains the native LangChain and LlamaIndex retrieval dependencies. Ragbot keeps using its normal configured embedding endpoint.

For the local framework adapters, export the **same embedding configuration used by the running Ragbot deployment**. Example for the current local Qwen3 setup:

```bash
export EMBEDDING_MODEL=qwen3-embedding:8b
export EMBEDDING_BASE_URL=http://127.0.0.1:11434
export QDRANT_DIM=4096
```

If the live Ragbot search diagnostics report a different embedding model or dimension, the three-way benchmark fails by default. This prevents an invalid comparison such as Ragbot/Qwen3 versus LangChain/hash embeddings.

## 2. One command: Level 2 + Level 3

```bash
python3 scripts/rag_benchmark.py \
  --dataset eval/datasets/deepseek_in_action_retrieval.json \
  --corpus-dir /absolute/path/to/the/same/deepseek-corpus \
  --level all \
  --dataset-profile development \
  --backends ragbot,langchain,llamaindex \
  --embedding env \
  --ragbot-mode vector \
  --chunk-size 800 \
  --chunk-overlap 100 \
  --top-k 10 \
  --repetitions 3
```

The local `--corpus-dir` must represent the same logical corpus already indexed in the live Ragbot tenant/filter scope. The report records a deterministic SHA256 manifest of the local corpus so benchmark runs are reproducible.

Outputs are written under:

```text
reports/rag-benchmark/
├── level2/                 # existing live rag_eval JSON/Markdown/HTML
├── level3/                 # native three-framework JSON/Markdown
├── latest-summary.json
└── latest-summary.md
```

### Vector comparison first, hybrid comparison second

The default Level 3 Ragbot mode is `vector`. LangChain `InMemoryVectorStore` and LlamaIndex `VectorStoreIndex` are vector retrievers, so `--ragbot-mode vector` is the closest apples-to-apples retrieval comparison.

After that, run a second benchmark with:

```bash
--ragbot-mode hybrid --rerank
```

That second run measures Ragbot's production retrieval stack (Qdrant + PostgreSQL lexical retrieval + fusion + optional reranker). It is valuable product evidence, but it is no longer a pure vector-retriever comparison.

## 3. Level 2: real Golden Dataset gate

Run only the live deployment evaluation:

```bash
python3 scripts/rag_benchmark.py \
  --level 2 \
  --dataset eval/datasets/deepseek_in_action_retrieval.json \
  --dataset-profile development \
  --tenant default \
  --top-k 10 \
  --with-answers
```

Level 2 delegates to the existing `scripts/rag_eval.py`, so it uses the real deployed index, ACL/filter behavior, embedding runtime, Qdrant/PostgreSQL retrieval and optional `/chat` answer/citation path. Dataset thresholds still determine PASS/FAIL.

### Dataset maturity profiles

The benchmark audits the Golden Dataset before scoring it.

| Profile | Minimum cases | Labeled cases | Categories | Stable cross-framework labels |
| --- | ---: | --- | ---: | ---: |
| `development` | 10 | all | >=2 | observed |
| `production` | 50 | all | >=3 | >=80% |
| `off` | none | none | none | none |

A **stable label** uses one of:

- `relevance.expected_chunk_ids`
- `relevance.doc_ids`
- `relevance.pages`
- `relevance.path_contains`

`all_terms` / `any_terms` are useful during early dataset development, but they are more brittle because a relevant passage can be paraphrased by a parser or document revision. A production-grade cross-framework corpus should progressively move toward document/page/chunk labels reviewed by a human.

The repository's `deepseek_in_action_retrieval.json` is a real-corpus development dataset. It is deliberately small and concept-labeled; it is not presented as a production 50+ case benchmark.

## 4. Golden Dataset schema

The existing Ragbot schema is shared by Level 2 and Level 3:

```json
{
  "schema_version": 1,
  "name": "servo-manual-golden",
  "defaults": {
    "top_k": 10,
    "filters": {
      "path_prefix": "servo/"
    }
  },
  "thresholds": {
    "semantic_embedding_required": true,
    "hit_at_5_min": 0.85,
    "mrr_at_10_min": 0.65,
    "recall_at_10_min": 0.80,
    "p95_search_ms_max": 2000
  },
  "cases": [
    {
      "id": "ethercat-dc-sync",
      "category": "exact",
      "query": "What is the specified EtherCAT distributed-clock synchronization behavior?",
      "relevance": {
        "doc_ids": ["manuals/ethercat.md"],
        "pages": [12],
        "max_rank": 5
      },
      "answer": {
        "contains_any": ["distributed clock", "DC"],
        "min_citations": 1
      }
    }
  ]
}
```

For a mature dataset, use real user questions rather than converting headings into obvious keyword queries. Recommended category coverage:

1. exact factual lookup;
2. paraphrase / synonym queries;
3. Chinese-to-English and English-to-Chinese retrieval;
4. long-document section lookup;
5. multi-document questions;
6. table / numeric facts;
7. page-specific citations;
8. ambiguous queries;
9. hard negatives / answer-not-present cases for answer evaluation;
10. multi-hop questions where more than one passage is required.

The benchmark does not auto-invent relevance labels. Human-reviewed labels remain the ground truth.

## 5. Level 3 native framework comparison

Level 3 has a different purpose from the existing controlled splitter benchmark.

### Native comparison

```text
same local corpus + same Golden queries + same Ragbot Embedder

Ragbot
  -> live /search
  -> vector | hybrid | lexical
  -> deployed Qdrant/PostgreSQL/reranker

LangChain
  -> RecursiveCharacterTextSplitter
  -> Ragbot Embedder adapter
  -> langchain_core InMemoryVectorStore
  -> similarity_search_with_score

LlamaIndex
  -> SentenceSplitter
  -> Ragbot BaseEmbedding adapter
  -> VectorStoreIndex
  -> as_retriever().retrieve()
```

LangChain's current vector-store API exposes `add_documents` and `similarity_search`; its in-memory store computes cosine similarity. LlamaIndex's `VectorStoreIndex` exposes `as_retriever`, which returns raw `NodeWithScore` retrieval results without requiring an LLM synthesis call. The adapters therefore exercise each framework's actual retrieval abstraction while keeping the semantic model constant.

LlamaIndex receives a character tokenizer for this benchmark so `chunk_size=800` has a comparable budget to Ragbot/LangChain. Run a separate token-budget experiment if you want to evaluate LlamaIndex's default tokenizer semantics.

### Metrics

Every backend reports:

- Hit@1 / Hit@3 / Hit@5 / Hit@10
- MRR@10
- Precision@5 / Precision@10
- exact Recall@10 when the label has a known relevant-entity denominator
- nDCG@10
- pass rate using each case's `relevance.max_rank`
- query p50 / p95 / mean latency
- approximate single-process queries/second
- category-level Hit@5 / MRR / nDCG
- native indexing time
- native chunk count
- Python peak allocation observed by `tracemalloc`
- backend package versions
- per-case top hits and scores

Repeated chunks from the same relevant document do **not** earn repeated relevance credit. This avoids rewarding a framework simply because it emits many adjacent chunks from one document.

## 6. Controlled benchmark vs native benchmark

Keep both benchmarks because they answer different questions.

| Benchmark | What changes | What stays fixed | Best use |
| --- | --- | --- | --- |
| `benchmarks.rag_framework_compare` | splitter only | embedder + cosine search implementation | determine whether chunking caused the quality change |
| `benchmarks.rag_native_compare` | splitter + native vector-store/retriever | corpus + queries + labels + embedding model + top-k | compare realistic framework retrieval implementations |
| `scripts/rag_eval.py` | nothing; tests deployed Ragbot | live deployment | release/domain quality gate |
| `scripts/rag_benchmark.py` | orchestrates Level 2 + Level 3 | one dataset/run contract | normal researcher workflow |

Do not claim "framework X is better" from one synthetic run. Use a real corpus, 50-100+ reviewed cases, repeated runs, a fixed embedding endpoint and the same machine.

## 7. CI smoke

The framework benchmark workflow runs two checks:

1. the existing controlled three-splitter smoke;
2. the native LangChain + LlamaIndex in-memory smoke with a deterministic `HashEmbedder`.

CI deliberately does not run the live Ragbot/Qwen3 three-way benchmark because GitHub Actions does not have the user's local indexed corpus, Ollama model and tenant credentials. The native CI smoke proves the adapters still execute; semantic conclusions come from the local/staging `--embedding env` run.

## 8. Recommended promotion sequence

Use these gates in order:

```text
Level 0  repository unit/integration tests
   ↓
Level 1  scripts/verify_ragbot.py synthetic production-path gate
   ↓
Level 2  scripts/rag_benchmark.py --level 2 real Golden Dataset
   ↓
Level 3A native vector comparison: Ragbot vs LangChain vs LlamaIndex
   ↓
Level 3B controlled splitter attribution benchmark
   ↓
Level 3C Ragbot hybrid + reranker production comparison
   ↓
Level 4  scale/load benchmark
   ↓
Level 5  long-running reliability/fault injection
```

A framework should only be promoted into Ragbot's production path if the real Golden Dataset shows a repeatable gain and the change preserves Ragbot's durable job semantics, ACL/multi-tenant guarantees, generation fencing and incremental reuse contract.
