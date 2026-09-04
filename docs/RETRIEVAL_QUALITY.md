# Retrieval quality engineering

Ragbot retrieval should be tuned with repeatable measurements, not by reading a few Top-K results by eye. The production default remains hybrid retrieval, but the search API and CLI expose vector-only and lexical-only modes so every ranking change can be ablated against the same corpus and relevance labels.

## Retrieval pipeline

```text
query
  ├─ semantic embedding -> vector candidates ─┐
  └─ lexical FTS       -> lexical candidates ─┼─ adaptive RRF
                                                │
                                                ├─ optional reranker over candidate pool
                                                │
                                                └─ final Top-K
```

`candidate_pool` is a recall budget before reranking; `top_k` is the number of final results. The default pool is `max(40, top_k * 4)` and can be overridden with `RAGBOT_RETRIEVAL_CANDIDATE_POOL` or per search request, capped at 200.

## Retrieval modes

Use the same query in all three modes before changing weights or models. Add `--no-rerank` when the goal is a clean first-stage ablation:

```powershell
python .\scripts\ragbot.py search `
  "What techniques lower VRAM consumption when running large language models?" `
  --tenant engineering --mode vector --top-k 10 --no-rerank --explain

python .\scripts\ragbot.py search `
  "What techniques lower VRAM consumption when running large language models?" `
  --tenant engineering --mode lexical --top-k 10 --no-rerank --explain

python .\scripts\ragbot.py search `
  "What techniques lower VRAM consumption when running large language models?" `
  --tenant engineering --mode hybrid --candidate-pool 50 --top-k 10 --no-rerank --explain
```

`--explain` prints:

- embedding backend/model and whether it is semantic;
- vector and lexical candidate counts;
- raw vector similarity and raw FTS score for each result;
- adaptive fusion weights and lexical confidence;
- whether the reranker is configured/requested/applied;
- pre-rerank score and reranker score;
- actual reranker candidate count.

JSON responses retain the existing `_retrieval.vector.score`, `_retrieval.lexical.score` and `_retrieval.rrf_score` compatibility fields while adding `raw_score`, `pre_rerank_score`, fusion policy and retrieval context.

## Adaptive hybrid policy

RRF is still rank based, but modality weights are no longer always 50/50.

- no vector candidates -> lexical 1.0;
- no lexical candidates -> vector 1.0;
- strong lexical query-term coverage -> vector 0.5 / lexical 0.5;
- moderate lexical coverage -> vector 0.65 / lexical 0.35;
- weak lexical coverage -> vector 0.8 / lexical 0.2;
- CJK query whose lexical candidates contain no CJK evidence -> vector 0.9 / lexical 0.1;
- HashEmbedder development fallback keeps lexical-first behavior.

The cross-language rule prevents an English corpus chunk that happens to contain one ASCII token such as `GPU` from receiving the same lexical authority as a genuinely strong full-query lexical match.

These weights are deliberately simple and observable. Change them only after running the regression dataset and comparing vector, lexical and hybrid metrics.

## Do we need Qwen embedding?

Ragbot itself cannot create useful semantic vectors without an embedding model. `HashEmbedder` is deterministic infrastructure scaffolding for development and tests; it is not a semantic model and must not be used to judge RAG quality.

A local semantic model is fully supported. For the current English-document + Chinese/English-query use case, **Qwen3-Embedding is the recommended local multilingual baseline**, not a hard dependency. Ragbot keeps the OpenAI-compatible embedding abstraction so hosted OpenAI-compatible services, TEI/vLLM, Ollama and other models can be compared with the same benchmark.

Recommended starting points:

| Model | Native dimension | Role |
| --- | ---: | --- |
| `qwen3-embedding:0.6b` | 1024 | default local benchmark; smallest Qwen3 multilingual option |
| `qwen3-embedding:4b` | 2560 | higher-quality local workstation option |
| `qwen3-embedding:8b` | 4096 | highest-quality Qwen3 baseline when memory/latency budget allows |

Qwen3 Embedding supports multilingual and cross-lingual retrieval and benefits from a query-side task instruction. Ragbot automatically applies a retrieval instruction to Qwen3 queries while leaving document text unchanged. Override it only for controlled experiments with `EMBEDDING_QUERY_INSTRUCTION`.

### Ollama example

```powershell
ollama pull qwen3-embedding:0.6b
```

`.env`:

```dotenv
EMBEDDING_MODEL=qwen3-embedding:0.6b
EMBEDDING_BASE_URL=http://127.0.0.1:11434
QDRANT_DIM=1024
```

No fake embedding API key is required for `localhost`, loopback or `host.docker.internal` endpoints. A remote endpoint without credentials must be explicitly trusted with:

```dotenv
EMBEDDING_ALLOW_ANONYMOUS=true
```

This prevents an empty hosted-provider key from silently becoming an anonymous remote request.

If you use persistent Qdrant, use a collection with the correct vector dimension. Do not point a 1024D Qwen index at an existing 1536D OpenAI collection.

## Embedding-model changes force re-vectorization

Chunk reuse is now fenced by:

- embedding model name;
- embedding dimension;
- existing lexical/content/source reuse identity.

The embedding identity is persisted in PostgreSQL chunk metadata and ingestion Job stats. Therefore unchanged text encoded with `text-embedding-3-small` will not be silently reused after switching to `qwen3-embedding:0.6b`.

Changing model/dimension still requires a compatible Qdrant collection. For clean A/B experiments, use separate collections when practical.

## DeepSeek in Action regression dataset

The repository includes:

```text
eval/datasets/deepseek_in_action_retrieval.json
```

It contains four concepts with three query forms each:

- exact English keyword query;
- English semantic paraphrase;
- Chinese -> English cross-lingual query.

The relevance labels use concepts/phrases instead of chunk IDs so the dataset remains useful after re-chunking.

Run the normal evaluator:

```powershell
python .\scripts\rag_eval.py `
  .\eval\datasets\deepseek_in_action_retrieval.json `
  --tenant engineering
```

Run all retrieval modes side by side, with reranking disabled by default so first-stage retrieval is isolated:

```powershell
python .\scripts\retrieval_ablation.py `
  .\eval\datasets\deepseek_in_action_retrieval.json `
  --tenant engineering `
  --candidate-pool 50 `
  --output .\reports\rag-eval\deepseek-ablation.json
```

Then measure the configured reranker's incremental lift separately:

```powershell
python .\scripts\retrieval_ablation.py `
  .\eval\datasets\deepseek_in_action_retrieval.json `
  --tenant engineering `
  --candidate-pool 50 `
  --with-reranker
```

The ablation report compares macro `Recall@1/3/5/10` and `MRR@10`, plus separate exact/paraphrase/cross-lingual category summaries, for vector, lexical and hybrid retrieval.

## How to interpret the ablation

### Vector good, lexical bad, hybrid worse than vector

The lexical branch is diluting semantic retrieval. Inspect lexical query-term coverage and adaptive weights before changing embedding models.

### Lexical good, vector bad

The embedding model or query instruction is weak for the domain/language pair. Compare another semantic model and re-vectorize the corpus.

### Vector and lexical both find relevant chunks, but final rank is poor

Increase candidate recall if necessary and enable a reranker. Reranking is a precision layer; it cannot recover evidence that never entered the candidate pool.

### Relevant evidence absent from vector Top-50

Treat this as an embedding/corpus representation problem before tuning RRF or reranking.

### Exact queries work but paraphrase/cross-lingual queries fail

This is the strongest signal that semantic embedding quality is the limiting layer. Qwen3-Embedding is a suitable local baseline for this specific multilingual case.

## Reranker candidate pool

When `RAGBOT_RERANK_ENABLED=true`, the reranker receives up to `candidate_pool` candidates instead of being hard-wired to `top_k * 2`. This makes recall and precision budgets independently tunable.

Start with:

```dotenv
RAGBOT_RETRIEVAL_CANDIDATE_POOL=50
RAGBOT_RERANK_ENABLED=true
```

First run `retrieval_ablation.py` without reranking, then repeat with `--with-reranker`. Compare MRR/Recall and latency. Avoid increasing the pool indefinitely: once relevant evidence is consistently inside the candidate set, larger pools mostly add reranker cost.

## Current chunking boundary

The PDF chunker still uses character-window splitting. The retrieval benchmark intentionally isolates ranking first. Once vector-only recall is stable, the next quality iteration should evaluate page-aware extraction, de-hyphenation, heading/paragraph-aware splitting and page/section citation metadata as a separate experiment rather than mixing chunking changes with embedding/RRF changes in the same benchmark run.
