# Local Ollama + Qwen3.8-27B RAG Validation

This guide validates a host-installed Ollama model against Ragbot's real indexed
knowledge path:

```text
local documents -> Ragbot ingestion -> embeddings -> Qdrant
                                             |
question -> Ragbot /search -> retrieved chunks -> agent -> Ollama Qwen3.8 -> answer + citations
```

The recommended topology keeps Ollama on the host so macOS/Windows/Linux GPU
acceleration remains simple, while Ragbot runs its durable PostgreSQL + Qdrant
stack in Docker.

## 1. Why this path

Ragbot already has an Ollama model provider. The important distinction is that
the **generation model and the embedding model are separate contracts**:

- `OLLAMA_MODEL=qwen3.8:27b` controls agent reasoning and answer generation.
- `EMBEDDING_*`, `QDRANT_DIM`, and `QDRANT_COLLECTION` control retrieval.
- Changing only the LLM does **not** require reindexing Qdrant.
- Changing the embedding model or vector dimension **does** require a compatible
  collection and reindex.

Do not change the embedding settings merely to test a different local LLM. Use
the same embedding model/dimension that created the existing Qdrant collection.

## 2. Prerequisites

- Docker + Docker Compose for Ragbot/PostgreSQL/Qdrant.
- Ollama installed on the host.
- Enough local memory for the selected Qwen3.8 variant.
- An already indexed Ragbot corpus, or local documents under `./data` to ingest.

Install the model:

```bash
ollama pull qwen3.8:27b
ollama list
```

Basic Ollama check:

```bash
curl http://127.0.0.1:11434/v1/models
```

## 3. Configure Ragbot to use host Ollama

Copy the normal environment template if `.env` does not exist:

```bash
cp .env.example .env
```

Set at least these values in `.env`:

```dotenv
RAGBOT_LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3.8:27b
OLLAMA_TIMEOUT_SECONDS=300
OLLAMA_REASONING_EFFORT=none
RAGBOT_DOCKER_OLLAMA_BASE_URL=http://host.docker.internal:11434
```

`OLLAMA_REASONING_EFFORT=none` is a good deterministic starting point for RAG
validation because it reduces hidden reasoning latency. After the baseline is
stable, compare `low`, `medium`, and `high` for difficult multi-hop questions.

Keep the **existing** retrieval settings. For example, if the collection was
created with OpenAI-compatible 1536-dimensional embeddings:

```dotenv
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=your-existing-key
EMBEDDING_BASE_URL=https://api.openai.com
QDRANT_COLLECTION=rag_chunks
QDRANT_DIM=1536
```

If your existing corpus uses another embedding service/model, preserve that
configuration instead.

## 4. Start durable Ragbot + Qdrant with host Ollama

Use the host-Ollama Compose overlay added by Ragbot:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.ollama-host.yml \
  up -d --build
```

Check services:

```bash
python scripts/ragbot.py status
python scripts/ragbot.py doctor --server http://127.0.0.1:8000
```

The overlay maps the API/worker to `host.docker.internal` and leaves PostgreSQL
and Qdrant on the normal durable Docker volumes.

### Linux host note

The overlay also defines:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

so Docker Engine on Linux can resolve the same hostname used by Docker Desktop.
Ollama itself must listen on an address reachable from Docker. If a host firewall
or Ollama bind policy blocks container access, verify connectivity from the API
container before debugging Ragbot.

## 5. Ingest local documents if needed

Put local files below `./data`. Then ingest a directory or PDF:

```bash
python scripts/ragbot.py ingest data/manuals --tenant default
```

or:

```bash
python scripts/ragbot.py ingest data/example.pdf --type pdf --tenant default
```

Wait for the job to complete. Then verify retrieval independently of generation:

```bash
python scripts/ragbot.py search \
  "a phrase or question that should exist in the documents" \
  --tenant default \
  --top-k 5
```

If `/search` returns irrelevant chunks, fix ingestion/embedding/retrieval before
judging Qwen's answer quality.

## 6. Run the new end-to-end validation

Use a question whose answer is definitely present in the indexed documents:

```bash
python scripts/ollama_rag_test.py \
  --model qwen3.8:27b \
  --tenant default \
  "What does the indexed document say about the system architecture?"
```

The script performs four independent checks:

1. `GET /v1/models` confirms `qwen3.8:27b` exists in Ollama.
2. A tiny direct Ollama completion confirms the model can actually generate.
3. Ragbot `/search` confirms the vector store returns evidence.
4. Ragbot `/chat` confirms the configured agent produces an answer and citations.

A successful run resembles:

```text
Ollama + Ragbot RAG validation: PASS
  Ollama: qwen3.8:27b @ http://127.0.0.1:11434
  Ragbot: http://127.0.0.1:8000 (vector_store=True)
  Retrieval: 5 chunks, top_score=..., latency=... ms
  Generation: ... chars, citations=..., latency=... ms
  Gates:
    OK  ollama_model_available
    OK  ragbot_ready
    OK  retrieval_count
    OK  answer_nonempty
    OK  citations_present
    OK  top_score
```

Write a machine-readable report:

```bash
python scripts/ollama_rag_test.py \
  --model qwen3.8:27b \
  --tenant default \
  --output tmp/qwen38-rag-report.json \
  --json \
  "Summarize the indexed deployment architecture."
```

Useful stricter gates:

```bash
python scripts/ollama_rag_test.py \
  --model qwen3.8:27b \
  --min-retrieved 3 \
  --min-top-score 0.50 \
  "Which safety constraints are defined in the indexed manual?"
```

The report also compares final citation chunk IDs with the chunks returned by
`/search`. That overlap is a useful smoke-level grounding signal; it is not a
substitute for a full RAG evaluation dataset.

## 7. Compare reasoning settings

First establish a baseline:

```dotenv
OLLAMA_REASONING_EFFORT=none
```

Restart the API after changing server-side settings:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.ollama-host.yml \
  up -d --build --force-recreate api
```

Then rerun the same fixed question set with `low`, `medium`, and `high`. Compare:

- retrieval count/top score (should stay approximately unchanged);
- answer correctness and citation coverage;
- generation latency;
- hallucination/unsupported-claim rate;
- behavior on multi-hop questions.

Do not compare models/settings using a different corpus or embedding index unless
that is the experiment you intend to run.

## 8. All-in-Docker Ollama alternative

Ragbot's base Compose file already contains an optional `ollama` service. If you
do not need host-native acceleration, use:

```bash
docker compose --profile ollama up -d --build
docker compose exec ollama ollama pull qwen3.8:27b
```

Set:

```dotenv
RAGBOT_LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3.8:27b
```

The base Compose path uses `http://ollama:11434`. Do **not** combine the
host-Ollama overlay with this topology unless you intentionally want the API to
use the host rather than the Compose Ollama service.

## 9. Important local-mode limitation

`python scripts/ragbot.py up --mode local` deliberately uses the development
in-memory repository/vector-store path. It is convenient for API development,
but it is **not** the recommended way to validate an existing durable Qdrant
knowledge base.

For a real-vector-database experiment use either:

- the Docker topology in this document; or
- a directly launched API with explicit `POSTGRES_DSN`, `QDRANT_URL`, matching
  embedding settings, and `RAGBOT_LLM_PROVIDER=ollama`.

## 10. Troubleshooting

### Model not installed

```text
ERROR: Ollama model 'qwen3.8:27b' is not installed
```

Run:

```bash
ollama pull qwen3.8:27b
```

### API container cannot reach Ollama

From the host, first verify:

```bash
curl http://127.0.0.1:11434/v1/models
```

Then verify the Docker endpoint/configuration. On the host-Ollama topology the
API must use:

```dotenv
RAGBOT_DOCKER_OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### Retrieval returns zero chunks

Check in this order:

1. ingestion job completed;
2. query uses the correct tenant;
3. API and worker point to the same Qdrant collection;
4. `QDRANT_DIM` matches the embedding output;
5. the query embedder is the same model used during ingestion;
6. ACL/security scope permits the indexed chunks.

### Qwen answers but citations are missing

Run `/search` first. If retrieval is good but `/chat` is not grounded, compare
`OLLAMA_REASONING_EFFORT`, inspect the retrieved evidence/citation IDs in the
JSON report, and use the repository's larger `scripts/rag_eval.py` evaluation
workflow for a multi-question benchmark.

### Structured JSON failures inside agent nodes

Ragbot's Ollama adapter uses Ollama's OpenAI-compatible
`/v1/chat/completions` endpoint and sends OpenAI-compatible `response_format`,
`max_tokens`, and `reasoning_effort`. If an older Ollama build rejects those
fields, upgrade Ollama before changing Ragbot back to native-only request fields.

## 11. Recommended validation sequence

For a meaningful model-quality comparison, keep retrieval fixed and run a small
golden set (for example 20-50 questions) covering:

1. direct factual lookup;
2. Chinese and English queries for the same fact;
3. multi-section synthesis;
4. questions whose answer is absent (the model should abstain);
5. near-duplicate/conflicting passages;
6. citation correctness.

Use `scripts/ollama_rag_test.py` for setup/smoke validation and
`scripts/rag_eval.py` for the larger repeatable evaluation report.
