# One-command local Ollama PDF RAG smoke test

Use `scripts/ollama_pdf_rag_test.py` when the goal is to validate the complete
local path, not only one component:

```text
./data/*.pdf
    -> Ragbot worker
    -> Ollama qwen3-embedding:0.6b
    -> Qdrant
    -> Ragbot /search
    -> Ragbot agent
    -> Ollama qwen3.8:27b
    -> answer + citations
```

The script starts the durable Docker Compose stack automatically. Ollama stays on
the host so macOS Metal / Windows GPU / Linux GPU acceleration remains native.

## Prerequisites

Start Docker Desktop, Colima, or another Docker runtime and verify:

```bash
docker info
docker compose version
```

Start Ollama on the host. The default test uses:

```bash
ollama pull qwen3.8:27b
ollama pull qwen3-embedding:0.6b
```

The generation and embedding models are intentionally separate. Qwen3.8 answers
questions; Qwen3 Embedding creates the document/query vectors used by Qdrant.

## Run the full test

Put one or more PDFs below `./data`, then run:

```bash
python3 scripts/ollama_pdf_rag_test.py \
  "文档中的关键技术指标是什么？"
```

The script performs all of these checks in order:

1. Docker daemon and Compose are available.
2. Ollama `/v1/models` contains both required models.
3. Ollama `/v1/embeddings` returns the expected 1024-dimensional vector.
4. A direct Ollama chat probe succeeds.
5. Ragbot starts with `docker-compose.yml` plus `docker-compose.ollama-host.yml`.
6. The worker can reach host Ollama and sees `RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS=/data`.
7. Every PDF below the selected data directory is mapped from the host path to `/data/...`.
8. PDFs are submitted with `reuse_source=false`, forcing fresh vectorization for this validation run.
9. Ragbot `/search` must return evidence with `semantic_embedding=true`.
10. Ragbot `/chat` must produce a non-empty Ollama answer.

A successful run ends with:

```text
Ollama PDF RAG smoke test: PASS
  PDFs=2, chunks=..., retrieved=5
  embedding=qwen3-embedding:0.6b/1024D, generation=qwen3.8:27b
```

## Pull missing models automatically

The script does not download a large model unexpectedly. To opt in:

```bash
python3 scripts/ollama_pdf_rag_test.py \
  --pull-missing \
  "根据PDF总结系统架构。"
```

## JSON report

```bash
python3 scripts/ollama_pdf_rag_test.py \
  --output tmp/ollama-pdf-rag-report.json \
  "文档中的关键技术指标是什么？"
```

The report includes model configuration, readiness, PDF/job/chunk counts,
retrieval diagnostics, top score, retrieved chunk IDs, answer, citations, and
confidence.

## Useful options

Use a different generation model:

```bash
python3 scripts/ollama_pdf_rag_test.py \
  --model qwen3.8:27b \
  "总结文档。"
```

Use a different embedding model/dimension only as a matched pair:

```bash
python3 scripts/ollama_pdf_rag_test.py \
  --embedding-model qwen3-embedding:0.6b \
  --embedding-dim 1024 \
  --collection rag_chunks_qwen3_embedding_0_6b_1024 \
  "总结文档。"
```

Stop containers after the test while preserving PostgreSQL/Qdrant volumes:

```bash
python3 scripts/ollama_pdf_rag_test.py --down-after "总结文档。"
```

Skip image rebuilds on repeated runs:

```bash
python3 scripts/ollama_pdf_rag_test.py --no-build "总结文档。"
```

## Why this script does not use host absolute PDF paths

Docker Compose mounts the selected host data directory read-only at `/data`.
Therefore the API/worker must receive:

```text
/data/manual.pdf
```

not:

```text
/Users/name/ragbot/data/manual.pdf
```

The script constructs container paths directly and verifies the worker contract.
It does not depend on a possibly stale `tmp/ragbot-runtime.json` to translate
paths.

## Why it forces fresh Source creation

Ragbot correctly reuses unchanged chunks during ordinary re-ingestion. That is
desirable in production, but it can hide an embedding-model change during a
smoke test. This validator submits each PDF with:

```json
{
  "reuse_source": false,
  "dedupe_active_job": false
}
```

so the selected PDFs are actually embedded on every validation run.

Existing Docker volumes are not deleted. The default test collection is:

```text
rag_chunks_qwen3_embedding_0_6b_1024
```

which separates this 1024-dimensional index from older 1536-dimensional or hash
collections.

## Troubleshooting

If Docker is unavailable:

```bash
docker info
```

If Ollama is unavailable:

```bash
curl http://127.0.0.1:11434/v1/models
```

If a model is missing:

```bash
ollama pull qwen3.8:27b
ollama pull qwen3-embedding:0.6b
```

If the script fails after Docker startup, inspect:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.ollama-host.yml \
  logs --tail=200 api worker
```

If the embedding probe reports a dimension mismatch, do not reuse a collection
with the wrong dimension. Pass the actual `--embedding-dim` together with a new
`--collection`, then rerun the smoke test.
