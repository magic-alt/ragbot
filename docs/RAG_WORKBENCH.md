# RAG Workbench: single-PDF ingestion and retrieval testing

Ragbot's built-in `/admin/ui` is a zero-build control plane served directly by FastAPI. This guide focuses on the shortest useful loop for one PDF:

```text
PDF -> ingest -> inspect chunks -> run retrieval -> diagnose vector/lexical fusion -> then test RAG answers
```

## 1. Start Ragbot

```powershell
python .\scripts\ragbot.py up --mode local
```

Open:

```text
http://127.0.0.1:8000/admin/ui
```

The UI now has three workspaces:

- **Overview** — Sources, Documents, Chunks, queue health, failed jobs, retry/requeue and source actions.
- **PDF Ingestion** — dedicated single-file form with chunk size/overlap and live job polling.
- **Retrieval Playground** — query the index and inspect vector rank, lexical rank, RRF score, page, section and embedding runtime.

## 2. Ingest one PDF

Place the PDF below the repository `data` directory. In local mode, an example path is:

```text
D:\Project\ragbot\data\DeepSeek in Action.pdf
```

In **PDF Ingestion**:

1. select the tenant;
2. enter the PDF path;
3. keep the first baseline at `chunk_size=800`, `chunk_overlap=100`;
4. click **Ingest PDF**;
5. wait for the job to reach `completed`;
6. record `documents`, `chunks total`, `written`, `reused` and duration.

The browser path field is not shell-parsed, so spaces in filenames do not require quoting.

## 3. Test retrieval before RAG answer generation

Use **Retrieval Playground** or the explainable CLI helper:

```powershell
python .\scripts\search_test.py `
  "What is the difference between LoRA and QLoRA?" `
  --tenant engineering `
  --source-type pdf `
  --top-k 10
```

The output separates:

- final fused rank/score;
- vector rank and raw vector score;
- lexical rank and lexical score;
- RRF score;
- optional reranker score;
- page/section/chunk ID;
- embedding backend/model/dimension.

This lets you distinguish `bad_retrieval` from a later answer-synthesis problem.

## 4. Why local HashEmbedder can look bad

If no semantic embedding is configured, development mode uses `HashEmbedder`.

The workbench shows a warning because this backend is deterministic infrastructure scaffolding, not a semantic model. The development retriever now:

- removes common English question stopwords from the in-memory lexical fallback;
- gives lexical matches higher weight than hash-vector matches;
- disables the invalid hash-vector branch for CJK queries instead of returning arbitrary zero-vector ties;
- exposes this as `fusion_mode=lexical-first-development`.

For real semantic and cross-lingual evaluation, configure a real embedding model and **re-ingest the corpus**. Example:

```dotenv
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=<your-key>
EMBEDDING_BASE_URL=https://api.openai.com
QDRANT_DIM=1536
```

Changing the embedding model/dimension is an index-contract change. Do not judge a new query model against vectors created by a different embedding configuration.

## 5. Recommended smoke queries

For an English technical book, start with three classes of query:

```text
What is the difference between LoRA and QLoRA?
How can quantization reduce GPU memory usage during LLM deployment?
大模型部署时如何减少显存占用？
```

Interpretation:

- If the first two fail with a real semantic model, inspect vector/lexical rank traces and chunk quality.
- If English works but the Chinese query fails, verify the embedding model's multilingual capability.
- If retrieval is correct but the final answer is wrong, move the investigation to synthesis/citation evaluation.

## 6. Query-file regression testing

Create a UTF-8 file with one query per line:

```text
# eval/deepseek-smoke.txt
What is the difference between LoRA and QLoRA?
How can quantization reduce GPU memory usage during LLM deployment?
大模型部署时如何减少显存占用？
```

Run:

```powershell
python .\scripts\search_test.py `
  --query-file .\eval\deepseek-smoke.txt `
  --tenant engineering `
  --source-type pdf `
  --top-k 10
```

Once relevant chunk IDs are known, promote these smoke queries into Ragbot's existing `eval/` golden datasets and measure MRR@10 / Recall@10 rather than relying on subjective inspection.
