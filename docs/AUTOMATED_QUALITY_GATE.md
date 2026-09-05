# Automated Ragbot quality gate

Ragbot has several focused tests and benchmarks. `scripts/verify_ragbot.py` is the release-style entry point that combines them into one repeatable decision:

```bash
python scripts/verify_ragbot.py --profile standard --pytest on
```

The command exits `0` only when the enabled gates pass and writes JSON/Markdown evidence under `reports/quality-gate/`.

## What it validates

The live portion is a black-box test of the running deployment. It does **not** insert chunks directly into PostgreSQL or Qdrant. Instead it generates a deterministic five-page PDF in memory and sends it through the public server-managed upload API:

```text
PDF upload
  -> Source + durable ingestion Job
  -> worker claim / parse
  -> chunking
  -> embedding
  -> Qdrant vector write + PostgreSQL metadata/FTS
  -> generation publication
  -> vector / lexical / hybrid retrieval
  -> optional reranker
  -> Agentic /chat synthesis
  -> citation checks
```

After ingestion completes, the gate discovers the new document ID from a unique marker and scopes every quality probe to that document. Existing user documents therefore cannot make the synthetic score look better.

The default corpus covers exact lexical retrieval, semantic paraphrase retrieval, fieldbus/safety concepts, and a Chinese query over English evidence. The report records the active embedding model/backend/dimension, semantic-vs-HashEmbedder status, retrieval diagnostics, Hit@1/Hit@5, semantic Hit@5, MRR@10, p50/p95 latency, answer correctness and citation presence.

When `--pytest on` is enabled, the same entry point also executes the repository test suite. That suite is where connector contracts, queue/retry/DLQ behavior, ACL/RBAC/multi-tenant behavior, generation fencing, SQL/code routes, control-plane behavior and other implementation-level invariants are validated. External SaaS connectors still require their dedicated credentialed staging tests for a true provider-side live check.

## Acceptance profiles

These are Ragbot engineering regression gates, not universal RAG benchmarks.

| Profile | Intended use | Semantic embedding | Vector semantic Hit@5 | Hybrid Hit@5 | Hybrid MRR@10 | Search p95 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `smoke` | pipeline/debug | optional | observe | >= 0.70 | >= 0.45 | <= 3000 ms |
| `standard` | normal local/release acceptance | required | >= 0.75 | >= 0.85 | >= 0.65 | <= 2000 ms |
| `strict` | tuned regression target | required | >= 0.90 | >= 0.92 | >= 0.75 | <= 1200 ms |

All profiles also require successful ingestion, zero retrieval request errors and a 100% deterministic answer/citation smoke pass on the bundled corpus.

`HashEmbedder` can pass a pipeline smoke but **cannot** pass the `standard` or `strict` profile. This prevents a working HTTP/RAG pipeline from being mistaken for production semantic retrieval.

## Recommended local command

For a locally running Docker or local-process deployment:

```bash
python scripts/verify_ragbot.py \
  --profile standard \
  --pytest on \
  --tenant default \
  --repetitions 2
```

PowerShell:

```powershell
python .\scripts\verify_ragbot.py `
  --profile standard `
  --pytest on `
  --tenant default `
  --repetitions 2
```

If scoped API principals are enabled, use a tenant that the key can operate and pass the key with `RAGBOT_API_KEY` or `--api-key`.

### Functional-test Python environment

The quality gate must be runnable from a machine where Ragbot was deployed but development test dependencies were never installed. The policies are therefore:

- `--pytest auto` (default): run the functional suite only when an existing Python environment already has pytest; never install packages.
- `--pytest on`: release-grade behavior. Prefer the repository `.venv`; if pytest is unavailable, create/repair `.venv` and install `.[dev,postgres,qdrant,worker,s3,saas,observability]`, matching the functional dependency surface used by GitHub CI. The system/Homebrew Python is not modified.
- `--pytest off`: skip the repository functional suite and run only the live/domain RAG gates.

This means the recommended `--pytest on` command is self-contained after deployment: a missing local pytest installation no longer causes a false overall failure before the RAG quality result is considered.

## Measure your real document corpus

The bundled corpus answers a different question from a domain Golden Dataset:

- bundled system gate: **is the deployed RAG machinery healthy and semantically capable?**
- domain Golden Dataset: **does this embedding/chunking/retrieval configuration work well on my actual documents and questions?**

Pass one or more existing `scripts/rag_eval.py` datasets to the unified command:

```bash
python scripts/verify_ragbot.py \
  --profile standard \
  --pytest on \
  --dataset eval/datasets/my_golden.json
```

Each domain dataset is executed against the live API with its own configured thresholds. A failed Golden Dataset makes the overall command fail.

For a mature corpus, label 50-100+ representative questions and progressively replace keyword relevance with expected pages or exact chunk IDs. The existing `docs/RAG_EVALUATION.md` schema supports Hit@K, MRR@10, Recall@10, latency and answer/citation checks.

## Reading failures

A useful diagnosis order is:

1. **upload/job failure**: inspect worker logs, parser/storage configuration and the Job error;
2. **`semantic_embedding=false`**: configure a real embedding endpoint and re-index; do not tune RRF against HashEmbedder;
3. **lexical passes, vector fails**: embedding model/dimension/index contract or semantic model quality problem;
4. **vector passes, hybrid degrades**: candidate-pool/fusion weights/reranker calibration problem;
5. **retrieval passes, answer fails**: synthesis/model/prompt/citation grounding problem;
6. **only p95 fails**: measure embedding, Qdrant, PostgreSQL FTS and reranker latency separately before changing ranking quality.

The report retains per-request diagnostics so the failed stage can be separated from the final score.

## Related evaluation surfaces

- `scripts/rag_eval.py`: live Golden Dataset evaluator for real corpora;
- `scripts/search_test.py`: interactive retrieval debugging;
- `eval/ci_gate.py`: in-process agent evaluation gate;
- `eval/staging_smoke.py`: credentialed/staging source smoke;
- retrieval/framework/parser benchmarks under `benchmarks/` and `scripts/` for focused A/B work.

The automated system gate is intentionally an orchestrator rather than a replacement for those more specialized tools.
