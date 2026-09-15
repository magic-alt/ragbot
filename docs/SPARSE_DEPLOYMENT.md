# Sparse retrieval deployment

Sparse retrieval is intentionally opt-in because FastEmbed pulls an ONNX runtime stack that is not required by dense-only Ragbot deployments.

## Docker Compose

Use the normal stack plus the sparse overlay:

```bash
docker compose --env-file .env.sparse.example \
  -f docker-compose.yml \
  -f docker-compose.sparse.yml \
  up -d --build
```

The infra layout has the equivalent overlay:

```bash
docker compose --env-file .env.sparse.example \
  -f infra/docker/docker-compose.yml \
  -f infra/docker/docker-compose.sparse.yml \
  up -d --build
```

The overlay does two things that must stay coupled:

1. builds every Ragbot service image with `RAGBOT_INSTALL_SPARSE=true`, which installs FastEmbed;
2. injects the same `RAGBOT_SPARSE_*` representation contract into API and worker replicas.

The migration service receives the same build argument only so all services that tag `ragbot:local` resolve to the same image contents; it does not run the sparse encoder.

## Sparse contract

The production defaults are:

```text
RAGBOT_SPARSE_ENABLED=true
RAGBOT_SPARSE_PROVIDER=fastembed
RAGBOT_SPARSE_MODEL=Qdrant/bm25
RAGBOT_SPARSE_VECTOR_NAME=sparse
RAGBOT_SPARSE_MODIFIER=idf
RAGBOT_SPARSE_BATCH_SIZE=32
```

`revision`, `tokenizer`, and `language` also participate in `SparseEmbeddingSpec` when set. API and worker values must be identical. An active dense+sparse `IndexVersion` is immutable with respect to that sparse contract; a replica with a missing or mismatched contract must fail rather than silently writing only dense vectors.

## Candidate workflow

Building the image does not activate sparse retrieval. The safe sequence remains:

```text
sparse-capable image
  -> build candidate IndexVersion
  -> Golden Dataset EvaluationRun
  -> PromotionDecision
  -> mark-ready only on accepted evidence
  -> explicit alias activation
  -> rollback if required
```

Before `mark-ready`, the Golden Dataset must also pass the promotion evidence-integrity rules in `docs/PROMOTION_EVIDENCE_INTEGRITY.md`; an unscoped heuristic relevance report is diagnostic only and cannot authorize activation.

The default `docker-compose.yml` continues to build without FastEmbed unless the sparse overlay/build argument is selected.
