# Ragbot Quickstart: from files to a queryable RAG database

This guide focuses on the shortest product path: start Ragbot, verify that its dependencies are healthy, ingest one or many knowledge sources, wait until indexing is complete, and query the resulting knowledge base.

## 1. Start the service

```bash
cp .env.example .env
mkdir -p data
# Put local documents below ./data and configure LLM/embedding credentials in .env.
docker compose up -d --build
```

The default Compose topology starts:

- Ragbot API;
- independent durable ingestion worker;
- PostgreSQL and migrations;
- Qdrant.

Local sources below host `./data` are visible to the API and worker as `/data`.

## 2. Verify deployment readiness

After installing the package/CLI:

```bash
python -m pip install -e ".[postgres,qdrant,worker]"
rag --server http://localhost:8000 doctor
```

Expected result:

```text
ragbot doctor: READY
  liveness: {'status': 'ok'}
  readiness: {'status': 'ready', ...}
```

`doctor` checks process liveness and storage dependency readiness. It does not establish the semantic quality of an external model; production deployments should still run staging smoke with real providers.

## 3. Build a RAG database from one source

### Local directory

```bash
rag --server http://localhost:8000 \
  --tenant engineering \
  ingest /data/manuals \
  --name "Engineering manuals" \
  --tag manuals \
  --wait
```

### Remote PDF

```bash
rag --server http://localhost:8000 \
  --tenant engineering \
  ingest https://example.com/product/guide.pdf \
  --wait
```

### Git repository

```bash
rag --server http://localhost:8000 \
  --tenant engineering \
  ingest https://github.com/magic-alt/ragbot \
  --ref main \
  --tag code \
  --wait
```

### Web page

```bash
rag --server http://localhost:8000 \
  --tenant engineering \
  ingest https://example.com/knowledge-base/ \
  --wait
```

When `--type` is omitted, the CLI and Quick Import API infer `local_fs`, `pdf`, `repo`, or `web` from the location. Use `--type` when a location is ambiguous.

`--wait` polls the durable ingestion Job until it is `completed` or `failed` and prints final document/chunk counts. Without `--wait`, submission returns immediately.

## 4. Build a knowledge base from a manifest

Start with [`examples/ragbot-manifest.json`](../examples/ragbot-manifest.json):

```json
{
  "tenant_id": "engineering",
  "sources": [
    {"location": "/data/manuals", "tags": ["manuals"]},
    {"location": "https://example.com/product/guide.pdf", "tags": ["product"]},
    {
      "location": "https://github.com/magic-alt/ragbot",
      "config": {"ref": "main"},
      "tags": ["code"]
    }
  ]
}
```

Submit and wait for all Jobs:

```bash
rag --server http://localhost:8000 import examples/ragbot-manifest.json --wait
```

The batch API accepts up to 100 sources per request. Each item returns its own Source/Job result, so one submission error does not hide the state of the other items.

## 5. Query the knowledge base

Pure retrieval:

```bash
rag --server http://localhost:8000 \
  --tenant engineering \
  search "How is the ingestion worker lease recovered?" \
  --top-k 5
```

Agentic answer:

```bash
rag --server http://localhost:8000 \
  --tenant engineering \
  ask "Summarize the ingestion architecture and cite the relevant sources"
```

## 6. Quick Import API

For applications that do not want to orchestrate `POST /sources` followed by `POST /ingest/jobs`, Ragbot exposes a high-level product endpoint.

```bash
curl -X POST http://localhost:8000/ingest/quick \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant_id": "engineering",
    "location": "/data/manuals",
    "name": "Engineering manuals",
    "tags": ["manuals"]
  }'
```

Representative response:

```json
{
  "status": "accepted",
  "source_id": "...",
  "source_type": "local_fs",
  "source_reused": false,
  "job_id": "...",
  "job_status": "pending",
  "job_reused": false
}
```

Batch import:

```bash
curl -X POST http://localhost:8000/ingest/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant_id": "engineering",
    "sources": [
      {"location": "/data/manuals"},
      {"location": "https://example.com/guide.pdf"}
    ]
  }'
```

## 7. Source reuse, configuration safety, and idempotency

Quick Import uses a stable Source identity derived from:

```text
tenant + source type + normalized location
```

By default:

1. a matching existing Source is reused;
2. supplied Source metadata/config is synchronized when it is safe to submit a new run;
3. if a `pending` or `running` Job already exists with the same connector configuration, that active Job is returned instead of creating an obvious duplicate;
4. if an active Job exists but the requested connector configuration differs (for example `ref=main` versus `ref=release`), the request returns `409` instead of silently treating the old Job as the new request.

Each durable Job stores the connector `source_type` and `source_config` captured at submission. The worker executes that snapshot even if the Source record is edited while the Job waits in the queue. This makes retries and delayed execution deterministic for connector path/ref/chunking settings.

### Strict idempotency

For deployment automation or concurrent callers, provide an explicit key:

```bash
rag --server http://localhost:8000 \
  ingest /data/manuals \
  --idempotency-key nightly-2026-09-02
```

Submitting the same stable Source + idempotency key returns the exact same Job, including after completion. This is the recommended mechanism when callers need strict repeat-request behavior across multiple API replicas.

The ordinary active-Job check is an ergonomic duplicate guard, not a distributed uniqueness primitive. Without an explicit idempotency key, two truly concurrent requests reaching different API replicas may still both enqueue before either observes the other.

`idempotency_key` requires Source reuse; combining it with `--no-reuse-source` is rejected because a newly generated Source ID would make strict replay impossible.

Advanced overrides:

```text
--no-reuse-source   create a distinct Source record
--force-new-job     bypass the active-Job convenience dedupe
```

Use these deliberately; the default behavior is designed for repeatable product bootstrap.

## 8. Production authentication

When production API-key principals are enabled, add the API key to CLI operations:

```bash
rag --server https://rag.example.com \
  --api-key "$RAGBOT_API_KEY" \
  --tenant tenant-a \
  doctor
```

The tenant requested by ingest/search/chat must be inside the API-key principal's allowed tenant scope.

## 9. Production checklist

Quick Import shortens data onboarding; it does not remove production release requirements. Before exposing Ragbot as a shared service, verify at minimum:

- `RAGBOT_ENV=production` and durable worker mode;
- real semantic embedding and LLM providers;
- PostgreSQL/Qdrant persistence and backup/restore;
- API-key principal mappings;
- source path mounts and Web/PDF/Git egress allowlists;
- TLS/Ingress/rate limiting;
- the repository's real-provider staging smoke workflow.

See [`DEPLOYMENT.md`](DEPLOYMENT.md), [`CONFIGURATION.md`](CONFIGURATION.md), and [`V1_RELEASE_READINESS.md`](V1_RELEASE_READINESS.md) for the complete production gates.
