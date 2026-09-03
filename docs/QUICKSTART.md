# Ragbot Quickstart: from files to a queryable RAG database

This guide focuses on the shortest product path: bootstrap Ragbot, verify readiness, ingest knowledge, and query it. For the full Windows/Linux/macOS operations guide, see [`CLI_DEPLOYMENT.md`](CLI_DEPLOYMENT.md).

## 1. One-command bootstrap

Requires Python 3.10+:

```bash
python scripts/ragbot.py up --mode auto
```

`auto` chooses the full Docker Compose stack when Docker Compose and a healthy Docker daemon are available. Otherwise it falls back to the no-Docker Python development mode.

Explicit modes:

```bash
python scripts/ragbot.py up --mode local
python scripts/ragbot.py up --mode docker
```

The helper:

- creates `.env` from `.env.example` when needed;
- creates `data/`, `logs/`, `tmp/`, and `.venv/`;
- repairs a venv missing `pip` with `ensurepip`;
- installs Ragbot and the required extras;
- starts the service;
- waits for `/admin/ready`;
- records runtime mode/PID state for later commands.

Windows users can run the same Python command from PowerShell without activating the venv:

```powershell
python .\scripts\ragbot.py up --mode auto
```

Optional wrappers:

```powershell
.\scripts\ragbot.ps1 up --mode auto
```

```bash
bash scripts/ragbot.sh up --mode auto
```

## 2. Deployment modes

| Mode | Storage | Ingestion | Persistence | Best for |
| --- | --- | --- | --- | --- |
| `local` | InMemoryRepo + InMemoryQdrant | inline | API process lifetime | fast functional tests |
| `docker` | PostgreSQL + Qdrant | independent durable worker | Docker volumes | long-lived local knowledge bases |

Local mode deliberately removes `POSTGRES_DSN` and `QDRANT_URL`, forces development + inline ingestion, and constrains local sources to the repository `data/` directory.

## 3. Verify readiness

```bash
python scripts/ragbot.py status
python scripts/ragbot.py doctor
```

Expected doctor result:

```text
ragbot doctor: READY
```

Admin UI:

```text
http://127.0.0.1:8000/admin/ui
```

## 4. Configure semantic embeddings and an LLM

The first bootstrap creates `.env`. For real semantic retrieval, configure at least:

```dotenv
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=<your-key>
EMBEDDING_BASE_URL=https://api.openai.com
QDRANT_DIM=1536
```

For Agentic RAG answers:

```dotenv
RAGBOT_LLM_PROVIDER=openai
OPENAI_API_KEY=<your-key>
OPENAI_BASE_URL=https://api.openai.com
OPENAI_MODEL=gpt-4o-mini
```

Restart after editing `.env`:

```bash
python scripts/ragbot.py restart --mode local
```

or:

```bash
python scripts/ragbot.py restart --mode docker
```

When `EMBEDDING_API_KEY` is empty, development mode can use the deterministic HashEmbedder fallback. That is useful for pipeline smoke tests, not for Chinese or real semantic retrieval evaluation.

## 5. Build a RAG database from local files

Put local sources below `./data`:

```text
data/
├─ manuals/
│  ├─ architecture.md
│  └─ notes.txt
└─ pdf/
   └─ product_manual.pdf
```

The local filesystem connector currently scans text-like files (`.txt`, `.md`, `.markdown`, `.rst`, `.csv`, `.log`). PDFs use the separate PDF connector.

Text directory:

```bash
python scripts/ragbot.py ingest data/manuals \
  --tenant engineering \
  --name "Engineering manuals" \
  --tag manuals
```

PDF:

```bash
python scripts/ragbot.py ingest data/pdf/product_manual.pdf \
  --tenant engineering \
  --type pdf
```

The helper automatically translates host `./data/...` into container `/data/...` when the active deployment is Docker.

## 6. Remote PDF, Git, and Web

Remote PDF:

```bash
python scripts/ragbot.py ingest https://example.com/product/guide.pdf \
  --tenant engineering \
  --type pdf
```

Git:

```bash
python scripts/ragbot.py ingest https://github.com/magic-alt/ragbot \
  --tenant engineering \
  --type repo \
  --ref main \
  --tag code
```

Web:

```bash
python scripts/ragbot.py ingest https://example.com/knowledge-base/ \
  --tenant engineering \
  --type web
```

By default the helper waits for ingestion completion. Add `--no-wait` to return immediately after submission.

## 7. Manifest import

```bash
python scripts/ragbot.py import examples/ragbot-manifest.json \
  --tenant engineering
```

The HTTP batch API accepts up to 100 sources per request.

For Docker deployments, local paths inside a manifest must already use container-visible `/data/...` paths. The helper rewrites single-source `ingest` paths, but does not mutate manifest JSON.

## 8. Query the knowledge base

Pure retrieval:

```bash
python scripts/ragbot.py search \
  "How is the ingestion worker lease recovered?" \
  --tenant engineering \
  --top-k 5
```

Agentic answer:

```bash
python scripts/ragbot.py ask \
  "Summarize the ingestion architecture and cite the relevant sources" \
  --tenant engineering
```

## 9. Daily operations

```bash
python scripts/ragbot.py status
python scripts/ragbot.py doctor
python scripts/ragbot.py logs --lines 200
python scripts/ragbot.py logs -f
python scripts/ragbot.py restart --mode local
python scripts/ragbot.py down
```

Docker `down` keeps PostgreSQL/Qdrant volumes by default. To intentionally destroy Compose volumes:

```bash
python scripts/ragbot.py down --mode docker --volumes
```

## 10. Native `rag` CLI

The bootstrap helper does not replace the product CLI. It orchestrates installation/deployment and delegates knowledge operations to `cli.rag`.

After setup, native commands remain available:

```bash
rag --server http://localhost:8000 doctor

rag --server http://localhost:8000 \
  --tenant engineering \
  ingest /data/manuals \
  --wait

rag --server http://localhost:8000 \
  --tenant engineering \
  search "query" \
  --top-k 5

rag --server http://localhost:8000 \
  --tenant engineering \
  ask "question"
```

On Windows, without activating the venv:

```powershell
.\.venv\Scripts\python.exe -m cli.rag --server http://127.0.0.1:8000 doctor
```

## 11. Troubleshooting

### `No module named pip`

```bash
python scripts/ragbot.py setup --mode local --force-install
```

The helper checks `.venv` and runs `python -m ensurepip --upgrade` when pip is missing.

### `No module named uvicorn` / `No module named requests`

```bash
python scripts/ragbot.py setup --mode local --force-install
```

This reinstalls the editable Ragbot package and its dependencies into the repository `.venv`.

### PowerShell blocks `Activate.ps1`

Do not activate the environment. Run:

```powershell
python .\scripts\ragbot.py up --mode local
```

### Docker is unavailable

```bash
python scripts/ragbot.py up --mode local
```

No PostgreSQL, Qdrant server, Docker daemon, or separate worker is required for this development path.

## 12. Production note

The no-Docker local mode is intentionally a development/testing path. Production still requires durable worker mode, PostgreSQL, Qdrant, semantic embeddings, scoped API principals, networking controls, backup/restore, and real-provider staging gates.

See [`DEPLOYMENT.md`](DEPLOYMENT.md), [`CONFIGURATION.md`](CONFIGURATION.md), [`DISASTER_RECOVERY.md`](DISASTER_RECOVERY.md), and [`V1_RELEASE_READINESS.md`](V1_RELEASE_READINESS.md).
