# Recursive PDF corpus ingestion

Ragbot treats client-local PDFs as uploaded objects, not server filesystem paths. The host discovers PDFs below `./data`, streams each file to the API, and the API stores the bytes through the Ragbot-owned `UploadStore` before submitting the normal durable PDF ingestion workflow.

The resulting Source/Job contract contains a logical URI such as:

```text
ragbot-upload:///8f36f7b3eaf445bf9cf9d681f00a1320
```

It never contains a client path such as `/Users/name/project/data/manual.pdf` or `C:\project\data\manual.pdf`.

## Fastest path

Start Ragbot:

```bash
python3 scripts/ragbot.py up --mode auto
```

Put PDFs anywhere below the repository `data` directory:

```text
data/
├─ manuals/
│  ├─ motor-control.pdf
│  └─ ethercat/
│     └─ dc-guide.pdf
├─ papers/
│  └─ rag-survey.PDF
└─ notes.md
```

The canonical product command is:

```bash
python3 scripts/ragbot.py ingest data/ \
  --tenant engineering \
  --type pdf
```

The lower-level helper remains available:

```bash
python3 scripts/ingest_pdfs.py data --tenant engineering
```

## Execution model

```text
client ./data/manual.pdf
        │
        │ streamed PDF bytes
        ▼
POST /ingest/upload/pdf
        │
        ▼
UploadStore
        │
        ├─ logical object_id
        └─ SHA-256 content blob
        │
        ▼
ragbot-upload:///object_id
        │
        ▼
Source + durable Job
        │
        ▼
worker → Parser Port → Chunker → embedding
        │
        ▼
staged knowledge generation → activation
```

`object_id` and SHA-256 are deliberately separate. Two logical uploads containing identical bytes can remain distinct Sources while the filesystem adapter stores a single content-addressed blob where hard links are available.

## Local and Docker behavior

Both modes use the same HTTP upload contract.

### Local development

When `RAGBOT_UPLOAD_DIR` is unset, development mode derives a repository-local temporary upload root. Production mode requires an explicit upload store configuration.

### Docker Compose

The API and worker share the named `ingestion_uploads` volume at:

```text
/var/lib/ragbot/uploads
```

PostgreSQL remains private to the Compose network; server-managed uploads do not require PostgreSQL to be published to the host.

## Security boundaries

Ragbot intentionally separates two trust domains:

```text
ragbot-data:///...
    → RAGBOT_DATA_DIR
    → RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS

ragbot-upload:///...
    → UploadStore
    → server-managed object root
```

Uploaded objects do **not** require adding the upload directory to `RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS`. This prevents a server-managed object namespace from weakening the generic local-file allowlist.

The upload endpoint also:

- requires ingestion/operator capability and tenant authorization;
- enforces `RAGBOT_PDF_MAX_BYTES`;
- accepts PDF/octet-stream media types only;
- verifies the `%PDF-` signature before registration;
- stores only `Path(filename).name` as original metadata;
- computes SHA-256 while streaming rather than buffering the whole request in memory.

## Object lifecycle

Uploaded objects are registered in `uploaded_objects` with states:

```text
staged → active → retired → deleted
          │
          └──────── failure/orphan path → orphaned → deleted
```

Deleting an uploaded Source retires its object. Garbage collection deletes unreferenced retired/orphaned objects only after `RAGBOT_UPLOAD_RETENTION_SECONDS` (default 86400 seconds).

Tenant-scoped operational endpoints are available:

```text
GET  /ingest/uploads?tenant_id=<tenant>
POST /ingest/uploads/gc?tenant_id=<tenant>&retention_seconds=<seconds>
```

## Useful options

Preview discovery:

```bash
python3 scripts/ingest_pdfs.py data --dry-run
```

Non-recursive scan:

```bash
python3 scripts/ingest_pdfs.py data/manuals --no-recursive
```

Tags and chunking:

```bash
python3 scripts/ingest_pdfs.py data \
  --tenant engineering \
  --tag manuals \
  --tag pdf \
  --chunk-size 900 \
  --chunk-overlap 120
```

Limit a validation run:

```bash
python3 scripts/ingest_pdfs.py data --tenant engineering --max-files 20
```

Submit without waiting:

```bash
python3 scripts/ingest_pdfs.py data --tenant engineering --no-wait
```

Continue after a per-file failure:

```bash
python3 scripts/ingest_pdfs.py data --tenant engineering --continue-on-error
```

## Re-running the corpus

Each client upload creates a new logical uploaded object and PDF Source. Physical blob storage is content-addressed, so identical bytes can still be deduplicated by the UploadStore adapter.

This differs intentionally from `ragbot-data:///` Source reuse: an HTTP upload represents a new client-to-server object transfer and receives an immutable object identity. Incremental reuse still happens downstream where the ingestion pipeline can reuse compatible document/chunk representations.

## Mixed local corpus

Text-like files may still use the local directory connector:

```bash
python3 scripts/ragbot.py ingest data/ \
  --tenant engineering \
  --type local_fs
```

PDFs use server-managed upload:

```bash
python3 scripts/ragbot.py ingest data/ \
  --tenant engineering \
  --type pdf
```

This keeps local filesystem access and client-uploaded document bytes as separate security and lifecycle domains.
