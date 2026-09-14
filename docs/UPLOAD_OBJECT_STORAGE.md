# Object-storage UploadStore

Issue #59 removes the requirement that API and worker replicas share the same upload filesystem. PostgreSQL remains the authoritative metadata/reference-count control plane; S3/MinIO stores the uploaded bytes.

## Contract

Client-local files enter Ragbot through the managed upload API and become stable references of the form:

```text
ragbot-upload:///<32-hex-object-id>
```

A Source stores that opaque URI, not an API-node filesystem path and not object-store credentials.

`UploadStore` separates durable object location from parser-local materialization:

- `temporary_path(object_id)` — bounded API-side staging path while receiving an upload;
- `commit_pdf(...)` — verify and make the logical object durable;
- `materialize_path(uri)` — return a verified disposable local parser copy;
- `delete_object(...)` — remove the logical object from the physical backend;
- `local_path(uri)` — compatibility facade; callers should prefer `materialize_path`.

## S3 / MinIO layout

The adapter uses two physical namespaces:

```text
<prefix>/blobs/<sha256>.pdf
<prefix>/objects/<object_id>.json
```

`blobs/` is content-addressed. Uploading identical bytes for multiple Sources uploads the blob once. Each logical upload has its own pointer containing object id, SHA-256, byte size, media type and blob key.

This deliberately separates logical identity from physical deduplication: two users may upload identical bytes and later delete one logical object without breaking the other.

## Cross-node ingestion

With object storage authoritative:

```text
API replica A
  receive upload
  verify SHA/size
  commit S3 blob + logical pointer
       |
       v
PostgreSQL UploadedObject metadata
       |
       v
Source config stores ragbot-upload URI
       |
       v
Worker replica N
  resolve URI
  read pointer
  stream blob to node-local cache
  enforce max bytes
  verify exact size + SHA-256
  atomic rename
  parse PDF
```

The worker never needs the API node's local filesystem. The materialized file may be deleted at any time and re-created from the authoritative object store.

## Integrity and resource bounds

Materialization rejects:

- logical pointer/object-id mismatch;
- missing pointer fields;
- pointer files larger than 64 KiB;
- objects larger than `RAGBOT_UPLOAD_MAX_OBJECT_BYTES`;
- downloads exceeding the expected byte count;
- final byte-count mismatch;
- SHA-256 mismatch.

Downloads stream in bounded chunks instead of loading the entire PDF in memory. Uploads use boto3 transfer configuration with bounded multipart chunk size and concurrency.

## Credentials

Use the standard AWS SDK credential chain. In production prefer workload identity / IAM role / Kubernetes secret injection. Do not persist access keys in `Source.config` or `UploadedObject` metadata.

For MinIO or another S3-compatible implementation set `RAGBOT_UPLOAD_S3_ENDPOINT_URL`. AWS S3 normally leaves this unset.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAGBOT_UPLOAD_STORE` | `filesystem` | `filesystem`, `s3`, or `minio` |
| `RAGBOT_UPLOAD_S3_BUCKET` | required | authoritative object bucket |
| `RAGBOT_UPLOAD_S3_PREFIX` | `ragbot/uploads` | object namespace prefix |
| `RAGBOT_UPLOAD_S3_ENDPOINT_URL` | empty | MinIO/S3-compatible endpoint |
| `RAGBOT_UPLOAD_S3_REGION` | empty | S3 region |
| `RAGBOT_UPLOAD_MATERIALIZE_DIR` | temp dir | node-local disposable parser cache |
| `RAGBOT_UPLOAD_MAX_OBJECT_BYTES` | 100 MiB | materialization hard limit |
| `RAGBOT_UPLOAD_MULTIPART_THRESHOLD_BYTES` | 8 MiB | boto3 multipart threshold |
| `RAGBOT_UPLOAD_MULTIPART_CHUNK_BYTES` | 8 MiB | multipart part size |
| `RAGBOT_UPLOAD_MAX_CONCURRENCY` | 4 | transfer concurrency bound |

See `.env.upload-s3.example` and the Compose overrides.

## Docker Compose

Root stack:

```bash
docker compose \
  --env-file .env \
  --env-file .env.upload-s3.example \
  -f docker-compose.yml \
  -f docker-compose.upload-s3.yml \
  up -d
```

The existing filesystem upload volume may still appear in the base stack, but it is not authoritative when `RAGBOT_UPLOAD_STORE=s3|minio`.

## Kubernetes / Helm

The chart already supports arbitrary non-secret `.Values.env` entries and secret injection. For object storage:

- disable `uploadStorage.enabled` unless another feature needs the shared PVC;
- set the `RAGBOT_UPLOAD_*` non-secret values in `.Values.env`;
- inject AWS credentials through workload identity or the deployment's secret mechanism;
- ensure API and worker receive the same bucket/prefix/endpoint settings.

No Source should contain bucket credentials.

## Metadata lifecycle and GC

Existing PostgreSQL `UploadedObject` state and `ref_count` remain the authority for lifecycle decisions. The S3 adapter does **not** maintain a second reference count.

Logical deletion removes:

- `objects/<object_id>.json`;
- the node-local materialized cache copy.

It intentionally does not immediately delete `blobs/<sha256>.pdf`, because another logical object may reference the same content. A future/extended metadata-aware blob GC may delete a blob only after PostgreSQL proves no live logical UploadedObject references that SHA.

This favors safe temporary leakage over deleting shared user data.

## Operational checks

For a horizontally scaled deployment verify:

1. upload on API replica A;
2. ingest on a worker that has never seen A's filesystem;
3. SHA/size validation succeeds;
4. re-ingestion can reuse/re-materialize the same logical object;
5. two logical objects with identical bytes share one blob;
6. deleting one logical object leaves the other ingestible;
7. orphan/retention cleanup follows PostgreSQL reference-count state.

## Phase-2 boundary

The S3/MinIO adapter establishes the horizontally-scaled port. Remaining Issue #59 work can include:

- real MinIO/AWS integration tests in CI/staging;
- direct-to-object-store presigned/multipart client uploads for very large files;
- Azure Blob and GCS adapters behind the same port;
- metadata-aware content-blob GC;
- production throughput/retry metrics and upload SLOs.
