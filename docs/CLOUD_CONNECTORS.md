# Cloud and SaaS connectors

Ragbot supports Google Drive, Notion and Confluence as scheduled, multi-document knowledge Sources. These connectors use the same durable PostgreSQL ingestion queue, Qdrant indexing path, ACL metadata and replacement lifecycle as local/PDF/Web/Git/S3 sources.

## Security model

Do not store credential values in `Source.config`.

Cloud Sources store only a reference:

```json
{
  "credential_ref": "env:RAGBOT_NOTION_TOKEN"
}
```

The API validates the reference syntax, but only the ingestion worker resolves the environment variable. This permits the API and worker to have different secret exposure. `file:` references are intentionally unsupported so a Source cannot request arbitrary local credential files.

Ragbot rejects common inline secret fields such as `access_token`, `refresh_token`, `api_key`, `password`, `private_key` and `client_secret` for SaaS Sources.

For Kubernetes, inject the referenced environment variable into the worker from a Secret or external secret provider. Do not put the secret value in Helm values checked into source control.

## Incremental synchronization model

The SaaS connectors are metadata-first rather than full-download polling:

1. the scheduled Source is enqueued through the normal durable scheduler;
2. the connector lists remote document/page metadata and calculates a remote version;
3. if a remote version matches the previous Ragbot snapshot, the existing chunks are reused and the body is not downloaded or re-embedded;
4. new/changed documents are downloaded, chunked, embedded and upserted;
5. documents no longer returned by the remote source are removed by the normal replacement-pruning phase;
6. the new complete knowledge view becomes the latest successful ingestion.

This preserves Ragbot's replacement-oriented retry semantics while avoiding repeated content downloads and embedding calls for unchanged documents.

## Google Drive

Source type: `gdrive`

Locations accepted by Quick Import include:

```text
gdrive://<folder-id>
https://drive.google.com/drive/folders/<folder-id>
```

Minimal configuration using a pre-issued access token:

```bash
export RAGBOT_DRIVE_TOKEN='...'

rag --server http://localhost:8000 \
  --tenant engineering \
  ingest gdrive://1AbCdEfFolder \
  --credential-ref env:RAGBOT_DRIVE_TOKEN \
  --wait
```

For long-lived production deployments, prefer Google JSON credentials so the worker can refresh credentials:

```bash
export RAGBOT_DRIVE_CREDENTIALS_JSON='{"type":"service_account", ...}'

rag --server http://localhost:8000 \
  --tenant engineering \
  ingest gdrive://1AbCdEfFolder \
  --credential-ref env:RAGBOT_DRIVE_CREDENTIALS_JSON \
  --credential-type google_json \
  --wait
```

`google_json` supports Google service-account information or authorized-user credential JSON through `google-auth`. Grant only Drive read-only scope and share only the required folder/content with that identity.

Supported content includes text/code/config files and PDFs. Native Google Docs export as text, Sheets as CSV, and Slides/Drawings as PDF before text extraction.

Important options:

- `recursive`: default `true`; traverse nested Drive folders;
- `max_file_bytes`: default 20 MiB per file;
- `chunk_size` / `chunk_overlap`.

Remote version uses Drive `modifiedTime`, `version` and `md5Checksum` metadata.

## Notion

Source type: `notion`

Locations:

```text
notion://<page-id>
https://www.notion.so/<page-title>-<page-id>
```

Create an internal integration, grant it read access only to the pages that should enter the knowledge base, then expose its token only to the worker:

```bash
export RAGBOT_NOTION_TOKEN='secret_...'

rag --server http://localhost:8000 \
  --tenant engineering \
  ingest notion://0123456789abcdef0123456789abcdef \
  --credential-ref env:RAGBOT_NOTION_TOKEN \
  --wait
```

By default Ragbot traverses child pages. Page `last_edited_time` is the remote version. Unchanged pages reuse existing chunks. In recursive mode the block tree still has to be inspected to discover child pages; unchanged page text is nevertheless not re-embedded.

Use `--no-recursive` when a single Notion page is the desired Source. In that mode an unchanged page needs only the page metadata request and can skip its block-content request entirely.

## Confluence

Source type: `confluence`

For Atlassian Cloud, Quick Import accepts:

```text
confluence://acme.atlassian.net/ENG
https://acme.atlassian.net/wiki/spaces/ENG/overview
```

Basic API-token authentication:

```bash
export RAGBOT_CONFLUENCE_TOKEN='...'
export RAGBOT_CONFLUENCE_ALLOWED_HOSTS='acme.atlassian.net'

rag --server http://localhost:8000 \
  --tenant engineering \
  ingest https://acme.atlassian.net/wiki/spaces/ENG/overview \
  --credential-ref env:RAGBOT_CONFLUENCE_TOKEN \
  --email ragbot@example.com \
  --auth-type basic \
  --wait
```

Bearer authentication is also supported:

```bash
rag ... --auth-type bearer --credential-ref env:RAGBOT_CONFLUENCE_TOKEN
```

For Data Center or self-hosted Confluence, supply `base_url` via `--base-url` or manifest config. In production the host must be explicitly listed in `RAGBOT_CONFLUENCE_ALLOWED_HOSTS`. If the installation resolves to RFC1918/private addresses, `RAGBOT_ALLOW_PRIVATE_SOURCE_NETWORKS=true` is additionally required. This is deliberately explicit to avoid turning a tenant-controlled Source into a general server-side network pivot.

Ragbot lists page metadata for the space first. Only pages with a changed Confluence version/last-updated value are fetched with `body.storage`. HTML storage is normalized to text before chunking. `root_page_id` can restrict ingestion to a subtree when the list result includes ancestor metadata.

## Manifest examples

```json
{
  "tenant_id": "engineering",
  "sources": [
    {
      "location": "gdrive://1AbCdEfFolder",
      "name": "Drive engineering docs",
      "config": {
        "credential_ref": "env:RAGBOT_DRIVE_CREDENTIALS_JSON",
        "credential_type": "google_json"
      }
    },
    {
      "location": "notion://0123456789abcdef0123456789abcdef",
      "name": "Notion runbooks",
      "config": {
        "credential_ref": "env:RAGBOT_NOTION_TOKEN"
      }
    },
    {
      "location": "confluence://acme.atlassian.net/ENG",
      "name": "Confluence ENG",
      "config": {
        "credential_ref": "env:RAGBOT_CONFLUENCE_TOKEN",
        "email": "ragbot@example.com",
        "auth_type": "basic"
      }
    }
  ]
}
```

Then:

```bash
rag --server http://localhost:8000 import cloud-sources.json --wait
```

## Scheduled synchronization

After initial ingestion, use the existing Source scheduler:

```bash
curl -X PUT http://localhost:8000/sources/<source-id>/sync \
  -H 'X-API-Key: ...' \
  -H 'Content-Type: application/json' \
  -d '{"enabled":true,"interval_seconds":3600,"run_immediately":false}'
```

The scheduled Job snapshot stores the non-secret `credential_ref`, not the secret value. Worker replicas resolve the current secret when they execute the Job. This permits credential rotation without rewriting Source configuration.

## Operational recommendations

- give each connector a dedicated least-privilege service identity/integration;
- inject credentials only into workers that need them;
- use different environment-variable names for different tenants/integrations when isolation requires it;
- start with hourly or multi-hour synchronization rather than minute-level polling unless remote rate limits are understood;
- monitor `/admin/queue/metrics` and failed Job details;
- configure worker KEDA backlog scaling only after the remote provider's request-rate limits are accounted for;
- use staging credentials and the optional SaaS smoke workflow before enabling a new production connector.

## Current limitations

- Google Drive does not yet use the Drive Changes API; it performs metadata enumeration of the configured folder tree, then downloads only changed documents. This is still incremental for content/embedding cost but not O(changes) metadata discovery.
- Notion recursive discovery reads block lists to find child pages because the public API does not expose an inexpensive complete descendant index for an arbitrary root page.
- Confluence currently uses the REST v1 content listing surface for broad Cloud/Data Center compatibility.
- Provider-native ACLs are not automatically translated to Ragbot ACL policies. The Source's Ragbot ACL policy remains authoritative for retrieval access.
