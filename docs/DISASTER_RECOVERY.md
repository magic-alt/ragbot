# Ragbot Disaster Recovery Runbook

This runbook defines the v1 recovery contract for Ragbot's two durable data stores: PostgreSQL and Qdrant. It is intentionally operational: the repository ships executable backup/restore scripts and CI performs a destructive seed → backup → delete → restore → verify smoke against PostgreSQL 16 and Qdrant v1.19.0.

## 1. Recovery scope

Ragbot's durable state is split across:

- **PostgreSQL**: Sources, ingestion Jobs, schedules, documents, chunks, FTS state, feedback/observability metadata and queue leases;
- **Qdrant**: vector points and payload indexes for the configured collection.

A consistent service recovery requires both stores. Restoring only PostgreSQL leaves lexical/catalog state without vectors; restoring only Qdrant leaves vectors without authoritative Source/Job/document metadata.

The scripts back up one configured Qdrant collection. If a deployment uses multiple active collections, run the Qdrant snapshot procedure for every collection and record all artifacts in the incident/change ticket.

## 2. Required tools and environment

The backup/restore host needs:

- `pg_dump` / `pg_restore` compatible with the PostgreSQL server major version;
- `curl`;
- `python`;
- network access to PostgreSQL and Qdrant;
- Qdrant API key when the service requires one.

Required environment:

```bash
export POSTGRES_DSN='postgresql://ragbot:***@postgres:5432/ragbot'
export QDRANT_URL='https://qdrant.internal'
export QDRANT_COLLECTION='rag_chunks'
export QDRANT_API_KEY='...optional...'
```

Never commit these values or backup archives to Git.

## 3. Backup

Run:

```bash
bash scripts/backup_ragbot.sh ./backups/ragbot-$(date -u +%Y%m%dT%H%M%SZ)
```

The backup directory contains:

```text
postgres.dump
qdrant.snapshot
manifest.json
```

`postgres.dump` is a PostgreSQL custom-format dump. `qdrant.snapshot` is a collection snapshot downloaded from Qdrant. `manifest.json` records creation time, collection name, file sizes and SHA-256 checksums.

### Production backup procedure

1. Confirm the target PostgreSQL database and Qdrant collection.
2. Record application image/tag, schema migration level, embedding model and `QDRANT_DIM`.
3. Prefer a low-write window for release/restore-point backups.
4. Run `backup_ragbot.sh`.
5. Copy the complete backup directory to encrypted durable object storage.
6. Verify retention/immutability according to the organization's policy.
7. Periodically test restore into an isolated environment; backup creation alone is not evidence of recoverability.

The v1 script does not create a distributed transaction across PostgreSQL and Qdrant. For strict point-in-time consistency, quiesce ingestion workers during the backup window or use infrastructure-level snapshots coordinated across both stores.

## 4. Restore

**Restore is destructive.** `pg_restore` runs with `--clean --if-exists`; Qdrant uploads the snapshot with `priority=snapshot`.

Before running restore:

1. stop or scale ingestion workers to zero;
2. stop scheduled sync or otherwise prevent new ingestion submissions;
3. drain/stop API write traffic;
4. verify the target database/collection and backup timestamp;
5. keep the pre-restore state available for rollback when possible.

Run:

```bash
bash scripts/restore_ragbot.sh ./backups/ragbot-20260902T030000Z
```

The script first verifies SHA-256 checksums from `manifest.json`, then restores PostgreSQL and uploads the Qdrant snapshot.

## 5. Post-restore validation

Do not reopen production traffic immediately after the restore command exits.

Required validation:

```bash
rag --server https://ragbot.example.com doctor
```

Then verify:

- `/admin/ready` is healthy;
- Source catalog counts look plausible;
- queue metrics have no unexpected stale leases;
- representative `/search` queries return expected citations;
- representative `/chat` requests can retrieve the restored knowledge;
- ACL negative tests still deny unauthorized principals;
- scheduled Sources have sensible next-run timestamps;
- worker logs show successful claim/heartbeat behavior after workers are re-enabled.

For a release rollback involving an embedding/index representation change, restore or point the application back to the matching Qdrant collection and embedding model/dimension together.

## 6. Queue state after restore

PostgreSQL contains durable queue state. A backup may therefore include `pending` or `running` Jobs.

After restore:

1. keep workers stopped;
2. inspect `/admin/queue/metrics`;
3. use `POST /admin/queue/reconcile` with an admin principal to repair expired leases/stranded failures;
4. inspect `dead_lettered` Jobs and their `failure_class`;
5. requeue only after the underlying cause is understood;
6. start workers gradually and watch oldest-pending age/provider throttling.

Dead-letter requeue defaults to the immutable connector snapshot captured by the failed Job. Operators may explicitly choose current Source config only when that is the intended recovery action.

## 7. Recovery objectives

Ragbot does not impose universal RPO/RTO numbers because they depend on deployment storage and workload. Before v1 production enablement, record:

- backup frequency and retention;
- target RPO;
- target RTO;
- expected PostgreSQL dump/restore duration at production size;
- expected Qdrant snapshot/upload duration at production size;
- who is authorized to restore;
- where encrypted backup artifacts live;
- how DNS/ingress traffic is held during recovery.

A GitHub-hosted CI smoke proves the mechanics on a small dataset; it does not prove production-size RTO.

## 8. CI recovery gate

The `Backup + restore smoke` CI job uses PostgreSQL 16 and Qdrant v1.19.0 and performs:

```text
migrations
  → seed PostgreSQL Source
  → seed Qdrant vector/payload
  → backup both stores
  → delete seeded state
  → restore both stores
  → verify PostgreSQL row and Qdrant payload
```

A release candidate should not be promoted when this gate is red.

## 9. Failure handling

If PostgreSQL restore fails, keep API/workers stopped, retain logs and validate dump/server-version compatibility before retrying.

If Qdrant restore fails, do not point query traffic at an empty or partially rebuilt collection. Keep the previous collection/snapshot available and validate collection dimension/config before retrying.

If only one store restores successfully, treat the deployment as unavailable until both durable stores are returned to a compatible state or the entire knowledge base is deliberately rebuilt from Sources.
