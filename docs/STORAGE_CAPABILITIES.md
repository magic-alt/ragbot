# Ragbot Storage Capability Boundaries

Ragbot keeps PostgreSQL as the reference durable system of record, but callers should not depend on one all-purpose repository interface when they only need a subset of persistence behavior.

## Capability ports

`services.api.app.storage.protocol` defines focused runtime-checkable protocols:

- `KnowledgeCatalogRepo` — documents and chunks;
- `ACLRepo` — ACL policies;
- `SourceRepo` — source catalog/lifecycle;
- `JobQueueRepo` — durable ingestion queue/lease/retry surface;
- `IngestionRepo` — composite required by the ingestion data plane;
- `ControlPlaneRepo` — source/job/ACL control-plane composite;
- `GenerationRepo` — staged knowledge generation activation;
- `PublicationOutboxRepo` — post-activation cleanup events;
- `DebugStateRepo` — health/export diagnostics;
- `DevelopmentTableRepo` — test/local table helper only.

`Repo` remains as a backwards-compatible composite while callers are gradually narrowed to the capabilities they actually consume.

## PostgreSQL ownership

The authoritative PostgreSQL implementation now lives in `storage/pg_repo.py`. The previous three-layer inheritance chain has been removed:

```text
before:
postgres_repo.PostgresRepo
        ↓
pg_repo.PostgresRepo
        ↓
ManagedPostgresRepo

after:
pg_repo.PostgresRepo     <- authoritative migration-aligned base
        ↓
ManagedPostgresRepo      <- scheduling / production DLQ control-plane extension

postgres_repo.PostgresRepo -> compatibility re-export only
```

`ManagedPostgresRepo` remains the production runtime implementation. A later change may move its scheduling/DLQ operations into composed focused stores, but no second SQL implementation should be reintroduced.

## Transaction boundaries

- individual catalog/source/job CRUD uses the repository connection context;
- job claim/lease state transitions use an explicit transaction with `FOR UPDATE SKIP LOCKED`;
- staged knowledge activation remains owned by the existing generation publication transaction;
- Qdrant remains a separate derived-index boundary and is never treated as part of a PostgreSQL distributed transaction.

## Extension rule

A new persistence adapter should implement only the capability it truthfully supports. Product code should detect optional behavior through explicit protocols/capabilities rather than concrete backend class names.

PostgreSQL remains the production reference backend; this interface split is not a requirement to add arbitrary databases.
