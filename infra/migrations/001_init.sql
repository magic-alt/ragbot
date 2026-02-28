-- ragbot schema initialization
-- Run this migration to set up the core tables.

-- documents: ingested document metadata
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    title           TEXT,
    uri             TEXT,
    version         TEXT NOT NULL DEFAULT '1',
    doc_updated_at  TIMESTAMPTZ,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tags            TEXT[] DEFAULT '{}',
    acl_policy_id   TEXT,
    metadata        JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents(tenant_id);
CREATE INDEX IF NOT EXISTS idx_documents_tags ON documents USING GIN(tags);

-- chunks: document chunks with full-text search
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    tenant_id       TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    path            TEXT,
    url             TEXT,
    page            INTEGER,
    section         TEXT,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON chunks(tenant_id);
-- GIN index for full-text search on chunk text
CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING GIN(to_tsvector('simple', text));

-- acl_policies: access control policies
CREATE TABLE IF NOT EXISTS acl_policies (
    acl_policy_id   TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL,
    rules           JSONB NOT NULL DEFAULT '{}',
    policy_hash     TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_acl_tenant ON acl_policies(tenant_id);
CREATE INDEX IF NOT EXISTS idx_acl_hash ON acl_policies(policy_hash);

-- ingestion_jobs: track ingestion pipeline runs
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id          TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    source_config   JSONB NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed
    doc_count       INTEGER DEFAULT 0,
    chunk_count     INTEGER DEFAULT 0,
    error           TEXT,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jobs_tenant ON ingestion_jobs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON ingestion_jobs(status);
