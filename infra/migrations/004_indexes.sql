-- Migration 004: Performance indexes + schema alignment for Milestone E
-- Align schema columns before creating indexes that depend on those columns.

-- Add columns that may be missing from earlier migrations.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS checksum TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS qdrant_point_id TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;

-- Chunk checksum index for deduplication lookups.
CREATE INDEX IF NOT EXISTS idx_chunks_checksum
    ON chunks(checksum) WHERE checksum IS NOT NULL;

-- Document URI index for source-based deletion.
CREATE INDEX IF NOT EXISTS idx_documents_uri ON documents(uri);

-- Document source_type index for filtered queries.
CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type);

-- Chunk doc_id index for listing chunks by document.
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

-- Chunk tenant_id index for tenant-scoped queries.
CREATE INDEX IF NOT EXISTS idx_chunks_tenant_id ON chunks(tenant_id);

-- Source tenant_id index.
CREATE INDEX IF NOT EXISTS idx_sources_tenant_id ON sources(tenant_id);

-- Ingestion job tenant_id + source_id composite index.
CREATE INDEX IF NOT EXISTS idx_jobs_tenant_source
    ON ingestion_jobs(tenant_id, source_id);
