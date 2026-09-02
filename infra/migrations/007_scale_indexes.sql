-- Scale-oriented lexical retrieval optimization.
-- Persist the tsvector once so searches do not rebuild it for every candidate row.

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS fts_document tsvector
    GENERATED ALWAYS AS (
        to_tsvector('simple', COALESCE(NULLIF(fts_text, ''), text))
    ) STORED;

DROP INDEX IF EXISTS idx_chunks_fts_text;

CREATE INDEX IF NOT EXISTS idx_chunks_fts_document
    ON chunks USING GIN (fts_document);

CREATE INDEX IF NOT EXISTS idx_chunks_tenant_doc
    ON chunks (tenant_id, doc_id);

CREATE INDEX IF NOT EXISTS idx_chunks_tenant_created
    ON chunks (tenant_id, created_at DESC);
