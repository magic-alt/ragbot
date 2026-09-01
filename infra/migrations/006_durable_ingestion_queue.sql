-- Durable ingestion queue lease metadata.
-- Jobs are claimed with SELECT ... FOR UPDATE SKIP LOCKED by independent workers.

ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS attempts INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS available_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS lease_owner TEXT,
    ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_queue
    ON ingestion_jobs (status, available_at, created_at);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_lease
    ON ingestion_jobs (status, lease_expires_at)
    WHERE status = 'running';

-- CJK lexical support uses application-generated bigram lexemes while keeping
-- PostgreSQL's built-in 'simple' text search configuration.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS fts_text TEXT NOT NULL DEFAULT '';

UPDATE chunks
SET fts_text = text
WHERE fts_text = '';

CREATE INDEX IF NOT EXISTS idx_chunks_fts_text
    ON chunks USING GIN (to_tsvector('simple', fts_text));
