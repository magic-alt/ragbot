-- Product control-plane scheduling state for recurring source ingestion.
ALTER TABLE sources
    ADD COLUMN IF NOT EXISTS sync_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS sync_interval_seconds INTEGER,
    ADD COLUMN IF NOT EXISTS sync_next_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS sync_last_enqueued_at TIMESTAMPTZ;

ALTER TABLE sources DROP CONSTRAINT IF EXISTS chk_sources_sync_interval;
ALTER TABLE sources
    ADD CONSTRAINT chk_sources_sync_interval
    CHECK (sync_interval_seconds IS NULL OR sync_interval_seconds >= 60);

CREATE INDEX IF NOT EXISTS idx_sources_sync_due
    ON sources(sync_next_at)
    WHERE sync_enabled = TRUE AND status = 'active';
