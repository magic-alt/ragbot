ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS failure_class TEXT;

ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS dead_lettered_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_dead_letter
    ON ingestion_jobs (tenant_id, dead_lettered_at DESC)
    WHERE status = 'dead_lettered';

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_reconcile
    ON ingestion_jobs (status, lease_expires_at, attempts)
    WHERE status IN ('pending', 'running', 'failed', 'dead_lettered');
