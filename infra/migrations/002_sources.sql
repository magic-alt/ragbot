-- Milestone B schema additions
-- Sources table + ingestion_jobs.source_id column

-- sources: data source configuration
CREATE TABLE IF NOT EXISTS sources (
    source_id       TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    name            TEXT NOT NULL,
    config          JSONB NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'active',  -- active, paused, deleted
    acl_policy_id   TEXT,
    tags            TEXT[] DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sources_tenant ON sources(tenant_id);
CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(status);
CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(source_type);

-- Add source_id to ingestion_jobs (nullable for backward compat with existing rows)
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS source_id TEXT;
CREATE INDEX IF NOT EXISTS idx_jobs_source ON ingestion_jobs(source_id);
