-- Durable connector resume state.
--
-- Checkpoints are control-plane metadata. They are advanced only after the
-- candidate Source generation is successfully published so a worker crash
-- cannot acknowledge provider changes that never became visible in Ragbot.

CREATE TABLE IF NOT EXISTS connector_checkpoints (
    source_id       TEXT PRIMARY KEY REFERENCES sources(source_id) ON DELETE CASCADE,
    tenant_id       TEXT NOT NULL,
    checkpoint      JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_connector_checkpoints_tenant
    ON connector_checkpoints(tenant_id, updated_at DESC);
