-- Durable connector resume state.
--
-- `checkpoint` is the last provider cursor/delta token known to correspond to a
-- published Ragbot generation. `pending_checkpoint` is written only after a
-- connector change stream is fully enumerated and is promoted by the repository
-- wrapper after generation activation succeeds. A crash may cause safe replay,
-- never acknowledgement of unpublished provider changes.

CREATE TABLE IF NOT EXISTS connector_checkpoints (
    source_id           TEXT PRIMARY KEY REFERENCES sources(source_id) ON DELETE CASCADE,
    tenant_id           TEXT NOT NULL,
    checkpoint          JSONB,
    pending_checkpoint  JSONB,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_connector_checkpoints_tenant
    ON connector_checkpoints(tenant_id, updated_at DESC);
