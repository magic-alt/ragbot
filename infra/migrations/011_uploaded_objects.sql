CREATE TABLE IF NOT EXISTS uploaded_objects (
    object_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    storage_backend TEXT NOT NULL,
    storage_key TEXT NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    media_type TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'staged'
        CHECK (state IN ('staged', 'active', 'orphaned', 'retired', 'deleted')),
    ref_count INTEGER NOT NULL DEFAULT 0 CHECK (ref_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_referenced_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_uploaded_objects_tenant_state_created
    ON uploaded_objects (tenant_id, state, created_at);

CREATE INDEX IF NOT EXISTS idx_uploaded_objects_sha256_state
    ON uploaded_objects (sha256, state);
