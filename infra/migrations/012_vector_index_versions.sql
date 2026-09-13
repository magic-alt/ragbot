-- First-class vector-index lifecycle for zero-downtime embedding/schema upgrades.
-- PostgreSQL records control-plane intent/history; the Qdrant alias is the
-- query-visible activation primitive and is reconciled after partial failures.

CREATE TABLE IF NOT EXISTS vector_index_versions (
    index_version_id TEXT PRIMARY KEY,
    tenant_id TEXT NULL,
    scope_key TEXT NOT NULL DEFAULT 'global',
    vector_backend TEXT NOT NULL DEFAULT 'qdrant',
    alias_name TEXT NOT NULL,
    physical_collection TEXT NOT NULL UNIQUE,
    embedding_contract_id TEXT NOT NULL,
    embedding_spec JSONB NOT NULL DEFAULT '{}'::jsonb,
    vector_schema JSONB NOT NULL DEFAULT '{}'::jsonb,
    parser_contracts JSONB NOT NULL DEFAULT '[]'::jsonb,
    chunking_contracts JSONB NOT NULL DEFAULT '[]'::jsonb,
    status TEXT NOT NULL,
    build_stats JSONB NOT NULL DEFAULT '{}'::jsonb,
    validation_evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    validating_at TIMESTAMPTZ NULL,
    ready_at TIMESTAMPTZ NULL,
    activated_at TIMESTAMPTZ NULL,
    retired_at TIMESTAMPTZ NULL,
    failed_at TIMESTAMPTZ NULL,
    delete_after TIMESTAMPTZ NULL,
    deleted_at TIMESTAMPTZ NULL,
    error TEXT NULL,
    CONSTRAINT vector_index_versions_status_check CHECK (
        status IN ('building', 'validating', 'ready', 'active', 'retired', 'failed', 'deleted')
    )
);

CREATE INDEX IF NOT EXISTS idx_vector_index_versions_alias_status
    ON vector_index_versions(alias_name, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_vector_index_versions_tenant_scope
    ON vector_index_versions(tenant_id, scope_key, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_vector_index_versions_retention
    ON vector_index_versions(status, delete_after)
    WHERE status IN ('retired', 'failed');

-- At most one PostgreSQL control-plane row may claim active for one logical
-- alias/scope. Qdrant alias state remains the query-visible authority and the
-- reconciler repairs this pointer after partial activation failures.
CREATE UNIQUE INDEX IF NOT EXISTS uq_vector_index_versions_active_alias
    ON vector_index_versions(alias_name, scope_key, COALESCE(tenant_id, ''))
    WHERE status = 'active';
