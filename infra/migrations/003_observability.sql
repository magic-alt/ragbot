-- Migration 003: Observability tables (feedback, audit, metrics)
-- Applies to ragbot database

-- User feedback on agent responses
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id     TEXT PRIMARY KEY,
    request_id      TEXT NOT NULL,
    tenant_id       TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    feedback_type   TEXT NOT NULL CHECK (feedback_type IN ('positive', 'negative')),
    comment         TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_feedback_request ON feedback(request_id);
CREATE INDEX IF NOT EXISTS idx_feedback_tenant ON feedback(tenant_id);

-- Audit log for agent actions
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id        TEXT PRIMARY KEY,
    request_id      TEXT NOT NULL,
    tenant_id       TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    action          TEXT NOT NULL,   -- chat, search, ingest, admin
    route           TEXT,
    confidence      TEXT,
    tool_calls      JSONB,
    duration_ms     INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_request ON audit_log(request_id);
CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_log(tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at);

-- Request metrics for observability
CREATE TABLE IF NOT EXISTS request_metrics (
    metrics_id          TEXT PRIMARY KEY,
    request_id          TEXT NOT NULL,
    tenant_id           TEXT NOT NULL,
    user_id             TEXT NOT NULL,
    route               TEXT,
    total_duration_ms   INTEGER,
    citation_count      INTEGER DEFAULT 0,
    evidence_count      INTEGER DEFAULT 0,
    confidence          TEXT,
    tool_success_count  INTEGER DEFAULT 0,
    tool_failure_count  INTEGER DEFAULT 0,
    iterations          INTEGER DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_metrics_tenant ON request_metrics(tenant_id);
CREATE INDEX IF NOT EXISTS idx_metrics_created ON request_metrics(created_at);
