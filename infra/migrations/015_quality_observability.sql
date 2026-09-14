-- Durable quality / observability control-plane records for Issue #61.
--
-- These tables intentionally store identifiers, rankings, metrics, timings and
-- contract lineage rather than raw prompts/document content. Raw query text is
-- nullable and is populated only when an operator explicitly enables content
-- storage.

CREATE TABLE IF NOT EXISTS rag_runs (
    request_id               TEXT PRIMARY KEY,
    trace_id                 TEXT,
    tenant_id                TEXT NOT NULL,
    user_id                  TEXT NOT NULL,
    run_kind                 TEXT NOT NULL CHECK (run_kind IN ('search', 'chat', 'agent')),
    status                   TEXT NOT NULL CHECK (
        status IN ('completed', 'failed', 'deadline_exceeded', 'cancelled')
    ),
    query_hash               TEXT NOT NULL,
    query_text               TEXT,
    route                    TEXT,
    retrieval_plan           TEXT,
    retrieval_contract_id    TEXT,
    embedding_contract_id    TEXT,
    index_version_id         TEXT,
    reranker_contract_id     TEXT,
    model_contracts          JSONB NOT NULL DEFAULT '[]'::jsonb,
    stage_latency_ms         JSONB NOT NULL DEFAULT '{}'::jsonb,
    retrieved                JSONB NOT NULL DEFAULT '[]'::jsonb,
    citations                JSONB NOT NULL DEFAULT '[]'::jsonb,
    usage                    JSONB NOT NULL DEFAULT '{}'::jsonb,
    trace                    JSONB NOT NULL DEFAULT '{}'::jsonb,
    trace_sampled            BOOLEAN NOT NULL DEFAULT TRUE,
    total_duration_ms        INTEGER NOT NULL DEFAULT 0,
    error_code               TEXT,
    error_message            TEXT,
    started_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at               TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_rag_runs_tenant_created
    ON rag_runs(tenant_id, completed_at DESC);
CREATE INDEX IF NOT EXISTS idx_rag_runs_retrieval_plan
    ON rag_runs(retrieval_plan, completed_at DESC);
CREATE INDEX IF NOT EXISTS idx_rag_runs_index_version
    ON rag_runs(index_version_id, completed_at DESC)
    WHERE index_version_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_rag_runs_embedding_contract
    ON rag_runs(embedding_contract_id, completed_at DESC)
    WHERE embedding_contract_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_rag_runs_expires
    ON rag_runs(expires_at)
    WHERE expires_at IS NOT NULL;

-- Extend the pre-existing feedback surface without breaking older clients.
ALTER TABLE feedback ADD COLUMN IF NOT EXISTS citation_id TEXT;
ALTER TABLE feedback ADD COLUMN IF NOT EXISTS rating DOUBLE PRECISION;
ALTER TABLE feedback ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
CREATE INDEX IF NOT EXISTS idx_feedback_citation ON feedback(citation_id)
    WHERE citation_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS evaluation_runs (
    evaluation_run_id         TEXT PRIMARY KEY,
    evaluation_contract_id    TEXT NOT NULL,
    tenant_id                 TEXT,
    dataset_name              TEXT NOT NULL,
    dataset_version           TEXT NOT NULL,
    code_revision             TEXT NOT NULL,
    candidate_ref             TEXT,
    baseline_ref              TEXT,
    runtime_contracts         JSONB NOT NULL DEFAULT '{}'::jsonb,
    config                    JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics                   JSONB NOT NULL DEFAULT '{}'::jsonb,
    latency                   JSONB NOT NULL DEFAULT '{}'::jsonb,
    cost                      JSONB NOT NULL DEFAULT '{}'::jsonb,
    artifacts                 JSONB NOT NULL DEFAULT '{}'::jsonb,
    status                    TEXT NOT NULL DEFAULT 'completed' CHECK (
        status IN ('running', 'completed', 'failed')
    ),
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at              TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_evaluation_runs_contract
    ON evaluation_runs(evaluation_contract_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evaluation_runs_dataset
    ON evaluation_runs(dataset_name, dataset_version, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evaluation_runs_candidate
    ON evaluation_runs(candidate_ref, created_at DESC)
    WHERE candidate_ref IS NOT NULL;

CREATE TABLE IF NOT EXISTS promotion_decisions (
    promotion_decision_id     TEXT PRIMARY KEY,
    baseline_evaluation_id    TEXT NOT NULL REFERENCES evaluation_runs(evaluation_run_id),
    candidate_evaluation_id   TEXT NOT NULL REFERENCES evaluation_runs(evaluation_run_id),
    decision                  TEXT NOT NULL CHECK (decision IN ('accept', 'reject')),
    policy                    JSONB NOT NULL DEFAULT '{}'::jsonb,
    deltas                    JSONB NOT NULL DEFAULT '{}'::jsonb,
    reasons                   JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_promotion_candidate
    ON promotion_decisions(candidate_evaluation_id, created_at DESC);
