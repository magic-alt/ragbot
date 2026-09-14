-- PostgreSQL operational-scale indexes for bounded catalog/query/queue paths.
-- These complement, rather than replace, the earlier retrieval and scheduling
-- indexes. Keyset pagination always includes a deterministic ID tiebreaker.

CREATE INDEX IF NOT EXISTS idx_sources_tenant_created_keyset
    ON sources (tenant_id, created_at DESC, source_id DESC)
    WHERE status <> 'deleted';

CREATE INDEX IF NOT EXISTS idx_sources_created_keyset
    ON sources (created_at DESC, source_id DESC)
    WHERE status <> 'deleted';

CREATE INDEX IF NOT EXISTS idx_documents_tenant_ingested_keyset
    ON documents (tenant_id, ingested_at DESC, doc_id DESC)
    WHERE status <> 'deleted';

CREATE INDEX IF NOT EXISTS idx_documents_source_ingested_keyset
    ON documents (source_id, ingested_at DESC, doc_id DESC)
    WHERE source_id IS NOT NULL AND status <> 'deleted';

CREATE INDEX IF NOT EXISTS idx_jobs_tenant_created_keyset
    ON ingestion_jobs (tenant_id, created_at DESC, job_id DESC);

CREATE INDEX IF NOT EXISTS idx_jobs_source_created_keyset
    ON ingestion_jobs (source_id, created_at DESC, job_id DESC);

-- Keep the claim scan compact and ordered without indexing terminal rows.
CREATE INDEX IF NOT EXISTS idx_jobs_pending_claim
    ON ingestion_jobs (available_at, created_at, job_id)
    WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_jobs_active_source
    ON ingestion_jobs (tenant_id, source_id, created_at DESC, job_id DESC)
    WHERE status IN ('pending', 'running');

CREATE INDEX IF NOT EXISTS idx_generations_tenant_created_keyset
    ON knowledge_generations (tenant_id, created_at DESC, generation_id DESC);

CREATE INDEX IF NOT EXISTS idx_generations_source_created_keyset
    ON knowledge_generations (source_id, created_at DESC, generation_id DESC);

-- Latest completed job per source is used by the bounded control-plane source
-- catalog and overview knowledge-size aggregation.
CREATE INDEX IF NOT EXISTS idx_jobs_completed_source_latest
    ON ingestion_jobs (source_id, created_at DESC, job_id DESC)
    WHERE status = 'completed';

-- Recent failure panels should never sort the complete queue history.
CREATE INDEX IF NOT EXISTS idx_jobs_failure_recent
    ON ingestion_jobs (created_at DESC, job_id DESC)
    WHERE status IN ('failed', 'dead_lettered');
