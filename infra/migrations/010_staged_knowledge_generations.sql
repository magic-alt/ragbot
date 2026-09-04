-- Migration 010: staged knowledge generations and durable publication outbox.
--
-- PostgreSQL remains the authoritative visibility boundary. Workers prepare a
-- candidate generation in staged_* tables and Qdrant, then activate it by one
-- PostgreSQL transaction that replaces the active document/chunk manifest,
-- swaps the Source active-generation pointer and records cleanup work.

ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_id TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS generation_id TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS source_id TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS generation_id TEXT;

CREATE TABLE IF NOT EXISTS knowledge_generations (
    generation_id       TEXT PRIMARY KEY,
    source_id           TEXT NOT NULL,
    tenant_id           TEXT NOT NULL,
    job_id              TEXT,
    status              TEXT NOT NULL DEFAULT 'staging',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    prepared_at         TIMESTAMPTZ,
    activated_at        TIMESTAMPTZ,
    retired_at          TIMESTAMPTZ,
    failed_at           TIMESTAMPTZ,
    error               TEXT,
    stats               JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT knowledge_generation_status_check
        CHECK (status IN ('staging', 'prepared', 'active', 'retired', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_knowledge_generations_source
    ON knowledge_generations(source_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_generations_status
    ON knowledge_generations(status, created_at);

CREATE TABLE IF NOT EXISTS source_active_generations (
    source_id           TEXT PRIMARY KEY,
    generation_id       TEXT NOT NULL REFERENCES knowledge_generations(generation_id),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS staged_documents (
    generation_id       TEXT NOT NULL REFERENCES knowledge_generations(generation_id) ON DELETE CASCADE,
    source_id           TEXT NOT NULL,
    doc_id              TEXT NOT NULL,
    tenant_id           TEXT NOT NULL,
    source_type         TEXT NOT NULL,
    title               TEXT,
    uri                 TEXT,
    version             TEXT NOT NULL DEFAULT '1',
    doc_updated_at      TIMESTAMPTZ,
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tags                TEXT[] DEFAULT '{}',
    acl_policy_id       TEXT,
    status              TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (generation_id, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_staged_documents_source
    ON staged_documents(source_id, generation_id);

CREATE TABLE IF NOT EXISTS staged_chunks (
    generation_id       TEXT NOT NULL REFERENCES knowledge_generations(generation_id) ON DELETE CASCADE,
    source_id           TEXT NOT NULL,
    chunk_id            TEXT NOT NULL,
    doc_id              TEXT NOT NULL,
    tenant_id           TEXT NOT NULL,
    chunk_index         INTEGER NOT NULL,
    text                TEXT NOT NULL,
    path                TEXT,
    url                 TEXT,
    page                INTEGER,
    section             TEXT,
    checksum            TEXT,
    qdrant_point_id     TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    fts_text            TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (generation_id, chunk_id),
    FOREIGN KEY (generation_id, doc_id)
        REFERENCES staged_documents(generation_id, doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_staged_chunks_generation
    ON staged_chunks(generation_id, doc_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_staged_chunks_point
    ON staged_chunks(qdrant_point_id) WHERE qdrant_point_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS publication_outbox (
    outbox_id           BIGSERIAL PRIMARY KEY,
    event_type          TEXT NOT NULL,
    source_id           TEXT NOT NULL,
    generation_id       TEXT,
    payload             JSONB NOT NULL DEFAULT '{}'::jsonb,
    status              TEXT NOT NULL DEFAULT 'pending',
    attempts            INTEGER NOT NULL DEFAULT 0,
    available_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    lease_owner         TEXT,
    lease_expires_at    TIMESTAMPTZ,
    last_error          TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    CONSTRAINT publication_outbox_status_check
        CHECK (status IN ('pending', 'running', 'completed', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_publication_outbox_ready
    ON publication_outbox(status, available_at, outbox_id);
CREATE INDEX IF NOT EXISTS idx_publication_outbox_source
    ON publication_outbox(source_id, created_at DESC);

-- Bootstrap existing active knowledge into deterministic legacy generations.
-- Ragbot document IDs are source-owned by either config.doc_id or doc-<source_id>
-- and multi-document sources append a ':' suffix.
INSERT INTO knowledge_generations (
    generation_id, source_id, tenant_id, status, created_at, prepared_at, activated_at, stats
)
SELECT
    'legacy:' || s.source_id,
    s.source_id,
    s.tenant_id,
    'active',
    COALESCE(s.created_at, NOW()),
    NOW(),
    NOW(),
    jsonb_build_object('bootstrap', true)
FROM sources AS s
WHERE EXISTS (
    SELECT 1
    FROM documents AS d
    WHERE d.doc_id = COALESCE(NULLIF(s.config->>'doc_id', ''), 'doc-' || s.source_id)
       OR d.doc_id LIKE COALESCE(NULLIF(s.config->>'doc_id', ''), 'doc-' || s.source_id) || ':%'
)
ON CONFLICT (generation_id) DO NOTHING;

INSERT INTO source_active_generations(source_id, generation_id, updated_at)
SELECT source_id, generation_id, NOW()
FROM knowledge_generations
WHERE generation_id LIKE 'legacy:%' AND status = 'active'
ON CONFLICT (source_id) DO NOTHING;

UPDATE documents AS d
SET source_id = s.source_id,
    generation_id = 'legacy:' || s.source_id
FROM sources AS s
WHERE d.source_id IS NULL
  AND (
      d.doc_id = COALESCE(NULLIF(s.config->>'doc_id', ''), 'doc-' || s.source_id)
      OR d.doc_id LIKE COALESCE(NULLIF(s.config->>'doc_id', ''), 'doc-' || s.source_id) || ':%'
  );

UPDATE chunks AS c
SET source_id = d.source_id,
    generation_id = d.generation_id
FROM documents AS d
WHERE c.doc_id = d.doc_id
  AND (c.source_id IS NULL OR c.generation_id IS NULL);

CREATE INDEX IF NOT EXISTS idx_documents_source_generation
    ON documents(source_id, generation_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source_generation
    ON chunks(source_id, generation_id);
CREATE INDEX IF NOT EXISTS idx_chunks_qdrant_point
    ON chunks(qdrant_point_id) WHERE qdrant_point_id IS NOT NULL;
