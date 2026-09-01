-- Migration 005: align persisted ingestion-job schema with IngestionJob model.
-- PostgresRepo has stored/loaded job.stats since Milestone E, but the column was
-- missing from migrations 001-004.

ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS stats JSONB NOT NULL DEFAULT '{}'::jsonb;
