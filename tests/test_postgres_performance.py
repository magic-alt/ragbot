from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest

from services.api.app.storage.generation_support import ensure_generation_repository
from services.api.app.storage.managed_pg_repo import ManagedPostgresRepo
from services.api.app.storage.models import Chunk, Document, IngestionJob, KnowledgeGeneration, Source
from services.api.app.storage.postgres_performance import ensure_database_performance


pytestmark = pytest.mark.skipif(
    not os.getenv("POSTGRES_TEST_DSN"), reason="POSTGRES_TEST_DSN not configured"
)


def _repo() -> ManagedPostgresRepo:
    repo = ManagedPostgresRepo(os.environ["POSTGRES_TEST_DSN"], pool_min=1, pool_max=4)
    ensure_generation_repository(repo)
    ensure_database_performance(repo)
    repo._ragbot_pg_copy_min_rows = 1
    return repo


def _source(suffix: str) -> Source:
    now = datetime.now(timezone.utc).isoformat()
    return Source(
        source_id=f"perf-source-{suffix}",
        tenant_id=f"perf-tenant-{suffix}",
        source_type="local_fs",
        name="PostgreSQL performance",
        config={"path": "/tmp/perf"},
        created_at=now,
        updated_at=now,
    )


def test_claim_hot_path_does_not_reconcile_every_poll(monkeypatch) -> None:
    repo = _repo()
    try:
        suffix = uuid.uuid4().hex[:12]
        source = _source(suffix)
        repo.add_source(source)
        job = IngestionJob(
            job_id=f"perf-job-{suffix}",
            tenant_id=source.tenant_id,
            source_id=source.source_id,
            source_type=source.source_type,
            source_config=dict(source.config),
        )
        repo.add_job(job)

        def forbidden(*args, **kwargs):
            raise AssertionError("reconcile_ingestion_jobs must not run inside claim_next_job")

        monkeypatch.setattr(repo, "reconcile_ingestion_jobs", forbidden)
        claimed = repo.claim_next_job("perf-worker", lease_seconds=30, max_attempts=3)
        assert claimed is not None
        assert claimed.job_id == job.job_id
        assert claimed.status == "running"

        metrics = repo.database_runtime_metrics()["hot_path"]
        assert metrics["claim_attempts"] == 1
        assert metrics["claim_hits"] == 1
    finally:
        repo.close()


def test_copy_chunk_upsert_and_bounded_query_round_trip() -> None:
    repo = _repo()
    try:
        suffix = uuid.uuid4().hex[:12]
        source = _source(suffix)
        repo.add_source(source)
        now = datetime.now(timezone.utc).isoformat()
        doc = Document(
            doc_id=f"perf-doc-{suffix}",
            tenant_id=source.tenant_id,
            source_type=source.source_type,
            title="perf",
            uri=f"source://{source.source_id}",
            version="1",
            doc_updated_at=now,
            ingested_at=now,
            source_id=source.source_id,
        )
        repo.add_document(doc)
        chunks = [
            Chunk(
                chunk_id=f"perf-chunk-{suffix}-{index}",
                doc_id=doc.doc_id,
                tenant_id=source.tenant_id,
                chunk_index=index,
                text=f"servo ethercat performance {index}",
                checksum=f"checksum-{suffix}-{index}",
                source_id=source.source_id,
                metadata={"source_type": source.source_type},
            )
            for index in range(4)
        ]
        assert repo.add_chunks(chunks) == 4
        assert all(repo.get_chunk(chunk.chunk_id) is not None for chunk in chunks)

        first = repo.page_documents(
            tenant_ids={source.tenant_id}, source_id=source.source_id, limit=1
        )
        assert first.total == 1
        assert first.items[0].doc_id == doc.doc_id
        source_page = repo.page_sources(tenant_ids={source.tenant_id}, limit=1)
        assert source_page.total == 1
        assert source_page.items[0].source_id == source.source_id
        assert repo.documents_for_source(source.source_id, source.tenant_id)[0].doc_id == doc.doc_id

        metrics = repo.database_runtime_metrics()["hot_path"]
        assert metrics["copy_chunk_batches"] >= 1
        assert metrics["copy_chunk_rows"] >= 4
    finally:
        repo.close()


def test_generation_staging_uses_copy_path() -> None:
    repo = _repo()
    try:
        suffix = uuid.uuid4().hex[:12]
        source = _source(suffix)
        repo.add_source(source)
        generation = KnowledgeGeneration(
            generation_id=f"perf-gen-{suffix}",
            source_id=source.source_id,
            tenant_id=source.tenant_id,
        )
        repo.begin_knowledge_generation(generation)
        now = datetime.now(timezone.utc).isoformat()
        document = Document(
            doc_id=f"perf-stage-doc-{suffix}",
            tenant_id=source.tenant_id,
            source_type=source.source_type,
            title="staged",
            uri=f"source://{source.source_id}/staged",
            version="1",
            doc_updated_at=now,
            ingested_at=now,
            source_id=source.source_id,
            generation_id=generation.generation_id,
        )
        chunk = Chunk(
            chunk_id=f"perf-stage-chunk-{suffix}",
            doc_id=document.doc_id,
            tenant_id=source.tenant_id,
            chunk_index=0,
            text="staged generation COPY path",
            checksum=f"stage-{suffix}",
            source_id=source.source_id,
            generation_id=generation.generation_id,
            metadata={"source_type": source.source_type},
        )
        stats = repo.stage_knowledge_generation(
            generation.generation_id, [document], [chunk]
        )
        assert stats == {"documents": 1, "chunks": 1}
        with repo._pool.connection() as conn:
            staged_docs = conn.execute(
                "SELECT COUNT(*) AS n FROM staged_documents WHERE generation_id = %s",
                (generation.generation_id,),
            ).fetchone()["n"]
            staged_chunks = conn.execute(
                "SELECT COUNT(*) AS n FROM staged_chunks WHERE generation_id = %s",
                (generation.generation_id,),
            ).fetchone()["n"]
        assert staged_docs == 1
        assert staged_chunks == 1
        metrics = repo.database_runtime_metrics()["hot_path"]
        assert metrics["copy_generation_batches"] >= 1
        assert metrics["copy_generation_rows"] >= 2
    finally:
        repo.close()
