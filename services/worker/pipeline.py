"""Ingestion pipeline orchestrator.

Coordinates the full ingest lifecycle:
  Source config → IngestionJob → connector (fetch) → chunker (ingest_*) → embed_and_upsert → update job status.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from services.api.app.retrieval.embedder import Embedder
from services.api.app.storage.models import Chunk, Document, IngestionJob, Source
from services.api.app.storage.repo import InMemoryRepo
from services.worker.dedup.hashing import content_hash
from services.worker.dedup.versioning import next_version
from services.worker.jobs.embed_and_upsert import embed_and_upsert

logger = logging.getLogger(__name__)


def run_ingest_pipeline(
    source: Source,
    repo: InMemoryRepo,
    qdrant: object,
    job_id: Optional[str] = None,
    embedder: Optional[Embedder] = None,
) -> IngestionJob:
    """Execute the full ingestion pipeline for a source.

    The caller-provided embedder is the same instance used by retrieval.  This
    prevents documents from being indexed in a different vector space from
    query embeddings.  The optional default is retained for direct unit-test
    and backwards-compatible callers.
    """
    now = datetime.now(timezone.utc).isoformat()
    job_id = job_id or uuid.uuid4().hex
    job = IngestionJob(
        job_id=job_id,
        tenant_id=source.tenant_id,
        source_id=source.source_id,
        source_type=source.source_type,
        source_config=source.config,
        status="running",
        started_at=now,
        created_at=now,
    )
    repo.add_job(job)

    try:
        chunks = _run_connector(source, repo)
        chunk_list = list(chunks)

        # Dedup: skip chunks whose checksum already exists
        chunk_list = _dedup_chunks(chunk_list, repo)

        # Create or update document record before persisting chunks.
        doc = _ensure_document(source, repo, chunk_count=len(chunk_list))

        # Embed and upsert to vector store using the query-time embedder.
        if chunk_list:
            embed_and_upsert(repo, qdrant, chunk_list, embedder=embedder)

        repo.update_job(
            job_id,
            status="completed",
            doc_count=1,
            chunk_count=len(chunk_list),
            completed_at=datetime.now(timezone.utc).isoformat(),
            stats={"doc_id": doc.doc_id, "chunks_ingested": len(chunk_list)},
        )
        logger.info("Pipeline completed: job=%s source=%s chunks=%d", job_id, source.source_id, len(chunk_list))

    except Exception as exc:
        logger.exception("Pipeline failed: job=%s source=%s", job_id, source.source_id)
        repo.update_job(
            job_id,
            status="failed",
            error=str(exc),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    return repo.get_job(job_id)


def _run_connector(source: Source, repo: InMemoryRepo) -> Iterable[Chunk]:
    """Select and run the appropriate connector based on source_type."""
    source_type = source.source_type
    config = source.config
    tenant_id = source.tenant_id
    # Keep chunk/document identity deterministic and identical.  Previously a
    # random chunk doc_id was generated while _ensure_document() used
    # ``doc-{source_id}``, creating orphaned metadata (and FK failures in PG).
    doc_id = config.get("doc_id") or f"doc-{source.source_id}"
    version = config.get("version", "1.0")
    tags = source.tags
    acl_hash = _resolve_acl_hash(source, repo)

    common = dict(doc_id=doc_id, tenant_id=tenant_id, version=version, tags=tags, acl_hash=acl_hash)

    if source_type == "pdf":
        from services.worker.jobs.ingest_pdf import ingest_pdf
        return ingest_pdf(path=config["path"], **common)

    elif source_type == "web":
        from services.worker.jobs.ingest_web import ingest_web
        return ingest_web(url=config["url"], **common)

    elif source_type == "repo":
        from services.worker.jobs.ingest_repo import ingest_repo
        return ingest_repo(
            url_or_path=config["path"],
            ref=config.get("ref"),
            **common,
        )

    elif source_type == "local_fs":
        from services.worker.jobs.ingest_text import ingest_local_fs
        return ingest_local_fs(
            directory=config["path"],
            extensions=config.get("extensions"),
            **common,
        )

    elif source_type == "database":
        # Database sources don't ingest chunks; they enable SQL tool access.
        # Return empty to allow doc record creation.
        return []

    else:
        raise ValueError(f"Unsupported source_type: {source_type}")


def _resolve_acl_hash(source: Source, repo: InMemoryRepo) -> str:
    if source.acl_policy_id:
        h = repo.get_policy_hash(source.acl_policy_id)
        if h:
            return h
    return "public"


def _dedup_chunks(chunks: list[Chunk], repo: InMemoryRepo) -> list[Chunk]:
    """Remove chunks whose checksum already exists in the repo."""
    existing_checksums = set()
    for c in repo.iter_chunks():
        if c.checksum:
            existing_checksums.add(c.checksum)

    deduped = []
    for chunk in chunks:
        if chunk.checksum and chunk.checksum in existing_checksums:
            logger.debug("Skipping duplicate chunk: %s", chunk.chunk_id)
            continue
        existing_checksums.add(chunk.checksum)
        deduped.append(chunk)
    return deduped


def _ensure_document(source: Source, repo: InMemoryRepo, chunk_count: int) -> Document:
    """Create or update the Document record for this source."""
    now = datetime.now(timezone.utc).isoformat()
    doc_id = source.config.get("doc_id") or f"doc-{source.source_id}"
    existing = repo.get_document(doc_id)

    if existing:
        existing.version = next_version(existing.version)
        existing.ingested_at = now
        repo.add_document(existing)
        return existing

    doc = Document(
        doc_id=doc_id,
        tenant_id=source.tenant_id,
        source_type=source.source_type,
        title=source.name,
        uri=f"source://{source.source_id}",
        version=source.config.get("version", "1.0"),
        doc_updated_at=now,
        ingested_at=now,
        tags=source.tags,
        acl_policy_id=source.acl_policy_id,
    )
    repo.add_document(doc)
    return doc
