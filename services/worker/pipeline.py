"""Ingestion pipeline orchestrator.

Coordinates the full ingest lifecycle:
  Source config → IngestionJob → connector (fetch) → chunker (ingest_*) → embed_and_upsert → update job status.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from services.api.app.retrieval.embedder import Embedder
from services.api.app.storage.models import Chunk, Document, IngestionJob, Source
from services.api.app.storage.repo import InMemoryRepo
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

    The caller-provided embedder is the same instance used by retrieval. This
    prevents documents from being indexed in a different vector space from
    query embeddings. The optional default is retained for direct unit-test
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

        # Dedup: skip chunks whose checksum already exists.
        chunk_list = _dedup_chunks(chunk_list, repo)

        # Persist Document rows before chunks. local_fs intentionally uses one
        # document per file; other source types use one document per source.
        documents = _ensure_documents(source, repo, chunk_list)

        # Embed and upsert to vector store using the query-time embedder.
        if chunk_list:
            embed_and_upsert(repo, qdrant, chunk_list, embedder=embedder)

        doc_ids = [doc.doc_id for doc in documents]
        stats = {
            "doc_ids": doc_ids,
            "chunks_ingested": len(chunk_list),
        }
        # Retain the historical single-document convenience key where it is
        # unambiguous for API consumers.
        if len(doc_ids) == 1:
            stats["doc_id"] = doc_ids[0]

        repo.update_job(
            job_id,
            status="completed",
            doc_count=len(documents),
            chunk_count=len(chunk_list),
            completed_at=datetime.now(timezone.utc).isoformat(),
            stats=stats,
        )
        logger.info(
            "Pipeline completed: job=%s source=%s documents=%d chunks=%d",
            job_id,
            source.source_id,
            len(documents),
            len(chunk_list),
        )

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
    # Stable base identity. local_fs derives deterministic per-file document
    # IDs from this value; other source types use it directly.
    doc_id = config.get("doc_id") or f"doc-{source.source_id}"
    version = config.get("version", "1.0")
    tags = source.tags
    acl_hash = _resolve_acl_hash(source, repo)

    common = dict(
        doc_id=doc_id,
        tenant_id=tenant_id,
        version=version,
        tags=tags,
        acl_hash=acl_hash,
    )

    if source_type == "pdf":
        from services.worker.jobs.ingest_pdf import ingest_pdf
        return ingest_pdf(path=config["path"], **common)

    if source_type == "web":
        from services.worker.jobs.ingest_web import ingest_web
        return ingest_web(url=config["url"], **common)

    if source_type == "repo":
        from services.worker.jobs.ingest_repo import ingest_repo
        return ingest_repo(
            url_or_path=config["path"],
            ref=config.get("ref"),
            **common,
        )

    if source_type == "local_fs":
        from services.worker.jobs.ingest_text import ingest_local_fs
        return ingest_local_fs(
            directory=config["path"],
            extensions=config.get("extensions"),
            **common,
        )

    if source_type == "database":
        # Database sources don't ingest chunks; they enable SQL tool access.
        return []

    raise ValueError(f"Unsupported source_type: {source_type}")


def _resolve_acl_hash(source: Source, repo: InMemoryRepo) -> str:
    if source.acl_policy_id:
        policy_hash = repo.get_policy_hash(source.acl_policy_id)
        if policy_hash:
            return policy_hash
    return "public"


def _dedup_chunks(chunks: list[Chunk], repo: InMemoryRepo) -> list[Chunk]:
    """Remove chunks whose checksum already exists in the repo."""
    existing_checksums = {
        chunk.checksum for chunk in repo.iter_chunks() if chunk.checksum
    }

    deduped: list[Chunk] = []
    for chunk in chunks:
        if chunk.checksum and chunk.checksum in existing_checksums:
            logger.debug("Skipping duplicate chunk: %s", chunk.chunk_id)
            continue
        if chunk.checksum:
            existing_checksums.add(chunk.checksum)
        deduped.append(chunk)
    return deduped


def _ensure_documents(
    source: Source,
    repo: InMemoryRepo,
    chunks: list[Chunk],
) -> list[Document]:
    """Persist all Document rows required by the chunks being written.

    Most connectors model one Source as one Document. local_fs has historically
    modelled each file as a separate document so file-level filtering/citations
    remain precise. In that case each unique chunk ``doc_id`` receives a matching
    Document row before chunk persistence, satisfying Postgres foreign keys.
    """
    if source.source_type != "local_fs":
        return [_ensure_document(source, repo)]

    first_chunk_by_doc_id: dict[str, Chunk] = {}
    for chunk in chunks:
        first_chunk_by_doc_id.setdefault(chunk.doc_id, chunk)

    documents: list[Document] = []
    for doc_id, chunk in first_chunk_by_doc_id.items():
        file_path = Path(chunk.path) if chunk.path else None
        title = file_path.name if file_path else source.name
        uri = file_path.resolve().as_uri() if file_path else f"source://{source.source_id}"
        documents.append(
            _ensure_document(
                source,
                repo,
                doc_id=doc_id,
                title=title,
                uri=uri,
            )
        )
    return documents


def _ensure_document(
    source: Source,
    repo: InMemoryRepo,
    *,
    doc_id: Optional[str] = None,
    title: Optional[str] = None,
    uri: Optional[str] = None,
) -> Document:
    """Create or update one Document record."""
    now = datetime.now(timezone.utc).isoformat()
    resolved_doc_id = doc_id or source.config.get("doc_id") or f"doc-{source.source_id}"
    existing = repo.get_document(resolved_doc_id)

    if existing:
        existing.version = next_version(existing.version)
        existing.ingested_at = now
        existing.doc_updated_at = now
        repo.add_document(existing)
        return existing

    doc = Document(
        doc_id=resolved_doc_id,
        tenant_id=source.tenant_id,
        source_type=source.source_type,
        title=title or source.name,
        uri=uri or f"source://{source.source_id}",
        version=source.config.get("version", "1.0"),
        doc_updated_at=now,
        ingested_at=now,
        tags=source.tags,
        acl_policy_id=source.acl_policy_id,
    )
    repo.add_document(doc)
    return doc
