"""Ingestion pipeline orchestrator.

Coordinates Source -> connector -> normalized chunks -> embedding/vector upsert ->
stale-data pruning. Re-ingestion writes changed data first and removes the old
view only after the new write succeeds, preserving a retryable last-good view.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from services.api.app.retrieval.embedder import Embedder
from services.api.app.storage.models import Chunk, Document, IngestionJob, Source
from services.api.app.storage.protocol import Repo
from services.worker.dedup.versioning import next_version
from services.worker.jobs.embed_and_upsert import embed_and_upsert

logger = logging.getLogger(__name__)
LEXICAL_VERSION = 2


def run_ingest_pipeline(
    source: Source,
    repo: Repo,
    qdrant: object,
    job_id: Optional[str] = None,
    embedder: Optional[Embedder] = None,
    existing_job: bool = False,
) -> IngestionJob:
    """Execute one replacement-oriented ingestion run for ``source``.

    ``existing_job=True`` is used by the durable worker after it atomically
    claims a queued job. Direct/local callers retain the historical behavior of
    creating the job record inside the pipeline.
    """
    now = datetime.now(timezone.utc).isoformat()
    job_id = job_id or uuid.uuid4().hex
    if existing_job:
        current = repo.get_job(job_id)
        if current is None:
            raise ValueError(f"Existing ingestion job not found: {job_id}")
        repo.update_job(job_id, status="running", started_at=current.started_at or now, error=None)
    else:
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
        previous_documents = source_documents(source, repo)
        previous_doc_ids = {doc.doc_id for doc in previous_documents}
        previous_chunks = {
            chunk.chunk_id: chunk
            for doc_id in previous_doc_ids
            for chunk in repo.list_chunks(doc_id)
        }

        candidate_chunks = list(_run_connector(source, repo))
        _normalize_chunk_metadata(source, candidate_chunks, now)
        candidate_chunks = _dedup_chunks(candidate_chunks)
        current_chunks, chunks_to_write, chunks_reused = _reuse_unchanged_chunks(
            candidate_chunks,
            previous_chunks.values(),
        )

        # Documents must exist before new chunks to satisfy PostgreSQL FKs.
        documents = _ensure_documents(source, repo, current_chunks)

        # Only changed/new chunks are re-embedded. Unchanged chunks keep their
        # point IDs and vector payloads, preserving the historical dedup contract
        # while avoiding repository-wide checksum data loss.
        if chunks_to_write:
            embed_and_upsert(repo, qdrant, chunks_to_write, embedder=embedder)

        current_doc_ids = {doc.doc_id for doc in documents}
        current_chunk_ids = {chunk.chunk_id for chunk in current_chunks}
        stale_chunk_ids = set(previous_chunks) - current_chunk_ids
        removed_doc_ids = previous_doc_ids - current_doc_ids

        vector_chunks_removed = _delete_qdrant_points(qdrant, stale_chunk_ids)
        chunks_removed = repo.delete_chunks(stale_chunk_ids)

        # Also delete by document to clean vector orphans from historical
        # partial failures that may no longer have a matching SQL chunk row.
        _delete_qdrant_documents(qdrant, removed_doc_ids)
        documents_removed = repo.delete_documents(removed_doc_ids)

        doc_ids = [doc.doc_id for doc in documents]
        stats = {
            "doc_ids": doc_ids,
            "chunks_total": len(current_chunks),
            "chunks_ingested": len(chunks_to_write),
            "chunks_reused": chunks_reused,
            "chunks_removed": chunks_removed,
            "vector_chunks_removed": vector_chunks_removed,
            "documents_removed": documents_removed,
        }
        if len(doc_ids) == 1:
            stats["doc_id"] = doc_ids[0]

        repo.update_job(
            job_id,
            status="completed",
            doc_count=len(documents),
            chunk_count=len(chunks_to_write),
            completed_at=datetime.now(timezone.utc).isoformat(),
            stats=stats,
            lease_owner=None,
            lease_expires_at=None,
        )
        logger.info(
            "Pipeline completed: job=%s source=%s documents=%d chunks_total=%d written=%d reused=%d removed=%d",
            job_id,
            source.source_id,
            len(documents),
            len(current_chunks),
            len(chunks_to_write),
            chunks_reused,
            chunks_removed,
        )

    except Exception as exc:
        logger.exception("Pipeline failed: job=%s source=%s", job_id, source.source_id)
        repo.update_job(
            job_id,
            status="failed",
            error=str(exc),
            completed_at=datetime.now(timezone.utc).isoformat(),
            lease_owner=None,
            lease_expires_at=None,
        )

    result = repo.get_job(job_id)
    if result is None:  # pragma: no cover - repository contract guard
        raise RuntimeError(f"Ingestion job disappeared after execution: {job_id}")
    return result


def source_documents(source: Source, repo: Repo) -> list[Document]:
    """Return documents currently owned by ``source`` within its tenant."""
    base_doc_id = source.config.get("doc_id") or f"doc-{source.source_id}"
    documents = repo.list_documents(source.tenant_id)
    if source.source_type == "local_fs":
        prefix = f"{base_doc_id}:"
        return [doc for doc in documents if doc.doc_id.startswith(prefix)]
    return [
        doc
        for doc in documents
        if doc.doc_id == base_doc_id
        or (doc.uri and doc.uri.startswith(f"source://{source.source_id}"))
    ]


def purge_source_knowledge(source: Source, repo: Repo, qdrant: object) -> dict[str, int]:
    """Remove all indexed knowledge owned by a source before tombstoning it."""
    doc_ids = {doc.doc_id for doc in source_documents(source, repo)}
    vector_documents = _delete_qdrant_documents(qdrant, doc_ids)
    documents = repo.delete_documents(doc_ids)
    return {"documents": documents, "vector_documents": vector_documents}


def _delete_qdrant_points(qdrant: object, point_ids: set[str]) -> int:
    if not point_ids:
        return 0
    delete = getattr(qdrant, "delete_points", None)
    if not callable(delete):
        logger.warning("Vector store does not support point deletion; stale vectors may remain")
        return 0
    return int(delete(point_ids) or 0)


def _delete_qdrant_documents(qdrant: object, doc_ids: set[str]) -> int:
    if not doc_ids:
        return 0
    delete = getattr(qdrant, "delete_by_doc_ids", None)
    if not callable(delete):
        logger.warning("Vector store does not support document deletion; orphan vectors may remain")
        return 0
    return int(delete(doc_ids) or 0)


def _run_connector(source: Source, repo: Repo) -> Iterable[Chunk]:
    source_type = source.source_type
    config = source.config
    doc_id = config.get("doc_id") or f"doc-{source.source_id}"
    common = dict(
        doc_id=doc_id,
        tenant_id=source.tenant_id,
        version=config.get("version", "1.0"),
        tags=source.tags,
        acl_hash=_resolve_acl_hash(source, repo),
    )

    if source_type == "pdf":
        from services.worker.jobs.ingest_pdf import ingest_pdf
        return ingest_pdf(path=config["path"], **common)
    if source_type == "web":
        from services.worker.jobs.ingest_web import ingest_web
        return ingest_web(url=config["url"], **common)
    if source_type == "repo":
        from services.worker.jobs.ingest_repo import ingest_repo
        return ingest_repo(url_or_path=config["path"], ref=config.get("ref"), **common)
    if source_type == "local_fs":
        from services.worker.jobs.ingest_text import ingest_local_fs
        return ingest_local_fs(
            directory=config["path"],
            extensions=config.get("extensions"),
            **common,
        )
    raise ValueError(f"Unsupported source_type: {source_type}")


def _resolve_acl_hash(source: Source, repo: Repo) -> str:
    if source.acl_policy_id:
        policy_hash = repo.get_policy_hash(source.acl_policy_id)
        if policy_hash:
            return policy_hash
    return "public"


def _normalize_chunk_metadata(source: Source, chunks: list[Chunk], now: str) -> None:
    """Enforce connector-independent metadata used by both retrieval backends."""
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        metadata["source_type"] = source.source_type
        metadata["tags"] = list(source.tags)
        metadata.setdefault("version", source.config.get("version", "1.0"))
        metadata["lexical_version"] = LEXICAL_VERSION
        metadata["ingested_at"] = now
        metadata["doc_updated_at"] = now
        chunk.metadata = metadata


def _dedup_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Deduplicate only within one logical document in the current ingest run."""
    seen: set[tuple[str, str]] = set()
    deduped: list[Chunk] = []
    for chunk in chunks:
        if not chunk.checksum:
            deduped.append(chunk)
            continue
        key = (chunk.doc_id, chunk.checksum)
        if key in seen:
            logger.debug("Skipping duplicate content within document: %s", chunk.chunk_id)
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


def _reuse_unchanged_chunks(
    candidates: list[Chunk],
    previous: Iterable[Chunk],
) -> tuple[list[Chunk], list[Chunk], int]:
    """Reuse stable point IDs when content and retrieval metadata are unchanged."""
    previous_by_key = {_reuse_key(chunk): chunk for chunk in previous}
    current: list[Chunk] = []
    to_write: list[Chunk] = []
    reused = 0

    for candidate in candidates:
        old = previous_by_key.get(_reuse_key(candidate))
        if old is None:
            current.append(candidate)
            to_write.append(candidate)
            continue

        candidate.chunk_id = old.chunk_id
        candidate.qdrant_point_id = old.qdrant_point_id or old.chunk_id
        candidate.created_at = old.created_at
        # Keep persisted metadata exactly aligned with the unchanged Qdrant
        # payload. Ingestion timestamps describe the chunk version, not the job.
        candidate.metadata = dict(old.metadata or {})
        current.append(candidate)
        reused += 1

    return current, to_write, reused


def _reuse_key(chunk: Chunk) -> tuple:
    metadata = chunk.metadata or {}
    return (
        chunk.doc_id,
        chunk.chunk_index,
        chunk.checksum,
        chunk.path,
        chunk.url,
        chunk.page,
        chunk.section,
        metadata.get("source_type"),
        tuple(metadata.get("tags") or []),
        metadata.get("acl_hash") or "public",
        metadata.get("version"),
        metadata.get("lexical_version"),
    )


def _ensure_documents(source: Source, repo: Repo, chunks: list[Chunk]) -> list[Document]:
    if source.source_type != "local_fs":
        if not chunks:
            return []
        return [_ensure_document(source, repo)]

    first_chunk_by_doc_id: dict[str, Chunk] = {}
    for chunk in chunks:
        first_chunk_by_doc_id.setdefault(chunk.doc_id, chunk)

    documents: list[Document] = []
    for doc_id, chunk in first_chunk_by_doc_id.items():
        file_path = Path(chunk.path) if chunk.path else None
        title = file_path.name if file_path else source.name
        uri = file_path.resolve().as_uri() if file_path else f"source://{source.source_id}"
        documents.append(_ensure_document(source, repo, doc_id=doc_id, title=title, uri=uri))
    return documents


def _ensure_document(
    source: Source,
    repo: Repo,
    *,
    doc_id: Optional[str] = None,
    title: Optional[str] = None,
    uri: Optional[str] = None,
) -> Document:
    now = datetime.now(timezone.utc).isoformat()
    resolved_doc_id = doc_id or source.config.get("doc_id") or f"doc-{source.source_id}"
    existing = repo.get_document(resolved_doc_id)

    if existing:
        existing.version = next_version(existing.version)
        existing.ingested_at = now
        existing.doc_updated_at = now
        existing.title = title or existing.title
        existing.uri = uri or existing.uri
        existing.tags = list(source.tags)
        existing.acl_policy_id = source.acl_policy_id
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
        tags=list(source.tags),
        acl_policy_id=source.acl_policy_id,
    )
    repo.add_document(doc)
    return doc
