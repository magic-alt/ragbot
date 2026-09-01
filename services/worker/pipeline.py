"""Ingestion pipeline orchestrator.

Coordinates the full ingest lifecycle:
  Source config -> IngestionJob -> connector -> normalize/dedup -> embed/upsert
  -> prune stale chunks/documents -> update job status.

Re-ingestion is replacement-oriented: new chunks are written first, then stale
vectors/chunks from the previous successful view are removed. A failed write
therefore keeps the previous knowledge available and can be repaired by retry.
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


def run_ingest_pipeline(
    source: Source,
    repo: Repo,
    qdrant: object,
    job_id: Optional[str] = None,
    embedder: Optional[Embedder] = None,
) -> IngestionJob:
    """Execute one replacement-oriented ingestion run for ``source``."""
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
        previous_documents = source_documents(source, repo)
        previous_doc_ids = {doc.doc_id for doc in previous_documents}
        previous_chunks = {
            chunk.chunk_id: chunk
            for doc_id in previous_doc_ids
            for chunk in repo.list_chunks(doc_id)
        }

        chunk_list = list(_run_connector(source, repo))
        _normalize_chunk_metadata(source, chunk_list, now)
        chunk_list = _dedup_chunks(chunk_list)

        # Persist Document rows before chunks to satisfy PostgreSQL FKs.
        documents = _ensure_documents(source, repo, chunk_list)

        # Write the new view before deleting the old one. If embedding/upsert
        # fails, the stale view remains queryable and a retry can reconcile it.
        if chunk_list:
            embed_and_upsert(repo, qdrant, chunk_list, embedder=embedder)

        current_doc_ids = {doc.doc_id for doc in documents}
        current_chunk_ids = {chunk.chunk_id for chunk in chunk_list}
        stale_chunk_ids = set(previous_chunks) - current_chunk_ids
        removed_doc_ids = previous_doc_ids - current_doc_ids

        chunks_removed = _delete_qdrant_points(qdrant, stale_chunk_ids)
        repo.delete_chunks(stale_chunk_ids)

        # Delete by document as well to clean orphan vectors that may have been
        # left by a historical partial failure and are not represented in SQL.
        _delete_qdrant_documents(qdrant, removed_doc_ids)
        documents_removed = repo.delete_documents(removed_doc_ids)

        doc_ids = [doc.doc_id for doc in documents]
        stats = {
            "doc_ids": doc_ids,
            "chunks_ingested": len(chunk_list),
            "chunks_removed": chunks_removed,
            "documents_removed": documents_removed,
        }
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
            "Pipeline completed: job=%s source=%s documents=%d chunks=%d stale_chunks=%d removed_docs=%d",
            job_id,
            source.source_id,
            len(documents),
            len(chunk_list),
            len(stale_chunk_ids),
            len(removed_doc_ids),
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
    """Remove all indexed knowledge owned by a source.

    The Source record itself is intentionally left intact so the API can first
    purge data and only then mark the source deleted.
    """
    doc_ids = {doc.doc_id for doc in source_documents(source, repo)}
    vector_docs = _delete_qdrant_documents(qdrant, doc_ids)
    documents = repo.delete_documents(doc_ids)
    return {"documents": documents, "vector_documents": vector_docs}


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
    """Select and run the connector for one supported document source."""
    source_type = source.source_type
    config = source.config
    tenant_id = source.tenant_id
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

    raise ValueError(f"Unsupported source_type: {source_type}")


def _resolve_acl_hash(source: Source, repo: Repo) -> str:
    if source.acl_policy_id:
        policy_hash = repo.get_policy_hash(source.acl_policy_id)
        if policy_hash:
            return policy_hash
    return "public"


def _normalize_chunk_metadata(source: Source, chunks: list[Chunk], now: str) -> None:
    """Enforce connector-independent metadata used by both retrieval backends."""
    acl_hash = _resolve_acl_hash(source, repo=_RepoProxy(source, chunks)) if False else None
    # ACL is already resolved by connectors; normalize the source-level fields
    # that filters and Qdrant payloads rely on.
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        metadata["source_type"] = source.source_type
        metadata["tags"] = list(source.tags)
        metadata.setdefault("version", source.config.get("version", "1.0"))
        metadata["ingested_at"] = now
        metadata["doc_updated_at"] = now
        chunk.metadata = metadata


def _dedup_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Remove duplicate content only within the same document and ingest run.

    Repository-wide checksum deduplication is unsafe: identical text can be
    valid evidence in two documents or tenants. Cross-document dedup belongs
    in a content-addressed storage layer, not in logical document ingestion.
    """
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


def _ensure_documents(
    source: Source,
    repo: Repo,
    chunks: list[Chunk],
) -> list[Document]:
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
