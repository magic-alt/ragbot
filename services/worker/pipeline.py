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
from services.api.app.retrieval.qdrant import normalize_qdrant_point_id
from services.api.app.storage.models import Chunk, Document, IngestionJob, Source
from services.api.app.storage.protocol import Repo
from services.worker.dedup.versioning import next_version
from services.worker.jobs.embed_and_upsert import embed_and_upsert
from services.worker.source_fence import (
    SourceFenceError,
    assert_source_fence,
    job_source_generation,
    job_stats_for_source,
    source_generation,
)

logger = logging.getLogger(__name__)
LEXICAL_VERSION = 2
_MULTI_DOCUMENT_SOURCE_TYPES = {"local_fs", "s3", "gdrive", "notion", "confluence"}
_REMOTE_DOCUMENT_SOURCE_TYPES = {"s3", "gdrive", "notion", "confluence"}


def run_ingest_pipeline(
    source: Source,
    repo: Repo,
    qdrant: object,
    job_id: Optional[str] = None,
    embedder: Optional[Embedder] = None,
    existing_job: bool = False,
    expected_source_generation: Optional[str] = None,
) -> IngestionJob:
    """Execute one replacement-oriented ingestion run for ``source``."""
    now = datetime.now(timezone.utc).isoformat()
    job_id = job_id or uuid.uuid4().hex
    embedding_model, embedding_dimension = _embedding_identity(embedder, qdrant)

    persisted_source = repo.get_source(source.source_id)
    current_job = repo.get_job(job_id) if existing_job else None
    expected_generation = expected_source_generation
    if expected_generation is None and current_job is not None:
        expected_generation = job_source_generation(current_job)
    if expected_generation is None:
        expected_generation = source_generation(persisted_source or source)

    if existing_job:
        if current_job is None:
            raise ValueError(f"Existing ingestion job not found: {job_id}")
        repo.update_job(
            job_id,
            status="running",
            started_at=current_job.started_at or now,
            error=None,
        )
    else:
        generation_source = persisted_source or source
        job = IngestionJob(
            job_id=job_id,
            tenant_id=source.tenant_id,
            source_id=source.source_id,
            source_type=source.source_type,
            source_config=source.config,
            status="running",
            started_at=now,
            created_at=now,
            stats=job_stats_for_source(generation_source),
        )
        repo.add_job(job)

    try:
        assert_source_fence(source, repo, expected_generation)
        previous_documents = source_documents(source, repo)
        previous_doc_ids = {doc.doc_id for doc in previous_documents}
        previous_chunks = {
            chunk.chunk_id: chunk
            for doc_id in previous_doc_ids
            for chunk in repo.list_chunks(doc_id)
        }

        candidate_chunks = list(_run_connector(source, repo, previous_chunks.values()))
        _normalize_chunk_metadata(
            source,
            candidate_chunks,
            now,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
        )
        candidate_chunks = _dedup_chunks(candidate_chunks)
        current_chunks, chunks_to_write, chunks_reused = _reuse_unchanged_chunks(
            candidate_chunks, previous_chunks.values()
        )

        assert_source_fence(source, repo, expected_generation)
        documents = _ensure_documents(source, repo, current_chunks)
        if chunks_to_write:
            embed_and_upsert(repo, qdrant, chunks_to_write, embedder=embedder)

        current_doc_ids = {doc.doc_id for doc in documents}
        current_chunk_ids = {chunk.chunk_id for chunk in current_chunks}
        stale_chunk_ids = set(previous_chunks) - current_chunk_ids
        removed_doc_ids = previous_doc_ids - current_doc_ids

        stale_point_ids = {
            normalize_qdrant_point_id(previous_chunks[chunk_id].qdrant_point_id, chunk_id)
            for chunk_id in stale_chunk_ids
        }
        vector_chunks_removed = _delete_qdrant_points(qdrant, stale_point_ids)
        chunks_removed = repo.delete_chunks(stale_chunk_ids)
        _delete_qdrant_documents(qdrant, removed_doc_ids)
        documents_removed = repo.delete_documents(removed_doc_ids)

        assert_source_fence(source, repo, expected_generation)

        doc_ids = [doc.doc_id for doc in documents]
        latest_job = repo.get_job(job_id)
        stats = dict((latest_job.stats if latest_job else {}) or {})
        stats.update({
            "doc_ids": doc_ids,
            "source_generation": expected_generation,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "parser_contracts": _parser_contracts(current_chunks),
            "chunking_contracts": _chunking_contracts(current_chunks),
            "chunks_total": len(current_chunks),
            "chunks_ingested": len(chunks_to_write),
            "chunks_reused": chunks_reused,
            "chunks_removed": chunks_removed,
            "vector_chunks_removed": vector_chunks_removed,
            "documents_removed": documents_removed,
        })
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
            "Pipeline completed: job=%s source=%s documents=%d chunks_total=%d written=%d reused=%d removed=%d embedding=%s/%d",
            job_id,
            source.source_id,
            len(documents),
            len(current_chunks),
            len(chunks_to_write),
            chunks_reused,
            chunks_removed,
            embedding_model,
            embedding_dimension,
        )
    except SourceFenceError as exc:
        logger.warning("Pipeline fenced: job=%s source=%s error=%s", job_id, source.source_id, exc)
        current_source = repo.get_source(source.source_id)
        if current_source is not None and current_source.status == "deleted":
            try:
                purge_source_knowledge(current_source, repo, qdrant)
            except Exception:
                logger.exception("Failed to purge knowledge after Source fence: %s", source.source_id)
        repo.update_job(
            job_id,
            status="failed",
            error=str(exc),
            completed_at=datetime.now(timezone.utc).isoformat(),
            lease_owner=None,
            lease_expires_at=None,
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
    if result is None:  # pragma: no cover
        raise RuntimeError(f"Ingestion job disappeared after execution: {job_id}")
    return result


def source_documents(source: Source, repo: Repo) -> list[Document]:
    """Return documents owned by ``source`` without unnecessary tenant scans."""
    base_doc_id = source.config.get("doc_id") or f"doc-{source.source_id}"
    if source.source_type not in _MULTI_DOCUMENT_SOURCE_TYPES:
        document = repo.get_document(base_doc_id)
        if document and document.tenant_id == source.tenant_id:
            return [document]
        return [
            doc
            for doc in repo.list_documents(source.tenant_id)
            if doc.uri and doc.uri.startswith(f"source://{source.source_id}")
        ]

    prefix = f"{base_doc_id}:"
    return [
        doc for doc in repo.list_documents(source.tenant_id)
        if doc.doc_id.startswith(prefix)
    ]


def purge_source_knowledge(source: Source, repo: Repo, qdrant: object) -> dict[str, int]:
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


def _run_connector(source: Source, repo: Repo, previous_chunks: Iterable[Chunk] = ()) -> Iterable[Chunk]:
    source_type = source.source_type
    config = source.config
    doc_id = config.get("doc_id") or f"doc-{source.source_id}"
    common = dict(
        doc_id=doc_id,
        tenant_id=source.tenant_id,
        version=config.get("version", "1.0"),
        tags=source.tags,
        acl_hash=_resolve_acl_hash(source, repo),
        chunking=config.get("chunking"),
    )

    if source_type == "pdf":
        from services.worker.jobs.ingest_pdf import ingest_pdf
        return ingest_pdf(
            path=config["path"],
            chunk_size=int(config.get("chunk_size", 800)),
            chunk_overlap=int(config.get("chunk_overlap", 100)),
            parsing=config.get("parsing"),
            **common,
        )
    if source_type == "web":
        from services.worker.jobs.ingest_web import ingest_web
        return ingest_web(
            url=config["url"],
            chunk_size=int(config.get("chunk_size", 800)),
            chunk_overlap=int(config.get("chunk_overlap", 100)),
            parsing=config.get("parsing"),
            **common,
        )
    if source_type == "repo":
        from services.worker.jobs.ingest_repo import ingest_repo
        return ingest_repo(
            url_or_path=config["path"],
            ref=config.get("ref"),
            chunk_size=int(config.get("chunk_size", 600)),
            chunk_overlap=int(config.get("chunk_overlap", 100)),
            **common,
        )
    if source_type == "local_fs":
        from services.worker.jobs.ingest_text import ingest_local_fs
        return ingest_local_fs(
            directory=config["path"],
            extensions=config.get("extensions"),
            chunk_size=int(config.get("chunk_size", 800)),
            chunk_overlap=int(config.get("chunk_overlap", 100)),
            parsing=config.get("parsing"),
            **common,
        )
    if source_type == "s3":
        from services.worker.jobs.ingest_s3 import ingest_s3
        return ingest_s3(
            bucket=config["bucket"],
            prefix=config.get("prefix", ""),
            endpoint_url=config.get("endpoint_url"),
            region_name=config.get("region_name"),
            credential_env_prefix=config.get("credential_env_prefix"),
            extensions=config.get("extensions"),
            max_object_bytes=int(config.get("max_object_bytes", 20 * 1024 * 1024)),
            chunk_size=int(config.get("chunk_size", 800)),
            chunk_overlap=int(config.get("chunk_overlap", 100)),
            parsing=config.get("parsing"),
            **common,
        )
    if source_type == "gdrive":
        from services.worker.jobs.ingest_google_drive import ingest_google_drive
        return ingest_google_drive(
            folder_id=config["folder_id"],
            credential_ref=config["credential_ref"],
            credential_type=config.get("credential_type", "access_token"),
            recursive=bool(config.get("recursive", True)),
            max_file_bytes=int(config.get("max_file_bytes", 20 * 1024 * 1024)),
            chunk_size=int(config.get("chunk_size", 800)),
            chunk_overlap=int(config.get("chunk_overlap", 100)),
            previous_chunks=previous_chunks,
            parsing=config.get("parsing"),
            **common,
        )
    if source_type == "notion":
        from services.worker.jobs.ingest_notion import ingest_notion
        return ingest_notion(
            page_id=config["page_id"],
            credential_ref=config["credential_ref"],
            recursive=bool(config.get("recursive", True)),
            notion_version=config.get("notion_version", "2022-06-28"),
            chunk_size=int(config.get("chunk_size", 800)),
            chunk_overlap=int(config.get("chunk_overlap", 100)),
            previous_chunks=previous_chunks,
            **common,
        )
    if source_type == "confluence":
        from services.worker.jobs.ingest_confluence import ingest_confluence
        return ingest_confluence(
            base_url=config["base_url"],
            space_key=config["space_key"],
            credential_ref=config["credential_ref"],
            auth_type=config.get("auth_type", "basic"),
            email=config.get("email"),
            root_page_id=config.get("root_page_id"),
            chunk_size=int(config.get("chunk_size", 800)),
            chunk_overlap=int(config.get("chunk_overlap", 100)),
            previous_chunks=previous_chunks,
            **common,
        )
    raise ValueError(f"Unsupported source_type: {source_type}")


def _resolve_acl_hash(source: Source, repo: Repo) -> str:
    if source.acl_policy_id:
        policy_hash = repo.get_policy_hash(source.acl_policy_id)
        if policy_hash:
            return policy_hash
    return "public"


def _embedding_identity(embedder: Optional[Embedder], qdrant: object) -> tuple[str, int]:
    dimension = int(getattr(embedder, "dimension", getattr(qdrant, "dim", 64)))
    model = str(getattr(embedder, "model_name", f"hash-{dimension}"))
    return model, dimension


def _normalize_chunk_metadata(
    source: Source,
    chunks: list[Chunk],
    now: str,
    embedding_model: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
) -> None:
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        metadata["source_type"] = source.source_type
        metadata["tags"] = list(source.tags)
        metadata.setdefault("version", source.config.get("version", "1.0"))
        metadata["lexical_version"] = LEXICAL_VERSION
        if embedding_model:
            metadata["embedding_model"] = embedding_model
        if embedding_dimension is not None:
            metadata["embedding_dimension"] = int(embedding_dimension)
        metadata["ingested_at"] = now
        metadata["doc_updated_at"] = now
        chunk.metadata = metadata


def _dedup_chunks(chunks: list[Chunk]) -> list[Chunk]:
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
        candidate.qdrant_point_id = normalize_qdrant_point_id(old.qdrant_point_id, old.chunk_id)
        candidate.created_at = old.created_at
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
        metadata.get("remote_version"),
        metadata.get("lexical_version"),
        metadata.get("parser_provider"),
        metadata.get("parser_strategy"),
        metadata.get("parser_version"),
        metadata.get("parser_config_hash"),
        metadata.get("chunker_provider"),
        metadata.get("chunker_strategy"),
        metadata.get("chunker_version"),
        metadata.get("chunker_config_hash"),
        metadata.get("embedding_model"),
        metadata.get("embedding_dimension"),
    )


def _parser_contracts(chunks: Iterable[Chunk]) -> list[dict[str, object]]:
    contracts: dict[str, dict[str, object]] = {}
    for chunk in chunks:
        metadata = chunk.metadata or {}
        config_hash = str(metadata.get("parser_config_hash") or "")
        if not config_hash or config_hash in contracts:
            continue
        contracts[config_hash] = {
            "provider": metadata.get("parser_provider"),
            "strategy": metadata.get("parser_strategy"),
            "version": metadata.get("parser_version"),
            "config_hash": config_hash,
        }
    return list(contracts.values())


def _chunking_contracts(chunks: Iterable[Chunk]) -> list[dict[str, object]]:
    contracts: dict[str, dict[str, object]] = {}
    for chunk in chunks:
        metadata = chunk.metadata or {}
        config_hash = str(metadata.get("chunker_config_hash") or "")
        if not config_hash or config_hash in contracts:
            continue
        contracts[config_hash] = {
            "provider": metadata.get("chunker_provider"),
            "strategy": metadata.get("chunker_strategy"),
            "version": metadata.get("chunker_version"),
            "config_hash": config_hash,
            "chunk_size": metadata.get("chunk_size"),
            "chunk_overlap": metadata.get("chunk_overlap"),
            "language": metadata.get("chunker_language"),
        }
    return list(contracts.values())


def _ensure_documents(source: Source, repo: Repo, chunks: list[Chunk]) -> list[Document]:
    if source.source_type not in _MULTI_DOCUMENT_SOURCE_TYPES:
        if not chunks:
            return []
        return [_ensure_document(source, repo)]

    first_chunk_by_doc_id: dict[str, Chunk] = {}
    for chunk in chunks:
        first_chunk_by_doc_id.setdefault(chunk.doc_id, chunk)
    documents: list[Document] = []
    for doc_id, chunk in first_chunk_by_doc_id.items():
        metadata = chunk.metadata or {}
        if source.source_type in _REMOTE_DOCUMENT_SOURCE_TYPES:
            title = str(
                metadata.get("document_title")
                or metadata.get("filename")
                or metadata.get("object_key")
                or source.name
            )
            uri = str(metadata.get("document_uri") or chunk.url or chunk.path or f"source://{source.source_id}/{doc_id}")
        else:
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