"""Ingestion pipeline orchestrator.

Built-in repositories publish through staged knowledge generations: connectors
produce a complete candidate snapshot, changed vectors are prepared under
physical generation-specific point IDs, PostgreSQL stages the candidate, and a
single PostgreSQL transaction activates the new manifest and records cleanup in
a durable outbox. Custom repositories without the generation capability retain
the legacy replacement path for compatibility.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from services.api.app.retrieval.embedder import Embedder
from services.api.app.retrieval.embedding_contract import EmbeddingSpec
from services.api.app.retrieval.qdrant import normalize_qdrant_point_id
from services.api.app.storage.generation_support import (
    ensure_generation_repository,
    supports_generation_publication,
)
from services.api.app.storage.models import Chunk, Document, IngestionJob, KnowledgeGeneration, Source
from services.api.app.storage.protocol import Repo
from services.worker.connectors.registry import connector_registry
from services.worker.dedup.versioning import next_version
from services.worker.jobs.embed_and_upsert import (
    embed_and_stage_vectors,
    embed_and_upsert,
    point_id_for_generation_chunk,
)
from services.worker.source_fence import (
    SourceFenceError,
    assert_source_fence,
    job_source_generation,
    job_stats_for_source,
    source_generation,
)

logger = logging.getLogger(__name__)
LEXICAL_VERSION = 2


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
    embedding_model, embedding_dimension, embedding_contract_id = _embedding_identity(embedder, qdrant)
    ensure_generation_repository(repo)
    staged_publication = supports_generation_publication(repo)
    publication_generation_id: Optional[str] = None
    staged_point_ids: list[str] = []
    publication_activated = False

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
            embedding_contract_id=embedding_contract_id,
        )
        candidate_chunks = _dedup_chunks(candidate_chunks)
        current_chunks, chunks_to_write, chunks_reused = _reuse_unchanged_chunks(
            candidate_chunks, previous_chunks.values()
        )

        current_chunk_ids = {chunk.chunk_id for chunk in current_chunks}
        stale_chunk_ids = set(previous_chunks) - current_chunk_ids
        stale_point_ids = {
            normalize_qdrant_point_id(previous_chunks[chunk_id].qdrant_point_id, chunk_id)
            for chunk_id in stale_chunk_ids
        }

        assert_source_fence(source, repo, expected_generation)
        if staged_publication:
            publication_generation_id = f"gen-{job_id}-{uuid.uuid4().hex[:12]}"
            generation = KnowledgeGeneration(
                generation_id=publication_generation_id,
                source_id=source.source_id,
                tenant_id=source.tenant_id,
                job_id=job_id,
                status="staging",
                created_at=now,
                stats={"source_generation": expected_generation},
            )
            repo.begin_knowledge_generation(generation)

            documents = _prepare_documents(
                source,
                repo,
                current_chunks,
                generation_id=publication_generation_id,
            )
            _tag_generation(source, current_chunks, publication_generation_id)
            for chunk in chunks_to_write:
                chunk.qdrant_point_id = point_id_for_generation_chunk(
                    publication_generation_id,
                    chunk.chunk_id,
                )

            repo.stage_knowledge_generation(
                publication_generation_id,
                documents,
                current_chunks,
            )

            if chunks_to_write:
                staged_point_ids = embed_and_stage_vectors(
                    qdrant,
                    chunks_to_write,
                    generation_id=publication_generation_id,
                    source_id=source.source_id,
                    embedder=embedder,
                )
                repo.stage_knowledge_generation(
                    publication_generation_id,
                    documents,
                    current_chunks,
                )

            current_doc_ids = {doc.doc_id for doc in documents}
            removed_doc_ids = previous_doc_ids - current_doc_ids
            repo.mark_knowledge_generation_prepared(
                publication_generation_id,
                stats={
                    "documents": len(documents),
                    "chunks": len(current_chunks),
                    "chunks_written": len(chunks_to_write),
                    "chunks_reused": chunks_reused,
                    "stale_chunks": len(stale_chunk_ids),
                },
            )

            assert_source_fence(source, repo, expected_generation)
            old_publication_generation = repo.activate_knowledge_generation(
                source.source_id,
                publication_generation_id,
                cleanup_point_ids=stale_point_ids,
                previous_doc_ids=previous_doc_ids,
            )
            publication_activated = True
            chunks_removed = len(stale_chunk_ids)
            documents_removed = len(removed_doc_ids)
            vector_chunks_removed = 0
            vector_cleanup_enqueued = len(stale_point_ids)
        else:
            documents = _ensure_documents_legacy(source, repo, current_chunks)
            if chunks_to_write:
                embed_and_upsert(repo, qdrant, chunks_to_write, embedder=embedder)

            current_doc_ids = {doc.doc_id for doc in documents}
            removed_doc_ids = previous_doc_ids - current_doc_ids
            vector_chunks_removed = _delete_qdrant_points(qdrant, stale_point_ids)
            chunks_removed = repo.delete_chunks(stale_chunk_ids)
            _delete_qdrant_documents(qdrant, removed_doc_ids)
            documents_removed = repo.delete_documents(removed_doc_ids)
            vector_cleanup_enqueued = 0
            old_publication_generation = None

        assert_source_fence(source, repo, expected_generation)

        doc_ids = [doc.doc_id for doc in documents]
        latest_job = repo.get_job(job_id)
        stats = dict((latest_job.stats if latest_job else {}) or {})
        stats.update({
            "doc_ids": doc_ids,
            "source_generation": expected_generation,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "embedding_contract_id": embedding_contract_id,
            "parser_contracts": _parser_contracts(current_chunks),
            "chunking_contracts": _chunking_contracts(current_chunks),
            "publication_mode": "staged-generation" if staged_publication else "legacy-direct",
            "knowledge_generation_id": publication_generation_id,
            "previous_knowledge_generation_id": old_publication_generation,
            "chunks_total": len(current_chunks),
            "chunks_ingested": len(chunks_to_write),
            "chunks_reused": chunks_reused,
            "chunks_removed": chunks_removed,
            "vector_chunks_removed": vector_chunks_removed,
            "vector_cleanup_enqueued": vector_cleanup_enqueued,
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
            "Pipeline completed: job=%s source=%s generation=%s documents=%d chunks_total=%d written=%d reused=%d removed=%d embedding=%s/%d contract=%s",
            job_id,
            source.source_id,
            publication_generation_id or "legacy-direct",
            len(documents),
            len(current_chunks),
            len(chunks_to_write),
            chunks_reused,
            chunks_removed,
            embedding_model,
            embedding_dimension,
            embedding_contract_id,
        )
    except SourceFenceError as exc:
        logger.warning("Pipeline fenced: job=%s source=%s error=%s", job_id, source.source_id, exc)
        if staged_publication and publication_generation_id and not publication_activated:
            try:
                repo.fail_knowledge_generation(
                    publication_generation_id,
                    str(exc),
                    cleanup_point_ids=staged_point_ids,
                )
            except Exception:
                logger.exception("Failed to mark fenced generation failed: %s", publication_generation_id)
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
        if staged_publication and publication_generation_id and not publication_activated:
            try:
                repo.fail_knowledge_generation(
                    publication_generation_id,
                    str(exc),
                    cleanup_point_ids=staged_point_ids,
                )
            except Exception:
                logger.exception("Failed to mark generation failed: %s", publication_generation_id)
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


def _connector_capability(source_type: str, capability: str) -> bool:
    spec = connector_registry().get(source_type)
    return bool(getattr(spec.capabilities, capability, False))


def source_documents(source: Source, repo: Repo) -> list[Document]:
    """Return documents owned by ``source`` without unnecessary tenant scans."""
    base_doc_id = source.config.get("doc_id") or f"doc-{source.source_id}"
    if not _connector_capability(source.source_type, "multi_document"):
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


def _delete_qdrant_points(qdrant: object, point_ids: Iterable[str]) -> int:
    ids = set(str(item) for item in point_ids if item)
    if not ids:
        return 0
    delete = getattr(qdrant, "delete_points", None)
    if not callable(delete):
        logger.warning("Vector store does not support point deletion; stale vectors may remain")
        return 0
    return int(delete(ids) or 0)


def _delete_qdrant_documents(qdrant: object, doc_ids: set[str]) -> int:
    if not doc_ids:
        return 0
    delete = getattr(qdrant, "delete_by_doc_ids", None)
    if not callable(delete):
        logger.warning("Vector store does not support document deletion; orphan vectors may remain")
        return 0
    return int(delete(doc_ids) or 0)


def _run_connector(source: Source, repo: Repo, previous_chunks: Iterable[Chunk] = ()) -> Iterable[Chunk]:
    """Resolve connector execution through the shared platform registry."""
    return connector_registry().ingest(source, repo, previous_chunks)


def _embedding_identity(embedder: Optional[Embedder], qdrant: object) -> tuple[str, int, str]:
    dimension = int(getattr(embedder, "dimension", getattr(qdrant, "dim", 64)))
    model = str(getattr(embedder, "model_name", f"hash-{dimension}"))
    contract = str(getattr(embedder, "contract_id", "") or "")
    if not contract:
        provider_id = "hash" if embedder is None else type(embedder).__name__.lower()
        contract = EmbeddingSpec(
            provider_id=provider_id,
            model=model,
            dimension=dimension,
            normalize=(embedder is None),
        ).contract_id
    return model, dimension, contract


def _normalize_chunk_metadata(
    source: Source,
    chunks: list[Chunk],
    now: str,
    embedding_model: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
    embedding_contract_id: Optional[str] = None,
) -> None:
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        metadata["source_type"] = source.source_type
        metadata["source_id"] = source.source_id
        metadata["tags"] = list(source.tags)
        metadata.setdefault("version", source.config.get("version", "1.0"))
        metadata["lexical_version"] = LEXICAL_VERSION
        if embedding_model:
            metadata["embedding_model"] = embedding_model
        if embedding_dimension is not None:
            metadata["embedding_dimension"] = int(embedding_dimension)
        if embedding_contract_id:
            metadata["embedding_contract_id"] = embedding_contract_id
        metadata["ingested_at"] = now
        metadata["doc_updated_at"] = now
        chunk.source_id = source.source_id
        chunk.metadata = metadata


def _tag_generation(source: Source, chunks: Iterable[Chunk], generation_id: str) -> None:
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        metadata["source_id"] = source.source_id
        metadata["generation_id"] = generation_id
        chunk.source_id = source.source_id
        chunk.generation_id = generation_id
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
        candidate.source_id = old.source_id
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
        metadata.get("embedding_contract_id"),
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


def _prepare_documents(
    source: Source,
    repo: Repo,
    chunks: list[Chunk],
    *,
    generation_id: str,
) -> list[Document]:
    if not _connector_capability(source.source_type, "multi_document"):
        if not chunks:
            return []
        return [
            _prepare_document(
                source,
                repo,
                generation_id=generation_id,
            )
        ]

    first_chunk_by_doc_id: dict[str, Chunk] = {}
    for chunk in chunks:
        first_chunk_by_doc_id.setdefault(chunk.doc_id, chunk)
    documents: list[Document] = []
    for doc_id, chunk in first_chunk_by_doc_id.items():
        metadata = chunk.metadata or {}
        if _connector_capability(source.source_type, "remote"):
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
        documents.append(
            _prepare_document(
                source,
                repo,
                generation_id=generation_id,
                doc_id=doc_id,
                title=title,
                uri=uri,
            )
        )
    return documents


def _prepare_document(
    source: Source,
    repo: Repo,
    *,
    generation_id: str,
    doc_id: Optional[str] = None,
    title: Optional[str] = None,
    uri: Optional[str] = None,
) -> Document:
    now = datetime.now(timezone.utc).isoformat()
    resolved_doc_id = doc_id or source.config.get("doc_id") or f"doc-{source.source_id}"
    existing = repo.get_document(resolved_doc_id)
    if existing:
        return Document(
            doc_id=resolved_doc_id,
            tenant_id=source.tenant_id,
            source_type=source.source_type,
            title=title or existing.title,
            uri=uri or existing.uri,
            version=next_version(existing.version),
            doc_updated_at=now,
            ingested_at=now,
            tags=list(source.tags),
            acl_policy_id=source.acl_policy_id,
            status="active",
            source_id=source.source_id,
            generation_id=generation_id,
        )
    return Document(
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
        status="active",
        source_id=source.source_id,
        generation_id=generation_id,
    )


def _ensure_documents_legacy(source: Source, repo: Repo, chunks: list[Chunk]) -> list[Document]:
    if not _connector_capability(source.source_type, "multi_document"):
        if not chunks:
            return []
        return [_ensure_document_legacy(source, repo)]

    first_chunk_by_doc_id: dict[str, Chunk] = {}
    for chunk in chunks:
        first_chunk_by_doc_id.setdefault(chunk.doc_id, chunk)
    documents: list[Document] = []
    for doc_id, chunk in first_chunk_by_doc_id.items():
        metadata = chunk.metadata or {}
        if _connector_capability(source.source_type, "remote"):
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
        documents.append(_ensure_document_legacy(source, repo, doc_id=doc_id, title=title, uri=uri))
    return documents


def _ensure_document_legacy(
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
        source_id=source.source_id,
    )
    repo.add_document(doc)
    return doc
