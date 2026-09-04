from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from services.api.app.retrieval.embedder import Embedder, HashEmbedder
from services.api.app.retrieval.qdrant import point_id_for_chunk, to_epoch
from services.api.app.storage.models import Chunk
from services.api.app.storage.protocol import Repo

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100
_GENERATION_POINT_NAMESPACE = uuid.UUID("8941f748-ff48-4dc1-943a-6879173be958")


def point_id_for_generation_chunk(generation_id: str, chunk_id: str) -> str:
    """Return a deterministic physical point id for one staged generation."""
    return str(uuid.uuid5(_GENERATION_POINT_NAMESPACE, f"{generation_id}:{chunk_id}"))


def embed_and_stage_vectors(
    qdrant: object,
    chunks: Iterable[Chunk],
    *,
    generation_id: str,
    source_id: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    embedder: Optional[Embedder] = None,
) -> list[str]:
    """Embed changed chunks into generation-specific Qdrant points only.

    PostgreSQL staging/activation is handled separately. This function never
    mutates the active repository tables, so a crash during vector preparation
    cannot expose the candidate generation through lexical retrieval.
    """
    emb = embedder or HashEmbedder(dim=qdrant.dim)
    chunk_list: List[Chunk] = []
    written_point_ids: list[str] = []

    def flush() -> None:
        nonlocal chunk_list
        if not chunk_list:
            return
        vectors = emb.embed_batch([chunk.text for chunk in chunk_list])
        if len(vectors) != len(chunk_list):
            raise RuntimeError(
                f"Embedder returned {len(vectors)} vectors for {len(chunk_list)} chunks"
            )
        points: List[Tuple[str, List[float], Dict[str, Any]]] = []
        for chunk, vector in zip(chunk_list, vectors):
            if len(vector) != qdrant.dim:
                raise RuntimeError(
                    "Embedding dimension does not match vector store: "
                    f"chunk={chunk.chunk_id}, vector={len(vector)}, qdrant={qdrant.dim}"
                )
            metadata = dict(chunk.metadata or {})
            metadata.update(
                {
                    "embedding_model": emb.model_name,
                    "embedding_dimension": emb.dimension,
                    "source_id": source_id,
                    "generation_id": generation_id,
                }
            )
            chunk.metadata = metadata
            chunk.source_id = source_id
            chunk.generation_id = generation_id
            point_id = point_id_for_generation_chunk(generation_id, chunk.chunk_id)
            chunk.qdrant_point_id = point_id
            points.append((point_id, vector, _build_payload(chunk, emb.model_name)))
            written_point_ids.append(point_id)
        qdrant.upsert(points)
        logger.debug(
            "Staged vector batch: generation=%s points=%d",
            generation_id,
            len(points),
        )
        chunk_list = []

    for chunk in chunks:
        chunk_list.append(chunk)
        if len(chunk_list) >= batch_size:
            flush()
    flush()
    return written_point_ids


def embed_and_upsert(
    repo: Repo,
    qdrant: object,
    chunks: Iterable[Chunk],
    batch_size: int = DEFAULT_BATCH_SIZE,
    embedder: Optional[Embedder] = None,
) -> None:
    """Legacy direct publication path retained for custom repositories/callers."""
    emb = embedder or HashEmbedder(dim=qdrant.dim)
    vector_batch: List[Tuple[str, List[float], Dict[str, Any]]] = []
    chunk_list: List[Chunk] = []
    total = 0

    def flush() -> None:
        nonlocal vector_batch, chunk_list, total
        if not chunk_list:
            return
        vectors = emb.embed_batch([c.text for c in chunk_list])
        if len(vectors) != len(chunk_list):
            raise RuntimeError(
                f"Embedder returned {len(vectors)} vectors for {len(chunk_list)} chunks"
            )
        for chunk, vector in zip(chunk_list, vectors):
            if len(vector) != qdrant.dim:
                raise RuntimeError(
                    "Embedding dimension does not match vector store: "
                    f"chunk={chunk.chunk_id}, vector={len(vector)}, qdrant={qdrant.dim}"
                )
            metadata = dict(chunk.metadata or {})
            metadata["embedding_model"] = emb.model_name
            metadata["embedding_dimension"] = emb.dimension
            chunk.metadata = metadata
            point_id = point_id_for_chunk(chunk.chunk_id)
            chunk.qdrant_point_id = point_id
            vector_batch.append((point_id, vector, _build_payload(chunk, emb.model_name)))

        add_chunks = getattr(repo, "add_chunks", None)
        if callable(add_chunks):
            add_chunks(chunk_list)
        else:
            for chunk in chunk_list:
                repo.add_chunk(chunk)
        qdrant.upsert(vector_batch)
        total += len(vector_batch)
        logger.debug("Upserted batch of %d points (total: %d)", len(vector_batch), total)
        vector_batch = []
        chunk_list = []

    for chunk in chunks:
        chunk_list.append(chunk)
        if len(chunk_list) >= batch_size:
            flush()
    flush()


def _build_payload(chunk: Chunk, embedding_model: str = "hash-64") -> Dict[str, Any]:
    metadata = chunk.metadata or {}
    ingested_at = metadata.get("ingested_at")
    doc_updated_at = metadata.get("doc_updated_at")
    acl_hash = metadata.get("acl_hash") or "public"
    return {
        "chunk_id": chunk.chunk_id,
        "tenant_id": chunk.tenant_id,
        "source_id": chunk.source_id or metadata.get("source_id"),
        "generation_id": chunk.generation_id or metadata.get("generation_id"),
        "source_type": metadata.get("source_type"),
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "path": chunk.path,
        "url": chunk.url,
        "page": chunk.page,
        "section": chunk.section,
        "ingested_at": ingested_at,
        "doc_updated_at": doc_updated_at,
        "ingested_at_ts": to_epoch(ingested_at),
        "doc_updated_at_ts": to_epoch(doc_updated_at),
        "version": metadata.get("version"),
        "checksum": chunk.checksum,
        "acl_hash": acl_hash,
        "tags": metadata.get("tags") or [],
        "parser_provider": metadata.get("parser_provider"),
        "parser_strategy": metadata.get("parser_strategy"),
        "parser_version": metadata.get("parser_version"),
        "parser_config_hash": metadata.get("parser_config_hash"),
        "block_index": metadata.get("block_index"),
        "block_kind": metadata.get("block_kind"),
        "bbox": metadata.get("bbox"),
        "chunker_provider": metadata.get("chunker_provider"),
        "chunker_strategy": metadata.get("chunker_strategy"),
        "chunker_version": metadata.get("chunker_version"),
        "chunker_config_hash": metadata.get("chunker_config_hash"),
        "chunker_language": metadata.get("chunker_language"),
        "chunk_size": metadata.get("chunk_size"),
        "chunk_overlap": metadata.get("chunk_overlap"),
        "embedding_model": embedding_model,
        "embedding_dimension": metadata.get("embedding_dimension"),
        "text": chunk.text,
    }
