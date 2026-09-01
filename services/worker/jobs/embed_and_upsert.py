from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from services.api.app.retrieval.embedder import Embedder, HashEmbedder
from services.api.app.retrieval.qdrant import point_id_for_chunk, to_epoch
from services.api.app.storage.models import Chunk
from services.api.app.storage.protocol import Repo

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100


def embed_and_upsert(
    repo: Repo,
    qdrant: object,
    chunks: Iterable[Chunk],
    batch_size: int = DEFAULT_BATCH_SIZE,
    embedder: Optional[Embedder] = None,
) -> None:
    """Persist chunks and vectors in bounded batches.

    SQL keeps Ragbot's logical ``chunk_id`` while Qdrant receives a stable UUID
    derived from it. The logical ID is also stored in the vector payload so
    retrieval can fuse vector and lexical rankings without coupling SQL primary
    keys to Qdrant's restricted point-ID type.
    """
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
            point_id = point_id_for_chunk(chunk.chunk_id)
            chunk.qdrant_point_id = point_id
            vector_batch.append((point_id, vector, _build_payload(chunk, emb.model_name)))

        add_chunks = getattr(repo, "add_chunks", None)
        if callable(add_chunks):
            add_chunks(chunk_list)
        else:  # compatibility with third-party Repo implementations
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
    ingested_at = chunk.metadata.get("ingested_at")
    doc_updated_at = chunk.metadata.get("doc_updated_at")
    acl_hash = chunk.metadata.get("acl_hash") or "public"
    return {
        "chunk_id": chunk.chunk_id,
        "tenant_id": chunk.tenant_id,
        "source_type": chunk.metadata.get("source_type"),
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
        "version": chunk.metadata.get("version"),
        "checksum": chunk.checksum,
        "acl_hash": acl_hash,
        "tags": chunk.metadata.get("tags") or [],
        "embedding_model": embedding_model,
        "text": chunk.text,
    }
