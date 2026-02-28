from __future__ import annotations

import logging
from typing import Iterable, List, Tuple, Any, Dict, Optional

from services.api.app.retrieval.qdrant import to_epoch
from services.api.app.retrieval.embedder import Embedder, HashEmbedder
from services.api.app.storage.models import Chunk
from services.api.app.storage.repo import InMemoryRepo

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100


def embed_and_upsert(
    repo: InMemoryRepo,
    qdrant: object,
    chunks: Iterable[Chunk],
    batch_size: int = DEFAULT_BATCH_SIZE,
    embedder: Optional[Embedder] = None,
) -> None:
    emb = embedder or HashEmbedder(dim=qdrant.dim)
    batch: List[Tuple[str, List[float], Dict[str, Any]]] = []
    chunk_list: List[Chunk] = []
    total = 0

    for chunk in chunks:
        chunk_list.append(chunk)
        repo.add_chunk(chunk)
        if len(chunk_list) >= batch_size:
            texts = [c.text for c in chunk_list]
            vectors = emb.embed_batch(texts)
            for c, vec in zip(chunk_list, vectors):
                batch.append((c.chunk_id, vec, _build_payload(c, emb.model_name)))
            qdrant.upsert(batch)
            total += len(batch)
            logger.debug("Upserted batch of %d points (total: %d)", len(batch), total)
            batch = []
            chunk_list = []

    if chunk_list:
        texts = [c.text for c in chunk_list]
        vectors = emb.embed_batch(texts)
        for c, vec in zip(chunk_list, vectors):
            batch.append((c.chunk_id, vec, _build_payload(c, emb.model_name)))
        qdrant.upsert(batch)
        total += len(batch)
        logger.debug("Upserted final batch of %d points (total: %d)", len(batch), total)


def _build_payload(chunk: Chunk, embedding_model: str = "hash-64") -> Dict[str, Any]:
    ingested_at = chunk.metadata.get("ingested_at")
    doc_updated_at = chunk.metadata.get("doc_updated_at")
    acl_hash = chunk.metadata.get("acl_hash") or "public"
    return {
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
        "tags": chunk.metadata.get("tags"),
        "embedding_model": embedding_model,
        "text": chunk.text,
    }
