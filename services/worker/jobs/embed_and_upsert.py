from __future__ import annotations

import logging
from typing import Iterable, List, Tuple, Any, Dict

from services.api.app.retrieval.qdrant import embed_text, to_epoch
from services.api.app.storage.models import Chunk
from services.api.app.storage.repo import InMemoryRepo

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100


def embed_and_upsert(
    repo: InMemoryRepo,
    qdrant: object,
    chunks: Iterable[Chunk],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    batch: List[Tuple[str, List[float], Dict[str, Any]]] = []
    total = 0
    for chunk in chunks:
        vector = embed_text(chunk.text, qdrant.dim)
        payload = _build_payload(chunk)
        batch.append((chunk.chunk_id, vector, payload))
        repo.add_chunk(chunk)
        if len(batch) >= batch_size:
            qdrant.upsert(batch)
            total += len(batch)
            logger.debug("Upserted batch of %d points (total: %d)", len(batch), total)
            batch = []
    if batch:
        qdrant.upsert(batch)
        total += len(batch)
        logger.debug("Upserted final batch of %d points (total: %d)", len(batch), total)


def _build_payload(chunk: Chunk) -> Dict[str, Any]:
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
        "embedding_model": "hash-64",
        "text": chunk.text,
    }
