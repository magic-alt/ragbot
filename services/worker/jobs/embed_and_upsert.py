from __future__ import annotations

from typing import Iterable

from ...api.app.retrieval.qdrant import embed_text, to_epoch
from ...api.app.storage.models import Chunk
from ...api.app.storage.repo import InMemoryRepo


def embed_and_upsert(repo: InMemoryRepo, qdrant: object, chunks: Iterable[Chunk]) -> None:
    points = []
    for chunk in chunks:
        vector = embed_text(chunk.text, qdrant.dim)
        ingested_at = chunk.metadata.get("ingested_at")
        doc_updated_at = chunk.metadata.get("doc_updated_at")
        acl_hash = chunk.metadata.get("acl_hash") or "public"
        payload = {
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
        points.append((chunk.chunk_id, vector, payload))
        repo.add_chunk(chunk)
    qdrant.upsert(points)

