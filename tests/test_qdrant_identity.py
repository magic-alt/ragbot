from __future__ import annotations

import uuid

from services.api.app.retrieval.embedder import HashEmbedder
from services.api.app.retrieval.qdrant import (
    InMemoryQdrant,
    normalize_qdrant_point_id,
    point_id_for_chunk,
)
from services.api.app.retrieval.service import Retriever
from services.api.app.storage.models import Chunk
from services.api.app.storage.repo import InMemoryRepo
from services.worker.jobs.embed_and_upsert import embed_and_upsert


def _chunk(chunk_id: str = "logical:chunk/not-a-qdrant-id") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-identity",
        tenant_id="tenant-identity",
        chunk_index=0,
        text="deterministic vector identity retrieval marker",
        checksum="checksum-identity",
        metadata={
            "source_type": "pdf",
            "tags": ["identity"],
            "acl_hash": "public",
            "version": "1.0",
        },
    )


def test_point_id_for_chunk_is_stable_valid_uuid():
    first = point_id_for_chunk("logical:chunk/not-a-qdrant-id")
    second = point_id_for_chunk("logical:chunk/not-a-qdrant-id")
    assert first == second
    assert str(uuid.UUID(first)) == first


def test_normalize_qdrant_point_id_repairs_legacy_arbitrary_string():
    repaired = normalize_qdrant_point_id("legacy:invalid/vector-id", "logical-chunk")
    assert repaired == point_id_for_chunk("logical-chunk")
    assert normalize_qdrant_point_id("123", "logical-chunk") == "123"


def test_embed_upsert_keeps_logical_id_and_uses_point_uuid():
    repo = InMemoryRepo()
    qdrant = InMemoryQdrant(dim=32)
    embedder = HashEmbedder(dim=32)
    chunk = _chunk()

    embed_and_upsert(repo, qdrant, [chunk], embedder=embedder)

    assert repo.get_chunk(chunk.chunk_id) is chunk
    assert chunk.qdrant_point_id == point_id_for_chunk(chunk.chunk_id)
    assert str(uuid.UUID(chunk.qdrant_point_id)) == chunk.qdrant_point_id

    hits = qdrant.search(
        embedder.embed(chunk.text),
        {"tenant_id": chunk.tenant_id, "security_scope": ["public"]},
        top_k=1,
    )
    assert hits[0][0] == chunk.qdrant_point_id
    assert hits[0][2]["chunk_id"] == chunk.chunk_id


def test_retriever_fuses_by_logical_chunk_id_not_point_uuid():
    repo = InMemoryRepo()
    qdrant = InMemoryQdrant(dim=32)
    embedder = HashEmbedder(dim=32)
    chunk = _chunk("logical-rag-chunk-001")
    embed_and_upsert(repo, qdrant, [chunk], embedder=embedder)

    results = Retriever(repo, qdrant, embedder=embedder).retrieve(
        "deterministic vector identity retrieval marker",
        {"tenant_id": chunk.tenant_id, "security_scope": ["public"]},
        top_k=5,
    )

    assert results
    assert results[0].chunk_id == chunk.chunk_id
    assert results[0].doc_id == chunk.doc_id
    assert results[0].metadata["qdrant_point_id"] == chunk.qdrant_point_id
