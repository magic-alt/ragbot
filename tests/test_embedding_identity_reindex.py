from __future__ import annotations

from services.api.app.storage.models import Chunk, Source
from services.worker.pipeline import _normalize_chunk_metadata, _reuse_unchanged_chunks


def _chunk(chunk_id: str, model: str, dimension: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-source-1",
        tenant_id="engineering",
        chunk_index=0,
        text="FP8 mixed precision reduces graphics memory usage",
        path="book.pdf",
        checksum="same-content",
        metadata={
            "source_type": "pdf",
            "tags": [],
            "acl_hash": "public",
            "version": "1.0",
            "lexical_version": 2,
            "embedding_model": model,
            "embedding_dimension": dimension,
        },
    )


def test_embedding_model_change_forces_revectorization():
    previous = _chunk("old", "text-embedding-3-small", 1536)
    candidate = _chunk("candidate", "qwen3-embedding:0.6b", 1024)

    current, to_write, reused = _reuse_unchanged_chunks([candidate], [previous])

    assert current == [candidate]
    assert to_write == [candidate]
    assert reused == 0
    assert candidate.chunk_id == "candidate"


def test_same_embedding_identity_can_reuse_unchanged_chunk():
    previous = _chunk("old", "qwen3-embedding:0.6b", 1024)
    candidate = _chunk("candidate", "qwen3-embedding:0.6b", 1024)

    current, to_write, reused = _reuse_unchanged_chunks([candidate], [previous])

    assert current == [candidate]
    assert to_write == []
    assert reused == 1
    assert candidate.chunk_id == "old"


def test_normalization_stamps_embedding_identity_on_candidate_chunks():
    source = Source(
        source_id="source-1",
        tenant_id="engineering",
        source_type="pdf",
        name="book",
        config={"path": "book.pdf", "version": "1.0"},
    )
    candidate = Chunk(
        chunk_id="candidate",
        doc_id="doc-source-1",
        tenant_id="engineering",
        chunk_index=0,
        text="content",
        metadata={"acl_hash": "public"},
    )

    _normalize_chunk_metadata(
        source,
        [candidate],
        "2026-09-04T00:00:00+00:00",
        embedding_model="qwen3-embedding:0.6b",
        embedding_dimension=1024,
    )

    assert candidate.metadata["embedding_model"] == "qwen3-embedding:0.6b"
    assert candidate.metadata["embedding_dimension"] == 1024
