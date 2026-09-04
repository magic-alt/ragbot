from __future__ import annotations

import threading

from services.api.app.retrieval.embedder import HashEmbedder
from services.api.app.retrieval.service import Retriever, build_citation
from services.api.app.storage.models import Chunk


class _ParallelRepo:
    def __init__(self, chunk: Chunk, barrier: threading.Barrier) -> None:
        self.chunk = chunk
        self.barrier = barrier
        self.get_chunk_calls = 0

    def search_chunks_fts(self, query, filters, top_k):
        self.barrier.wait(timeout=2)
        return [(self.chunk, 1.0)]

    def get_chunk(self, chunk_id):
        self.get_chunk_calls += 1
        raise AssertionError("retrieval should reuse first-stage evidence instead of N+1 get_chunk reads")


class _ParallelQdrant:
    dim = 64

    def __init__(self, chunk: Chunk, barrier: threading.Barrier) -> None:
        self.chunk = chunk
        self.barrier = barrier

    def search(self, query_vector, filters, top_k):
        self.barrier.wait(timeout=2)
        return [
            (
                "point-1",
                0.9,
                {
                    "chunk_id": self.chunk.chunk_id,
                    "doc_id": self.chunk.doc_id,
                    "chunk_index": self.chunk.chunk_index,
                    "path": self.chunk.path,
                    "page": self.chunk.page,
                    "text": self.chunk.text,
                },
            )
        ]


class _Reranker:
    enabled = True

    def rerank(self, query, documents, top_k):
        assert documents == ["parallel retrieval evidence"]
        return [(0, 0.99)]


def test_hybrid_fanout_is_parallel_and_rerank_reuses_candidates() -> None:
    barrier = threading.Barrier(2)
    chunk = Chunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        tenant_id="tenant",
        chunk_index=4,
        text="parallel retrieval evidence",
        path="paper.pdf",
        page=7,
        metadata={"source_type": "pdf"},
    )
    repo = _ParallelRepo(chunk, barrier)
    retriever = Retriever(
        repo,
        _ParallelQdrant(chunk, barrier),
        embedder=HashEmbedder(64),
        reranker=_Reranker(),
    )

    results = retriever.retrieve("parallel retrieval evidence", {}, top_k=5)

    assert len(results) == 1
    assert repo.get_chunk_calls == 0
    assert results[0].metadata["_retrieval"]["context"]["parallel_fanout"] is True
    assert results[0].citations == ["doc-1:paper.pdf:page=7:chunk=4"]


def test_page_aware_citation_falls_back_for_non_pdf_chunks() -> None:
    page_chunk = Chunk(
        chunk_id="p",
        doc_id="doc",
        tenant_id="tenant",
        chunk_index=2,
        text="x",
        path="paper.pdf",
        page=9,
    )
    text_chunk = Chunk(
        chunk_id="t",
        doc_id="doc",
        tenant_id="tenant",
        chunk_index=3,
        text="x",
        path="notes.md",
    )

    assert build_citation(page_chunk) == "doc:paper.pdf:page=9:chunk=2"
    assert build_citation(text_chunk) == "doc:notes.md:3"
