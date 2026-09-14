from __future__ import annotations

import asyncio
import time

import pytest

from services.api.app.retrieval.contracts import (
    RetrievalDeadlineExceeded,
    RetrievalPlan,
    RetrievalRequest,
    UnsupportedRetrievalPlan,
    resolve_retrieval_plan,
)
from services.api.app.retrieval.embedder import HashEmbedder
from services.api.app.retrieval.service import Retriever
from services.api.app.storage.models import Chunk


class _Repo:
    def __init__(self, chunk: Chunk, delay: float = 0.0) -> None:
        self.chunk = chunk
        self.delay = delay

    def search_chunks_fts(self, query, filters, top_k):
        if self.delay:
            time.sleep(self.delay)
        return [(self.chunk, 1.0)]


class _Vector:
    dim = 64

    def __init__(self, chunk: Chunk, delay: float = 0.0) -> None:
        self.chunk = chunk
        self.delay = delay

    def search(self, query_vector, filters, top_k):
        if self.delay:
            time.sleep(self.delay)
        return [
            (
                "point-1",
                0.9,
                {
                    "chunk_id": self.chunk.chunk_id,
                    "doc_id": self.chunk.doc_id,
                    "chunk_index": self.chunk.chunk_index,
                    "text": self.chunk.text,
                    "source_id": "source-1",
                },
            )
        ]


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        tenant_id="tenant-a",
        chunk_index=0,
        text="servo ethercat retrieval evidence",
        metadata={"source_type": "pdf"},
    )


def test_plan_contract_keeps_legacy_modes() -> None:
    assert resolve_retrieval_plan("dense") is RetrievalPlan.DENSE
    assert resolve_retrieval_plan(None, legacy_mode="vector") is RetrievalPlan.DENSE
    assert resolve_retrieval_plan(None, legacy_mode="hybrid") is RetrievalPlan.HYBRID_RRF
    assert resolve_retrieval_plan("qdrant_dense_sparse") is RetrievalPlan.QDRANT_DENSE_SPARSE


def test_async_query_returns_stage_trace_and_candidate_sources() -> None:
    chunk = _chunk()
    retriever = Retriever(_Repo(chunk), _Vector(chunk), embedder=HashEmbedder(64))
    try:
        response = asyncio.run(
            retriever.query(
                RetrievalRequest(
                    query="servo ethercat",
                    filters={"tenant_id": "tenant-a"},
                    top_k=5,
                    plan=RetrievalPlan.HYBRID_RRF,
                    rerank=False,
                    deadline_ms=1000,
                )
            )
        )
    finally:
        retriever.close()

    assert response.chunks
    assert response.trace.plan == "hybrid_rrf"
    assert response.trace.candidate_counts["dense"] == 1
    assert response.trace.candidate_counts["lexical"] == 1
    assert "candidates.parallel" in response.trace.stage_ms
    evidence_trace = response.chunks[0].metadata["_retrieval"]
    assert evidence_trace["dense"]["rank"] == 1
    assert evidence_trace["lexical"]["rank"] == 1
    assert evidence_trace["fusion_score"] is not None
    assert evidence_trace["final_rank"] == 1


def test_deadline_cancels_parallel_retrieval_and_carries_trace() -> None:
    chunk = _chunk()
    retriever = Retriever(
        _Repo(chunk, delay=0.2),
        _Vector(chunk, delay=0.2),
        embedder=HashEmbedder(64),
    )
    try:
        with pytest.raises(RetrievalDeadlineExceeded) as excinfo:
            asyncio.run(
                retriever.query(
                    RetrievalRequest(
                        query="deadline evidence",
                        filters={"tenant_id": "tenant-a"},
                        plan=RetrievalPlan.HYBRID_RRF,
                        deadline_ms=20,
                        rerank=False,
                    )
                )
            )
    finally:
        retriever.close()

    assert excinfo.value.trace.timed_out is True
    assert excinfo.value.trace.error_stage is not None


def test_experimental_native_sparse_plan_fails_fast_without_index_capability() -> None:
    chunk = _chunk()
    retriever = Retriever(_Repo(chunk), _Vector(chunk), embedder=HashEmbedder(64))
    try:
        with pytest.raises(UnsupportedRetrievalPlan, match="named dense\+sparse"):
            asyncio.run(
                retriever.query(
                    RetrievalRequest(
                        query="sparse hybrid",
                        filters={"tenant_id": "tenant-a"},
                        plan=RetrievalPlan.QDRANT_DENSE_SPARSE,
                        rerank=False,
                    )
                )
            )
    finally:
        retriever.close()
