from __future__ import annotations

import os
import uuid

import pytest

from services.api.app.retrieval.embedder import HashEmbedder
from services.api.app.retrieval.embedding_router import EmbeddingRouter
from services.api.app.retrieval.qdrant import QdrantClientAdapter, point_id_for_chunk
from services.api.app.retrieval.qdrant_hybrid import wrap_qdrant_hybrid
from services.api.app.retrieval.service import Retriever
from services.api.app.retrieval.sparse import DeterministicSparseEncoder
from services.api.app.retrieval.sparse_index_lifecycle import SparseIndexLifecycleService
from services.api.app.storage.index_support import ensure_index_repository
from services.api.app.storage.models import Chunk
from services.api.app.storage.repo import InMemoryRepo
from services.worker.jobs.embed_and_upsert import _build_payload


pytestmark = pytest.mark.skipif(
    not os.getenv("QDRANT_TEST_URL"),
    reason="QDRANT_TEST_URL not configured",
)


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        tenant_id="tenant-a",
        chunk_index=0,
        text=text,
        checksum=f"checksum-{chunk_id}",
        metadata={
            "source_type": "pdf",
            "acl_hash": "public",
            "tags": ["dense-sparse"],
        },
    )


def test_real_qdrant_named_dense_sparse_prefetch_rrf_and_active_incremental_upsert() -> None:
    suffix = uuid.uuid4().hex[:10]
    collection = f"ds_base_{suffix}"
    alias = f"ds_active_{suffix}"
    dense = HashEmbedder(dim=32)
    sparse = DeterministicSparseEncoder(vector_name="sparse")
    raw = QdrantClientAdapter(
        url=os.environ["QDRANT_TEST_URL"],
        api_key=None,
        collection_name=collection,
        dim=dense.dimension,
        alias_name=alias,
    )
    qdrant = wrap_qdrant_hybrid(raw, sparse)
    repo = ensure_index_repository(InMemoryRepo())
    router = EmbeddingRouter([dense])
    chunks = [
        _chunk("servo", "servo motor current loop tuning"),
        _chunk("ethercat", "ethercat distributed clocks fieldbus synchronization"),
    ]
    repo.add_chunks(chunks)
    points = []
    for chunk in chunks:
        vector = dense.embed(chunk.text)
        point_id = point_id_for_chunk(chunk.chunk_id)
        chunk.qdrant_point_id = point_id
        chunk.metadata["embedding_contract_id"] = dense.contract_id
        chunk.metadata["embedding_dimension"] = dense.dimension
        payload = _build_payload(chunk, dense.model_name)
        payload["embedding_contract_id"] = dense.contract_id
        points.append((point_id, vector, payload))
    qdrant.upsert(points)

    lifecycle = SparseIndexLifecycleService(
        repo,
        qdrant,
        router,
        alias_name=alias,
        sparse_encoder=sparse,
    )
    baseline = lifecycle.bootstrap_current(dense)
    candidate = lifecycle.create_candidate(
        dense.contract_id,
        sparse_contract_id=sparse.contract_id,
    )
    candidate = lifecycle.build(candidate.index_version_id, batch_size=1)
    assert candidate.status == "validating"
    schema = qdrant.collection_vector_schema(candidate.physical_collection)
    assert schema["dense"]["dense"]["dimension"] == dense.dimension
    assert "sparse" in schema["sparse"]

    # Verify tenant eligibility is applied inside each prefetch candidate branch,
    # before RRF spends its candidate budget. The tenant-b point is an exact
    # match for this query and would otherwise consume a prefetch_limit=1 slot.
    foreign = _chunk("foreign", "foreign secret exact candidate")
    foreign.tenant_id = "tenant-b"
    foreign_payload = _build_payload(foreign, dense.model_name)
    foreign_payload["embedding_contract_id"] = dense.contract_id
    foreign_payload["sparse_contract_id"] = sparse.contract_id
    qdrant.upsert_hybrid_to_collection(
        candidate.physical_collection,
        [
            (
                point_id_for_chunk(foreign.chunk_id),
                dense.embed(foreign.text),
                sparse.embed_query(foreign.text),
                foreign_payload,
            )
        ],
        dense_name="dense",
        sparse_name="sparse",
    )
    prefiltered = qdrant.native_hybrid_search(
        dense.embed(foreign.text),
        sparse.embed_query(foreign.text),
        {"tenant_id": "tenant-a"},
        1,
        collection_name=candidate.physical_collection,
        dense_name="dense",
        sparse_name="sparse",
        prefetch_limit=1,
    )
    assert prefiltered
    assert all((payload or {}).get("tenant_id") == "tenant-a" for _pid, _score, payload in prefiltered)

    retriever = Retriever(
        repo,
        qdrant,
        embedder=dense,
        reranker=None,
        sparse_encoder=sparse,
    )
    try:
        candidate_hits = retriever.retrieve(
            "ethercat fieldbus synchronization",
            {"tenant_id": "tenant-a"},
            top_k=2,
            plan="qdrant_dense_sparse",
            rerank=False,
            index_version_id=candidate.index_version_id,
        )
        assert candidate_hits
        assert candidate_hits[0].chunk_id == "ethercat"
        context = candidate_hits[0].metadata["_retrieval"]["context"]
        assert context["index_version_id"] == candidate.index_version_id
        assert context["representation_contracts"]["sparse"] == sparse.contract_id
        assert qdrant.active_collection_name() == baseline.physical_collection

        lifecycle.mark_ready(
            candidate.index_version_id,
            {"integration_smoke": True, "sparse_contract_id": sparse.contract_id},
            approved=True,
        )
        lifecycle.activate(candidate.index_version_id, retention_seconds=3600)
        assert qdrant.active_collection_name() == candidate.physical_collection

        incremental = _chunk("brake", "emergency brake safety holding torque")
        incremental.metadata["embedding_contract_id"] = dense.contract_id
        incremental.metadata["embedding_dimension"] = dense.dimension
        payload = _build_payload(incremental, dense.model_name)
        payload["embedding_contract_id"] = dense.contract_id
        qdrant.upsert(
            [(point_id_for_chunk(incremental.chunk_id), dense.embed(incremental.text), payload)]
        )
        active_hits = retriever.retrieve(
            "emergency brake holding torque",
            {"tenant_id": "tenant-a"},
            top_k=3,
            plan="qdrant_dense_sparse",
            rerank=False,
        )
        assert any(item.chunk_id == "brake" for item in active_hits)
    finally:
        retriever.close()
        # The Qdrant service is ephemeral in CI. Restore the baseline alias so
        # the candidate is never left active even when the test is run locally.
        try:
            qdrant.switch_alias(baseline.physical_collection)
        except Exception:
            pass
        qdrant.close()
