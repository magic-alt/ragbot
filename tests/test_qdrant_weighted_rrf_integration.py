from __future__ import annotations

import os
import uuid

import pytest

from services.api.app.retrieval.qdrant import QdrantClientAdapter
from services.api.app.retrieval.qdrant_hybrid import wrap_qdrant_hybrid
from services.api.app.retrieval.sparse import DeterministicSparseEncoder, SparseVector


pytestmark = pytest.mark.skipif(
    not os.getenv("QDRANT_TEST_URL"),
    reason="QDRANT_TEST_URL not configured",
)


def test_real_qdrant_weighted_rrf_changes_winner_without_reindexing() -> None:
    suffix = uuid.uuid4().hex[:10]
    bootstrap = f"weighted_base_{suffix}"
    hybrid = f"weighted_hybrid_{suffix}"
    raw = QdrantClientAdapter(
        url=os.environ["QDRANT_TEST_URL"],
        api_key=None,
        collection_name=bootstrap,
        dim=2,
        alias_name=None,
    )
    store = wrap_qdrant_hybrid(raw, DeterministicSparseEncoder())
    try:
        store.create_hybrid_collection(
            hybrid,
            dim=2,
            distance="cosine",
            dense_name="dense",
            sparse_name="sparse",
            sparse_modifier="idf",
        )
        store.upsert_hybrid_to_collection(
            hybrid,
            [
                (
                    "00000000-0000-0000-0000-000000000001",
                    [1.0, 0.0],
                    SparseVector(indices=[8], values=[1.0]),
                    {"chunk_id": "dense-winner", "tenant_id": "tenant-a", "text": "dense winner"},
                ),
                (
                    "00000000-0000-0000-0000-000000000002",
                    [0.0, 1.0],
                    SparseVector(indices=[7], values=[1.0]),
                    {"chunk_id": "sparse-winner", "tenant_id": "tenant-a", "text": "sparse winner"},
                ),
            ],
            dense_name="dense",
            sparse_name="sparse",
        )

        dense_first = store.native_hybrid_search(
            [1.0, 0.0],
            SparseVector(indices=[7], values=[1.0]),
            {"tenant_id": "tenant-a"},
            1,
            collection_name=hybrid,
            dense_name="dense",
            sparse_name="sparse",
            prefetch_limit=1,
            rrf_weights=[4.0, 1.0],
            rrf_k=2,
        )
        sparse_first = store.native_hybrid_search(
            [1.0, 0.0],
            SparseVector(indices=[7], values=[1.0]),
            {"tenant_id": "tenant-a"},
            1,
            collection_name=hybrid,
            dense_name="dense",
            sparse_name="sparse",
            prefetch_limit=1,
            rrf_weights=[1.0, 4.0],
            rrf_k=2,
        )
        assert dense_first[0][2]["chunk_id"] == "dense-winner"
        assert sparse_first[0][2]["chunk_id"] == "sparse-winner"

        # No collection rebuild or alias operation occurs between the two
        # queries: only the RRF fusion contract changes.
        schema = store.collection_vector_schema(hybrid)
        assert schema["dense"]["dense"]["dimension"] == 2
        assert "sparse" in schema["sparse"]
        assert store.count_collection(hybrid) == 2
    finally:
        try:
            store.delete_collection(hybrid)
        except Exception:
            pass
        try:
            store.delete_collection(bootstrap)
        except Exception:
            pass
        store.close()
