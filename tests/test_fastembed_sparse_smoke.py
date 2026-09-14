from __future__ import annotations

import pytest

from services.api.app.retrieval.sparse import FastEmbedSparseEncoder, SparseEmbeddingSpec


fastembed = pytest.importorskip("fastembed")


def test_fastembed_qdrant_bm25_query_and_passage_contract() -> None:
    spec = SparseEmbeddingSpec(
        provider_id="fastembed",
        model="Qdrant/bm25",
        vector_name="sparse",
        modifier="idf",
    )
    encoder = FastEmbedSparseEncoder(spec, batch_size=2)

    query = encoder.embed_query("ethercat distributed clocks synchronization")
    passages = encoder.embed_documents(
        [
            "EtherCAT distributed clocks synchronize servo axes.",
            "A holding brake keeps a robot joint stationary when power is removed.",
        ]
    )

    assert encoder.contract_id == spec.contract_id
    assert query.indices
    assert len(query.indices) == len(query.values)
    assert len(passages) == 2
    assert all(item.indices for item in passages)
    assert all(len(item.indices) == len(item.values) for item in passages)
