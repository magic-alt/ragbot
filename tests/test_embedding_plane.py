from __future__ import annotations

import asyncio

import pytest

from services.api.app.retrieval.embedder import APIEmbedder, MemoryEmbeddingCache
from services.api.app.retrieval.embedding_contract import EmbeddingSpec
from services.api.app.storage.models import Chunk
from services.worker.pipeline import _reuse_key


class FakeTransport:
    def __init__(self) -> None:
        self.sync_inputs: list[list[str]] = []
        self.async_inputs: list[list[str]] = []

    def post_json(self, url, *, headers, payload):
        inputs = list(payload["input"])
        self.sync_inputs.append(inputs)
        return _response(inputs), 1, 12.5

    async def apost_json(self, url, *, headers, payload):
        inputs = list(payload["input"])
        self.async_inputs.append(inputs)
        await asyncio.sleep(0)
        return _response(inputs), 0, 5.0

    def close(self):
        return None

    async def aclose(self):
        return None


def _response(inputs: list[str]) -> dict:
    return {
        "data": [
            {"index": index, "embedding": [float(index + 1), float(len(text))]}
            for index, text in enumerate(inputs)
        ]
    }


def test_contract_id_changes_for_representation_affecting_settings():
    baseline = EmbeddingSpec(provider_id="openai-compatible", model="m", dimension=2)
    same = EmbeddingSpec(provider_id="openai-compatible", model="m", dimension=2)
    query_changed = EmbeddingSpec(provider_id="openai-compatible", model="m", dimension=2, query_instruction="retrieve")
    doc_changed = EmbeddingSpec(provider_id="openai-compatible", model="m", dimension=2, document_instruction="passage")
    normalized = EmbeddingSpec(provider_id="openai-compatible", model="m", dimension=2, normalize=True)
    revision = EmbeddingSpec(provider_id="openai-compatible", model="m", dimension=2, revision="2026-09")

    assert baseline.contract_id == same.contract_id
    assert len({baseline.contract_id, query_changed.contract_id, doc_changed.contract_id, normalized.contract_id, revision.contract_id}) == 5


def test_query_and_document_instruction_are_separate_contract_inputs():
    transport = FakeTransport()
    embedder = APIEmbedder(
        api_key="key",
        base_url="https://embedding.test",
        model="model",
        dimension=2,
        query_instruction="retrieve relevant evidence",
        document_instruction="represent knowledge passage",
        transport=transport,  # type: ignore[arg-type]
    )
    embedder.embed_query("question")
    assert transport.sync_inputs[-1] == ["Instruct: retrieve relevant evidence\nQuery:question"]
    embedder.embed_batch(["document"])
    assert transport.sync_inputs[-1] == ["Instruct: represent knowledge passage\nDocument:document"]


def test_adaptive_batching_respects_item_and_byte_limits():
    transport = FakeTransport()
    embedder = APIEmbedder(
        api_key="key",
        base_url="https://embedding.test",
        model="model",
        dimension=2,
        batch_size=2,
        max_batch_bytes=7,
        transport=transport,  # type: ignore[arg-type]
    )
    vectors = embedder.embed_batch(["aaa", "bbb", "cccc"])
    assert len(vectors) == 3
    assert transport.sync_inputs == [["aaa", "bbb"], ["cccc"]]


def test_async_embedding_preserves_order_across_batches():
    transport = FakeTransport()
    embedder = APIEmbedder(
        api_key="key",
        base_url="https://embedding.test",
        model="model",
        dimension=2,
        batch_size=2,
        transport=transport,  # type: ignore[arg-type]
    )

    vectors = asyncio.run(embedder.aembed_documents(["a", "bb", "ccc"]))
    assert len(vectors) == 3
    assert len(transport.async_inputs) == 2


def test_content_cache_is_scoped_by_contract_and_role():
    cache = MemoryEmbeddingCache(max_entries=8)
    transport = FakeTransport()
    embedder = APIEmbedder(
        api_key="key",
        base_url="https://embedding.test",
        model="model",
        dimension=2,
        query_instruction="query",
        cache=cache,
        transport=transport,  # type: ignore[arg-type]
    )
    first = embedder.embed_query("same")
    second = embedder.embed_query("same")
    assert first == second
    assert len(transport.sync_inputs) == 1
    assert embedder.diagnostics()["metrics"]["cache_hits"] == 1

    # Document role gets a distinct cache key even for the same raw text.
    embedder.embed_batch(["same"])
    assert len(transport.sync_inputs) == 2


def test_normalization_is_explicit_and_defaults_to_previous_provider_behavior():
    transport = FakeTransport()
    baseline = APIEmbedder(
        api_key="key",
        base_url="https://embedding.test",
        model="model",
        dimension=2,
        transport=transport,  # type: ignore[arg-type]
    )
    raw = baseline.embed_batch(["abc"])[0]
    assert raw == [1.0, 3.0]
    assert baseline.spec.normalize is False

    normalized = APIEmbedder(
        api_key="key",
        base_url="https://embedding.test",
        model="model",
        dimension=2,
        normalize=True,
        transport=FakeTransport(),  # type: ignore[arg-type]
    ).embed_batch(["abc"])[0]
    assert sum(value * value for value in normalized) == pytest.approx(1.0)


def test_reuse_key_includes_embedding_contract_id():
    common = dict(
        chunk_id="c",
        doc_id="d",
        tenant_id="t",
        chunk_index=0,
        text="text",
        checksum="sum",
    )
    a = Chunk(**common, metadata={"embedding_model": "m", "embedding_dimension": 2, "embedding_contract_id": "emb-a"})
    b = Chunk(**common, metadata={"embedding_model": "m", "embedding_dimension": 2, "embedding_contract_id": "emb-b"})
    assert _reuse_key(a) != _reuse_key(b)


def test_diagnostics_expose_contract_and_transport_metrics_without_credentials():
    transport = FakeTransport()
    embedder = APIEmbedder(
        api_key="super-secret",
        base_url="https://embedding.test",
        model="model",
        dimension=2,
        transport=transport,  # type: ignore[arg-type]
    )
    embedder.embed_batch(["abc"])
    diagnostics = embedder.diagnostics()
    assert diagnostics["contract_id"].startswith("emb-")
    assert diagnostics["metrics"]["requests"] == 1
    assert diagnostics["metrics"]["retries"] == 1
    assert "super-secret" not in repr(diagnostics)
