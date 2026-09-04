from __future__ import annotations

from typing import Any

import pytest

from services.api.app.retrieval.embedder import (
    APIEmbedder,
    build_embedder,
    default_query_instruction,
    model_dimension,
)
from services.api.app.retrieval.policy import (
    adaptive_fusion_policy,
    resolve_candidate_pool,
    validate_retrieval_mode,
)
from services.api.app.retrieval.service import Retriever
from services.api.app.storage.models import Chunk


class FakeRepo:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.iterations = 0

    def get_chunk(self, chunk_id: str):
        return self.chunks.get(chunk_id)

    def iter_chunks(self):
        self.iterations += 1
        return iter(self.chunks.values())


class FakeQdrant:
    def __init__(self, hits: list[tuple[str, float, dict[str, Any]]]) -> None:
        self.dim = 3
        self.hits = hits
        self.calls: list[int] = []

    def search(self, vector, filters, top_k):
        self.calls.append(top_k)
        return self.hits[:top_k]


class SemanticEmbedder:
    model_name = "semantic-test"
    dimension = 3

    def embed(self, text: str):
        return [1.0, 0.0, 0.0]

    def embed_query(self, text: str):
        return [1.0, 0.0, 0.0]

    def embed_batch(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class CapturingReranker:
    enabled = True

    def __init__(self) -> None:
        self.documents: list[str] = []

    def rerank(self, query: str, documents: list[str], top_k: int = 10):
        self.documents = list(documents)
        return [(index, 1.0 - index * 0.01) for index in range(min(top_k, len(documents)))]


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        tenant_id="engineering",
        chunk_index=int(chunk_id.removeprefix("c") or 0),
        text=text,
        metadata={"acl_hash": "public", "source_type": "pdf"},
    )


def _hits(chunks: list[Chunk]) -> list[tuple[str, float, dict[str, Any]]]:
    return [
        (
            f"point-{chunk.chunk_id}",
            0.95 - index * 0.05,
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "tenant_id": chunk.tenant_id,
            },
        )
        for index, chunk in enumerate(chunks)
    ]


def test_retrieval_modes_are_true_ablations():
    chunks = [_chunk("c1", "GPU mixed precision FP8"), _chunk("c2", "cache optimization")]
    repo = FakeRepo(chunks)
    qdrant = FakeQdrant(_hits(chunks))
    retriever = Retriever(repo, qdrant, embedder=SemanticEmbedder())

    vector = retriever.retrieve("GPU", {"tenant_id": "engineering"}, top_k=2, mode="vector")
    assert qdrant.calls == [40]
    assert repo.iterations == 0
    assert vector[0].metadata["_retrieval"]["vector"]["raw_score"] == pytest.approx(0.95)
    assert vector[0].metadata["_retrieval"]["lexical"] is None
    assert vector[0].metadata["_retrieval"]["context"]["fusion_method"] == "vector-only"

    qdrant.calls.clear()
    lexical = retriever.retrieve("GPU", {"tenant_id": "engineering"}, top_k=2, mode="lexical")
    assert qdrant.calls == []
    assert repo.iterations == 1
    assert lexical[0].metadata["_retrieval"]["vector"] is None
    assert lexical[0].metadata["_retrieval"]["lexical"]["raw_score"] > 0
    assert lexical[0].metadata["_retrieval"]["context"]["fusion_method"] == "lexical-only"


def test_cross_language_hybrid_caps_weak_lexical_authority():
    chunks = [
        _chunk("c1", "Distributed training uses GPU resources efficiently"),
        _chunk("c2", "FP8 mixed precision can reduce graphics memory usage on GPU"),
    ]
    repo = FakeRepo(chunks)
    qdrant = FakeQdrant(_hits(chunks))
    retriever = Retriever(repo, qdrant, embedder=SemanticEmbedder())

    results = retriever.retrieve(
        "运行大语言模型时如何降低 GPU 显存占用？",
        {"tenant_id": "engineering"},
        top_k=2,
        mode="hybrid",
    )
    context = results[0].metadata["_retrieval"]["context"]
    policy = context["fusion_policy"]
    assert context["fusion_method"] == "adaptive-rrf"
    assert policy["cross_language_lexical"] is True
    assert policy["vector_weight"] == pytest.approx(0.9)
    assert policy["lexical_weight"] == pytest.approx(0.1)
    assert policy["reason"] == "cross-language-semantic-first"


def test_candidate_pool_is_recall_budget_before_reranking():
    chunks = [_chunk(f"c{i}", f"GPU evidence candidate {i}") for i in range(1, 7)]
    repo = FakeRepo(chunks)
    qdrant = FakeQdrant(_hits(chunks))
    reranker = CapturingReranker()
    retriever = Retriever(repo, qdrant, embedder=SemanticEmbedder(), reranker=reranker)

    results = retriever.retrieve(
        "GPU evidence",
        {"tenant_id": "engineering"},
        top_k=2,
        mode="hybrid",
        candidate_pool=4,
    )
    assert qdrant.calls == [4]
    assert len(reranker.documents) == 4
    assert len(results) == 2
    context = results[0].metadata["_retrieval"]["context"]
    assert context["candidate_pool"] == 4
    assert context["reranker_candidate_count"] == 4
    assert context["reranker_enabled"] is True
    assert results[0].metadata["_retrieval"]["rerank_score"] == pytest.approx(1.0)


def test_reranker_can_be_disabled_for_clean_ablation():
    chunks = [_chunk(f"c{i}", f"GPU evidence candidate {i}") for i in range(1, 5)]
    repo = FakeRepo(chunks)
    qdrant = FakeQdrant(_hits(chunks))
    reranker = CapturingReranker()
    retriever = Retriever(repo, qdrant, embedder=SemanticEmbedder(), reranker=reranker)

    results = retriever.retrieve(
        "GPU evidence",
        {"tenant_id": "engineering"},
        top_k=2,
        mode="hybrid",
        candidate_pool=4,
        rerank=False,
    )

    assert reranker.documents == []
    context = results[0].metadata["_retrieval"]["context"]
    assert context["reranker_configured"] is True
    assert context["reranker_requested"] is False
    assert context["reranker_enabled"] is False
    assert context["reranker_candidate_count"] == 0
    assert results[0].metadata["_retrieval"]["rerank_score"] is None


def test_adaptive_policy_prefers_balanced_fusion_for_strong_lexical_overlap():
    lexical_chunk = _chunk("c1", "mixed precision FP8 GPU memory usage")
    policy = adaptive_fusion_policy(
        "mixed precision FP8 GPU memory usage",
        [("p1", 0.9, {"chunk_id": "c1"})],
        [(lexical_chunk, 1.0)],
        hash_fallback=False,
    )
    assert policy.lexical_confidence >= 0.65
    assert policy.vector_weight == pytest.approx(0.5)
    assert policy.lexical_weight == pytest.approx(0.5)


def test_retrieval_policy_validation_and_pool_bounds(monkeypatch):
    assert validate_retrieval_mode("VECTOR") == "vector"
    with pytest.raises(ValueError):
        validate_retrieval_mode("magic")

    monkeypatch.setenv("RAGBOT_RETRIEVAL_CANDIDATE_POOL", "12")
    assert resolve_candidate_pool(5) == 12
    assert resolve_candidate_pool(20, requested=5) == 20
    assert resolve_candidate_pool(5, requested=999) == 200


def test_qwen3_dimensions_and_local_keyless_build(monkeypatch):
    assert model_dimension("qwen3-embedding:0.6b") == 1024
    assert model_dimension("qwen3-embedding:0.6b-q8_0") == 1024
    assert model_dimension("Qwen/Qwen3-Embedding-4B") == 2560
    assert model_dimension("Qwen/Qwen3-Embedding-4B-GGUF") == 2560
    assert model_dimension("qwen3-embedding:8b") == 4096
    assert "retrieve relevant passages" in default_query_instruction("qwen3-embedding:0.6b")

    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("QDRANT_DIM", raising=False)
    embedder = build_embedder()
    assert isinstance(embedder, APIEmbedder)
    assert embedder.dimension == 1024
    assert embedder.query_instruction


def test_qwen_query_instruction_is_applied_only_to_query(monkeypatch):
    calls: list[dict[str, Any]] = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        return Response()

    monkeypatch.setattr("services.api.app.retrieval.embedder.requests.post", fake_post)
    embedder = APIEmbedder(
        api_key="",
        base_url="http://127.0.0.1:11434",
        model="qwen3-embedding:0.6b",
        dimension=2,
    )

    embedder.embed_query("How do I reduce VRAM usage?")
    assert calls[-1]["input"][0].startswith("Instruct: ")
    assert "\nQuery:How do I reduce VRAM usage?" in calls[-1]["input"][0]

    embedder.embed_batch(["FP8 mixed precision reduces graphics memory usage"])
    assert calls[-1]["input"][0] == "FP8 mixed precision reduces graphics memory usage"
