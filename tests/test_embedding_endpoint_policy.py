from __future__ import annotations

from services.api.app.retrieval.embedder import APIEmbedder, HashEmbedder, build_embedder


def _clear_embedding_auth(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_ALLOW_ANONYMOUS", raising=False)
    monkeypatch.delenv("QDRANT_DIM", raising=False)


def test_hosted_embedding_endpoint_without_key_keeps_development_fallback(monkeypatch):
    _clear_embedding_auth(monkeypatch)
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.openai.com")

    embedder = build_embedder()

    assert isinstance(embedder, HashEmbedder)
    assert embedder.dimension == 1536


def test_local_ollama_endpoint_is_allowed_without_fake_key(monkeypatch):
    _clear_embedding_auth(monkeypatch)
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://127.0.0.1:11434")

    embedder = build_embedder()

    assert isinstance(embedder, APIEmbedder)
    assert embedder.dimension == 1024


def test_host_docker_internal_is_treated_as_local_embedding_endpoint(monkeypatch):
    _clear_embedding_auth(monkeypatch)
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://host.docker.internal:11434")

    assert isinstance(build_embedder(), APIEmbedder)


def test_remote_anonymous_endpoint_requires_explicit_opt_in(monkeypatch):
    _clear_embedding_auth(monkeypatch)
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://embeddings.internal.example:8080")

    assert isinstance(build_embedder(), HashEmbedder)

    monkeypatch.setenv("EMBEDDING_ALLOW_ANONYMOUS", "true")
    assert isinstance(build_embedder(), APIEmbedder)
