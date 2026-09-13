from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Mapping, Optional

from services.platform import TypedRegistry


RuntimeBuilder = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class RuntimeFactorySpec:
    kind: str
    provider_id: str
    builder: RuntimeBuilder
    capabilities: frozenset[str] = frozenset()
    version: str = "1"
    optional_dependency: Optional[str] = None

    @property
    def component_id(self) -> str:
        return f"{self.kind}:{self.provider_id}"

    def public_metadata(self) -> dict[str, object]:
        return {
            "id": self.component_id,
            "kind": self.kind,
            "provider_id": self.provider_id,
            "version": self.version,
            "capabilities": sorted(self.capabilities),
            "optional_dependency": self.optional_dependency,
        }


class RuntimeComponentRegistry:
    ENTRYPOINT_GROUP = "ragbot.runtime_components"

    def __init__(self) -> None:
        self._registry: TypedRegistry[RuntimeFactorySpec] = TypedRegistry(
            kind="runtime",
            entrypoint_group=self.ENTRYPOINT_GROUP,
        )

    def register(self, spec: RuntimeFactorySpec, *, replace: bool = False) -> RuntimeFactorySpec:
        self._registry.register(spec, replace=replace)
        return spec

    def get(self, kind: str, provider_id: str) -> RuntimeFactorySpec:
        return self._registry.get(f"{kind}:{provider_id}")

    def build(self, kind: str, provider_id: str, config: Optional[Mapping[str, Any]] = None) -> Any:
        return self.get(kind, provider_id).builder(dict(config or {}))

    def public_metadata(self) -> list[dict[str, object]]:
        return [spec.public_metadata() for spec in self._registry.values()]


@lru_cache(maxsize=1)
def runtime_component_registry() -> RuntimeComponentRegistry:
    registry = RuntimeComponentRegistry()
    for spec in _builtin_specs():
        registry.register(spec)
    registry.public_metadata()
    return registry


def _build_memory_repo(_config: Mapping[str, Any]):
    from services.api.app.storage.repo import InMemoryRepo
    return InMemoryRepo()


def _build_postgres_repo(config: Mapping[str, Any]):
    from services.api.app.storage.managed_pg_repo import ManagedPostgresRepo
    dsn = str(config.get("dsn") or "").strip()
    if not dsn:
        raise ValueError("repository:postgres requires dsn")
    return ManagedPostgresRepo(dsn=dsn)


def _build_memory_vector(config: Mapping[str, Any]):
    from services.api.app.retrieval.qdrant import InMemoryQdrant
    return InMemoryQdrant(dim=int(config["dim"]))


def _build_qdrant_vector(config: Mapping[str, Any]):
    from services.api.app.retrieval.qdrant import QdrantClientAdapter
    url = str(config.get("url") or "").strip()
    if not url:
        raise ValueError("vector:qdrant requires url")
    return QdrantClientAdapter(
        url=url,
        api_key=config.get("api_key"),
        collection_name=str(config.get("collection_name") or "rag_chunks"),
        dim=int(config["dim"]),
    )


def _build_hash_embedding(config: Mapping[str, Any]):
    from services.api.app.retrieval.embedder import HashEmbedder
    return HashEmbedder(dim=int(config["dimension"]))


def _build_openai_compatible_embedding(config: Mapping[str, Any]):
    from services.api.app.retrieval.embedder import build_embedder
    return build_embedder(dimension=int(config["dimension"]))


def _build_openai_llm(_config: Mapping[str, Any]):
    from services.api.app.llm.client import OpenAIClient
    return OpenAIClient()


def _build_ollama_llm(_config: Mapping[str, Any]):
    from services.api.app.llm.ollama import OllamaAdapter
    return OllamaAdapter()


def _build_noop_reranker(_config: Mapping[str, Any]):
    from services.api.app.retrieval.cross_encoder import NoOpReranker
    return NoOpReranker()


def _build_configured_reranker(_config: Mapping[str, Any]):
    from services.api.app.retrieval.cross_encoder import build_reranker
    return build_reranker()


def _builtin_specs() -> tuple[RuntimeFactorySpec, ...]:
    return (
        RuntimeFactorySpec("repository", "memory", _build_memory_repo, frozenset({"development"})),
        RuntimeFactorySpec("repository", "postgres", _build_postgres_repo, frozenset({"durable", "queue", "fts", "generations"}), optional_dependency="ragbot[postgres]"),
        RuntimeFactorySpec("vector", "memory", _build_memory_vector, frozenset({"development", "dense"})),
        RuntimeFactorySpec("vector", "qdrant", _build_qdrant_vector, frozenset({"dense", "metadata-filter"}), optional_dependency="ragbot[qdrant]"),
        RuntimeFactorySpec("embedding", "hash", _build_hash_embedding, frozenset({"development"})),
        RuntimeFactorySpec("embedding", "openai-compatible", _build_openai_compatible_embedding, frozenset({"semantic", "batch"})),
        RuntimeFactorySpec("llm", "openai", _build_openai_llm, frozenset({"structured-output", "streaming", "web-search"})),
        RuntimeFactorySpec("llm", "ollama", _build_ollama_llm, frozenset({"structured-output", "streaming"})),
        RuntimeFactorySpec("reranker", "none", _build_noop_reranker),
        RuntimeFactorySpec("reranker", "cohere", _build_configured_reranker, frozenset({"rerank"})),
        RuntimeFactorySpec("reranker", "local", _build_configured_reranker, frozenset({"rerank"})),
    )
