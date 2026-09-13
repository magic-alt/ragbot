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


def _qdrant_alias_target(
    url: str,
    api_key: Any,
    alias_name: str,
) -> Optional[str]:
    if not alias_name:
        return None
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return None
    client = QdrantClient(url=url, api_key=api_key)
    try:
        aliases = client.get_aliases()
        for item in getattr(aliases, "aliases", None) or []:
            if str(getattr(item, "alias_name", "")) == alias_name:
                return str(getattr(item, "collection_name", "")) or None
        return None
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _qdrant_alias_dimension(
    url: str,
    api_key: Any,
    alias_name: str,
) -> Optional[int]:
    """Resolve the query-visible schema before constructing the adapter."""
    target = _qdrant_alias_target(url, api_key, alias_name)
    if not target:
        return None
    from qdrant_client import QdrantClient
    client = QdrantClient(url=url, api_key=api_key)
    try:
        info = client.get_collection(target)
        vectors = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
        actual = getattr(vectors, "size", None)
        if actual is None and isinstance(vectors, dict):
            unnamed = vectors.get("") or next(iter(vectors.values()), None)
            actual = getattr(unnamed, "size", None)
        return int(actual) if actual is not None else None
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _ensure_qdrant_alias(
    url: str,
    api_key: Any,
    alias_name: str,
    collection_name: str,
) -> None:
    """Create the bootstrap alias once, tolerating another replica winning."""
    target = _qdrant_alias_target(url, api_key, alias_name)
    if target:
        if target != collection_name:
            # A pre-existing alias target is authoritative. Do not overwrite it
            # during startup; activation/reconcile owns later transitions.
            return
        return

    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest

    client = QdrantClient(url=url, api_key=api_key)
    try:
        try:
            client.update_collection_aliases(
                change_aliases_operations=[
                    rest.CreateAliasOperation(
                        create_alias=rest.CreateAlias(
                            collection_name=collection_name,
                            alias_name=alias_name,
                        )
                    )
                ]
            )
        except Exception:
            # API and worker commonly start at the same time after migrations.
            # If the other replica created the same alias first, converge rather
            # than failing this process startup.
            current = _qdrant_alias_target(url, api_key, alias_name)
            if current != collection_name:
                raise
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _build_qdrant_vector(config: Mapping[str, Any]):
    from services.api.app.retrieval.qdrant import QdrantClientAdapter

    url = str(config.get("url") or "").strip()
    if not url:
        raise ValueError("vector:qdrant requires url")
    api_key = config.get("api_key")
    collection_name = str(config.get("collection_name") or "rag_chunks")
    alias_name = str(config.get("alias_name") or "").strip() or None
    configured_dim = int(config["dim"])

    if alias_name and _qdrant_alias_target(url, api_key, alias_name) is None:
        # Ensure the legacy physical collection/payload indexes first. This
        # direct adapter has no alias side effects.
        bootstrap = QdrantClientAdapter(
            url=url,
            api_key=api_key,
            collection_name=collection_name,
            dim=configured_dim,
            alias_name=None,
        )
        bootstrap.close()
        _ensure_qdrant_alias(url, api_key, alias_name, collection_name)

    visible_dim = (
        _qdrant_alias_dimension(url, api_key, alias_name)
        if alias_name
        else None
    )
    return QdrantClientAdapter(
        url=url,
        api_key=api_key,
        collection_name=collection_name,
        dim=visible_dim or configured_dim,
        alias_name=alias_name,
    )


def _build_hash_embedding(config: Mapping[str, Any]):
    from services.api.app.retrieval.embedder import HashEmbedder
    return HashEmbedder(dim=int(config["dimension"]))


def _build_openai_compatible_embedding(config: Mapping[str, Any]):
    from services.api.app.retrieval.embedder import build_embedder
    return build_embedder(dimension=int(config["dimension"]))


def _endpoint(config: Mapping[str, Any]):
    endpoint = config.get("endpoint")
    if endpoint is None:
        raise ValueError("LLM runtime component requires ModelEndpoint config")
    return endpoint


def _build_openai_llm(config: Mapping[str, Any]):
    from services.api.app.llm.client import OpenAIClient
    return OpenAIClient(endpoint=_endpoint(config))


def _build_ollama_llm(config: Mapping[str, Any]):
    from services.api.app.llm.ollama import OllamaAdapter
    return OllamaAdapter(endpoint=_endpoint(config))


def _build_anthropic_llm(config: Mapping[str, Any]):
    from services.api.app.llm.anthropic import AnthropicAdapter
    return AnthropicAdapter(_endpoint(config))


def _build_gemini_llm(config: Mapping[str, Any]):
    from services.api.app.llm.gemini import GeminiAdapter
    return GeminiAdapter(_endpoint(config))


def _build_azure_llm(config: Mapping[str, Any]):
    from services.api.app.llm.azure import AzureOpenAIAdapter
    return AzureOpenAIAdapter(_endpoint(config))


def _build_bedrock_llm(config: Mapping[str, Any]):
    from services.api.app.llm.bedrock import BedrockAdapter
    return BedrockAdapter(_endpoint(config))


def _build_noop_reranker(_config: Mapping[str, Any]):
    from services.api.app.retrieval.cross_encoder import NoOpReranker
    return NoOpReranker()


def _build_configured_reranker(_config: Mapping[str, Any]):
    from services.api.app.retrieval.cross_encoder import build_reranker
    return build_reranker()


def _builtin_specs() -> tuple[RuntimeFactorySpec, ...]:
    return (
        RuntimeFactorySpec("repository", "memory", _build_memory_repo, frozenset({"development"})),
        RuntimeFactorySpec("repository", "postgres", _build_postgres_repo, frozenset({"durable", "queue", "fts", "generations", "index-lifecycle"}), optional_dependency="ragbot[postgres]"),
        RuntimeFactorySpec("vector", "memory", _build_memory_vector, frozenset({"development", "dense"})),
        RuntimeFactorySpec("vector", "qdrant", _build_qdrant_vector, frozenset({"dense", "metadata-filter", "aliases", "versioned-index"}), optional_dependency="ragbot[qdrant]"),
        RuntimeFactorySpec("embedding", "hash", _build_hash_embedding, frozenset({"development"})),
        RuntimeFactorySpec("embedding", "openai-compatible", _build_openai_compatible_embedding, frozenset({"semantic", "batch"})),
        RuntimeFactorySpec("llm", "openai", _build_openai_llm, frozenset({"structured-output", "json-schema", "streaming", "tools", "web-search"})),
        RuntimeFactorySpec("llm", "openai-compatible", _build_openai_llm, frozenset({"structured-output", "json-schema", "streaming"})),
        RuntimeFactorySpec("llm", "ollama", _build_ollama_llm, frozenset({"structured-output", "json-schema", "streaming"})),
        RuntimeFactorySpec("llm", "anthropic", _build_anthropic_llm, frozenset({"structured-output", "streaming", "tools"})),
        RuntimeFactorySpec("llm", "gemini", _build_gemini_llm, frozenset({"structured-output", "json-schema", "streaming", "tools"})),
        RuntimeFactorySpec("llm", "vertex", _build_gemini_llm, frozenset({"structured-output", "json-schema", "streaming", "tools"})),
        RuntimeFactorySpec("llm", "azure-openai", _build_azure_llm, frozenset({"structured-output", "json-schema", "streaming", "tools"})),
        RuntimeFactorySpec("llm", "bedrock", _build_bedrock_llm, frozenset({"structured-output", "tools"}), optional_dependency="ragbot[s3]"),
        RuntimeFactorySpec("reranker", "none", _build_noop_reranker),
        RuntimeFactorySpec("reranker", "cohere", _build_configured_reranker, frozenset({"rerank"})),
        RuntimeFactorySpec("reranker", "local", _build_configured_reranker, frozenset({"rerank"})),
    )
