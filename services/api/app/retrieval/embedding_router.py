from __future__ import annotations

import json
import os
from typing import Any, Iterable, Optional

from .embedder import APIEmbedder, Embedder
from .embedding_contract import embedding_contract_id


class EmbeddingRouter:
    """In-process registry of immutable embedding contracts.

    Index activation can only select a contract registered here. This keeps old
    and candidate embedders available concurrently, allowing the Qdrant alias
    and query-time embedding contract to switch together without restarting the
    serving process.
    """

    def __init__(self, embedders: Iterable[Embedder] = ()) -> None:
        self._embedders: dict[str, Embedder] = {}
        for embedder in embedders:
            self.register(embedder)

    def register(self, embedder: Embedder, *, replace: bool = False) -> str:
        contract_id = embedding_contract_id(embedder)
        if contract_id in self._embedders and not replace:
            existing = self._embedders[contract_id]
            if existing is not embedder:
                raise ValueError(f"Embedding contract already registered: {contract_id}")
        self._embedders[contract_id] = embedder
        return contract_id

    def get(self, contract_id: str) -> Embedder:
        try:
            return self._embedders[contract_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._embedders)) or "<none>"
            raise KeyError(
                f"Embedding contract is not registered: {contract_id}; available={available}"
            ) from exc

    def has(self, contract_id: str) -> bool:
        return contract_id in self._embedders

    def contracts(self) -> tuple[str, ...]:
        return tuple(sorted(self._embedders))

    def public_metadata(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for contract_id in sorted(self._embedders):
            embedder = self._embedders[contract_id]
            spec = getattr(embedder, "spec", None)
            as_public_dict = getattr(spec, "as_public_dict", None)
            output.append(
                {
                    "contract_id": contract_id,
                    "model": getattr(embedder, "model_name", "unknown"),
                    "dimension": int(getattr(embedder, "dimension", 0) or 0),
                    "spec": as_public_dict() if callable(as_public_dict) else None,
                }
            )
        return output

    def close(self) -> None:
        closed: set[int] = set()
        for embedder in self._embedders.values():
            if id(embedder) in closed:
                continue
            close = getattr(embedder, "close", None)
            if callable(close):
                close()
            closed.add(id(embedder))


class ActiveIndexEmbedder:
    """Embedder façade that follows the active IndexVersion contract."""

    def __init__(self, repo: Any, router: EmbeddingRouter, alias_name: str, fallback: Embedder) -> None:
        self._repo = repo
        self._router = router
        self._alias_name = alias_name
        self._fallback = fallback

    def _active_version(self):
        getter = getattr(self._repo, "get_active_index_version", None)
        return getter(self._alias_name) if callable(getter) else None

    def _active_embedder(self) -> Embedder:
        version = self._active_version()
        if version is None:
            return self._fallback
        try:
            return self._router.get(version.embedding_contract_id)
        except KeyError as exc:
            raise RuntimeError(
                "Active vector index requires an embedding contract that is not registered in this runtime: "
                f"index={version.index_version_id}, contract={version.embedding_contract_id}"
            ) from exc

    @property
    def index_version_id(self) -> Optional[str]:
        version = self._active_version()
        return version.index_version_id if version is not None else None

    @property
    def spec(self):
        return getattr(self._active_embedder(), "spec", None)

    @property
    def model_name(self) -> str:
        return self._active_embedder().model_name

    @property
    def dimension(self) -> int:
        return self._active_embedder().dimension

    @property
    def contract_id(self) -> str:
        return embedding_contract_id(self._active_embedder())

    def embed(self, text: str) -> list[float]:
        return self._active_embedder().embed(text)

    def embed_query(self, text: str) -> list[float]:
        embed_query = getattr(self._active_embedder(), "embed_query", None)
        return embed_query(text) if callable(embed_query) else self._active_embedder().embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._active_embedder().embed_batch(texts)

    async def aembed_query(self, text: str) -> list[float]:
        embedder = self._active_embedder()
        method = getattr(embedder, "aembed_query", None)
        if callable(method):
            return await method(text)
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        embedder = self._active_embedder()
        method = getattr(embedder, "aembed_documents", None)
        if callable(method):
            return await method(texts)
        return embedder.embed_batch(texts)


def build_embedding_router(default_embedder: Embedder) -> EmbeddingRouter:
    """Build the default + optional standby embedding contracts from env.

    `RAGBOT_EMBEDDING_PROFILES_JSON` is deliberately non-secret. Each profile
    references an API-key environment variable by name instead of embedding a
    credential in the JSON. Example:

    {"qwen-new":{"model":"qwen3-embedding:4b","dimension":2560,
      "base_url":"http://host.docker.internal:11434","api_key_env":""}}
    """
    router = EmbeddingRouter([default_embedder])
    raw = os.getenv("RAGBOT_EMBEDDING_PROFILES_JSON", "").strip()
    if not raw:
        return router
    try:
        profiles = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("RAGBOT_EMBEDDING_PROFILES_JSON must be valid JSON") from exc
    if not isinstance(profiles, dict):
        raise ValueError("RAGBOT_EMBEDDING_PROFILES_JSON must be a JSON object")

    for profile_name, config in profiles.items():
        if not isinstance(config, dict):
            raise ValueError(f"Embedding profile {profile_name!r} must be an object")
        if "api_key" in config:
            raise ValueError(
                f"Embedding profile {profile_name!r} must use api_key_env, not inline api_key"
            )
        model = str(config.get("model") or "").strip()
        base_url = str(config.get("base_url") or "").strip()
        dimension = int(config.get("dimension") or 0)
        if not model or not base_url or dimension <= 0:
            raise ValueError(
                f"Embedding profile {profile_name!r} requires model, base_url and positive dimension"
            )
        api_key_env = str(config.get("api_key_env") or "").strip()
        api_key = os.getenv(api_key_env, "") if api_key_env else ""
        embedder = APIEmbedder(
            api_key=api_key,
            base_url=base_url,
            model=model,
            dimension=dimension,
            provider_id=str(config.get("provider_id") or "openai-compatible").strip(),
            revision=str(config.get("revision") or "").strip(),
            query_instruction=config.get("query_instruction"),
            document_instruction=str(config.get("document_instruction") or "").strip(),
            normalize=bool(config.get("normalize", False)),
            batch_size=int(config.get("max_batch_items") or 100),
            max_batch_bytes=int(config.get("max_batch_bytes") or 1_000_000),
            timeout=int(config.get("timeout_seconds") or 30),
            max_attempts=int(config.get("max_attempts") or 4),
            concurrency=int(config.get("concurrency") or 8),
        )
        actual = router.register(embedder)
        expected = str(config.get("contract_id") or "").strip()
        if expected and expected != actual:
            raise ValueError(
                f"Embedding profile {profile_name!r} contract mismatch: expected={expected}, actual={actual}"
            )
    return router
