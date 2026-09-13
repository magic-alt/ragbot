from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional


def _flag(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeProfile:
    """Non-secret typed snapshot of component selections made at bootstrap."""

    environment: str
    repository_provider: str
    vector_provider: str
    embedding_provider: str
    llm_provider: str
    reranker_provider: str
    upload_provider: str
    connector_ids: tuple[str, ...] = ()

    @classmethod
    def from_environment(
        cls,
        env: Optional[Mapping[str, str]] = None,
        *,
        connector_ids: Iterable[str] = (),
    ) -> "RuntimeProfile":
        values = os.environ if env is None else env
        repository = values.get("RAGBOT_REPOSITORY_PROVIDER", "").strip().lower()
        if not repository:
            repository = "postgres" if values.get("POSTGRES_DSN", "").strip() else "memory"

        vector = values.get("RAGBOT_VECTOR_PROVIDER", "").strip().lower()
        if not vector:
            vector = "qdrant" if values.get("QDRANT_URL", "").strip() else "memory"

        embedding = values.get("RAGBOT_EMBEDDING_PROVIDER", "").strip().lower()
        if not embedding:
            embedding = "openai-compatible" if values.get("EMBEDDING_MODEL", "").strip() else "hash"

        reranker = "none"
        if _flag(values, "RAGBOT_RERANK_ENABLED", False):
            reranker = values.get("RAGBOT_RERANK_PROVIDER", "cohere").strip().lower() or "cohere"

        return cls(
            environment=values.get("RAGBOT_ENV", "development").strip().lower() or "development",
            repository_provider=repository,
            vector_provider=vector,
            embedding_provider=embedding,
            llm_provider=values.get("RAGBOT_LLM_PROVIDER", "openai").strip().lower() or "openai",
            reranker_provider=reranker,
            upload_provider=values.get("RAGBOT_UPLOAD_STORE", "filesystem").strip().lower() or "filesystem",
            connector_ids=tuple(sorted(str(item) for item in connector_ids)),
        )

    def as_public_dict(self) -> dict[str, object]:
        return {
            "environment": self.environment,
            "repository_provider": self.repository_provider,
            "vector_provider": self.vector_provider,
            "embedding_provider": self.embedding_provider,
            "llm_provider": self.llm_provider,
            "reranker_provider": self.reranker_provider,
            "upload_provider": self.upload_provider,
            "connector_ids": list(self.connector_ids),
        }
