from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import requests

logger = logging.getLogger(__name__)


@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranking query-document pairs."""

    @property
    def enabled(self) -> bool: ...

    def rerank(
        self, query: str, documents: List[str], top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Return list of (original_index, relevance_score), sorted by score desc."""
        ...


class NoOpReranker:
    """Pass-through reranker that preserves original ordering."""

    @property
    def enabled(self) -> bool:
        return False

    def rerank(
        self, query: str, documents: List[str], top_k: int = 10
    ) -> List[Tuple[int, float]]:
        return [(i, 1.0 - i * 0.001) for i in range(min(top_k, len(documents)))]


class CohereReranker:
    """Calls Cohere rerank API."""

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-v3.5",
        top_k: int = 10,
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._top_k = top_k
        self._timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def rerank(
        self, query: str, documents: List[str], top_k: int = 10
    ) -> List[Tuple[int, float]]:
        if not documents:
            return []
        effective_top_k = min(top_k or self._top_k, len(documents))
        url = "https://api.cohere.com/v2/rerank"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": effective_top_k,
        }
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self._timeout
            )
            response.raise_for_status()
            data = response.json()
            results = []
            for item in data.get("results", []):
                results.append((item["index"], item["relevance_score"]))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:effective_top_k]
        except Exception as exc:
            logger.warning("Cohere rerank failed: %s", exc)
            raise


class LocalCrossEncoder:
    """Uses a local HTTP endpoint for cross-encoder scoring.

    Compatible with Hugging Face TEI (Text Embeddings Inference) rerank endpoint
    or any service exposing POST /rerank with {query, texts} -> [{index, score}].
    """

    def __init__(
        self,
        base_url: str,
        model: str = "",
        top_k: int = 10,
        timeout: int = 30,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._top_k = top_k
        self._timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self._base_url)

    def rerank(
        self, query: str, documents: List[str], top_k: int = 10
    ) -> List[Tuple[int, float]]:
        if not documents:
            return []
        effective_top_k = min(top_k or self._top_k, len(documents))
        url = f"{self._base_url}/rerank"
        payload: Dict[str, Any] = {"query": query, "texts": documents}
        if self._model:
            payload["model"] = self._model
        try:
            response = requests.post(url, json=payload, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
            results = []
            for item in data if isinstance(data, list) else data.get("results", []):
                idx = item.get("index", 0)
                score = item.get("score", item.get("relevance_score", 0.0))
                results.append((idx, score))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:effective_top_k]
        except Exception as exc:
            logger.warning("Local cross-encoder rerank failed: %s", exc)
            raise


def build_reranker() -> Reranker:
    """Factory: build a Reranker from environment variables.

    Environment variables:
        RAGBOT_RERANK_ENABLED: 'true' to enable (default: false)
        RAGBOT_RERANK_PROVIDER: 'cohere' or 'local' (default: cohere)
        RAGBOT_RERANK_MODEL: Model name
        RAGBOT_RERANK_API_KEY: API key (for Cohere)
        RAGBOT_RERANK_BASE_URL: Base URL (for local cross-encoder)
        RAGBOT_RERANK_TOP_K: Default top-k (default: 10)
    """
    enabled = os.getenv("RAGBOT_RERANK_ENABLED", "false").lower() == "true"
    if not enabled:
        return NoOpReranker()

    provider = os.getenv("RAGBOT_RERANK_PROVIDER", "cohere").lower()
    model = os.getenv("RAGBOT_RERANK_MODEL", "")
    top_k = int(os.getenv("RAGBOT_RERANK_TOP_K", "10"))

    if provider == "cohere":
        api_key = os.getenv("RAGBOT_RERANK_API_KEY", "")
        if not api_key:
            logger.warning("RAGBOT_RERANK_API_KEY not set, disabling reranker")
            return NoOpReranker()
        logger.info("Using Cohere reranker: model=%s", model or "rerank-v3.5")
        return CohereReranker(
            api_key=api_key,
            model=model or "rerank-v3.5",
            top_k=top_k,
        )

    if provider == "local":
        base_url = os.getenv("RAGBOT_RERANK_BASE_URL", "")
        if not base_url:
            logger.warning("RAGBOT_RERANK_BASE_URL not set, disabling reranker")
            return NoOpReranker()
        logger.info("Using local cross-encoder: url=%s, model=%s", base_url, model)
        return LocalCrossEncoder(
            base_url=base_url,
            model=model,
            top_k=top_k,
        )

    logger.warning("Unknown rerank provider '%s', disabling reranker", provider)
    return NoOpReranker()
