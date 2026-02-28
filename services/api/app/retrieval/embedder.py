from __future__ import annotations

import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import requests

logger = logging.getLogger(__name__)


# Well-known model dimensions
MODEL_DIMENSIONS: Dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "bge-small-en-v1.5": 384,
    "bge-base-en-v1.5": 768,
    "bge-large-en-v1.5": 1024,
    "e5-small-v2": 384,
    "e5-base-v2": 768,
    "e5-large-v2": 1024,
}


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding text into vectors."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> List[float]: ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]: ...


class HashEmbedder:
    """Hash-based embedder for dev/testing. Zero external dependencies."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return f"hash-{self._dim}"

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> List[float]:
        tokens = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
        vec = [0.0] * self._dim
        for tok in tokens:
            idx = (hash(tok) % self._dim + self._dim) % self._dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


class APIEmbedder:
    """Embedder that calls an OpenAI-compatible /v1/embeddings endpoint.

    Works with OpenAI, Ollama, vLLM, LiteLLM, and any other service that
    exposes the standard embeddings API.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        dimension: Optional[int] = None,
        timeout: int = 30,
        batch_size: int = 100,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimension = dimension or MODEL_DIMENSIONS.get(model, 1536)
        self._timeout = timeout
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            vectors = self._call_api(batch)
            all_vectors.extend(vectors)
        return all_vectors

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        url = f"{self._base_url}/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {"model": self._model, "input": texts}
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self._timeout
            )
            response.raise_for_status()
            data = response.json()
            # Sort by index to maintain order
            items = sorted(data["data"], key=lambda x: x["index"])
            vectors = []
            for item in items:
                vec = item["embedding"]
                vec = self._normalize_dimension(vec)
                vectors.append(vec)
            return vectors
        except Exception as exc:
            logger.warning("Embedding API failed: %s", exc)
            raise

    def _normalize_dimension(self, vec: List[float]) -> List[float]:
        if len(vec) == self._dimension:
            return vec
        if len(vec) > self._dimension:
            return vec[: self._dimension]
        return vec + [0.0] * (self._dimension - len(vec))


def build_embedder() -> Embedder:
    """Factory: build an Embedder from environment variables.

    Environment variables:
        EMBEDDING_MODEL: Model name (e.g., 'text-embedding-3-small')
        EMBEDDING_API_KEY: API key (falls back to OPENAI_API_KEY)
        EMBEDDING_BASE_URL: API base URL (falls back to OPENAI_BASE_URL)
        QDRANT_DIM: Override dimension (optional)

    Returns HashEmbedder if EMBEDDING_MODEL is not set.
    """
    model = os.getenv("EMBEDDING_MODEL", "")
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = (
        os.getenv("EMBEDDING_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.openai.com"
    )

    if model and api_key:
        dim_override = os.getenv("QDRANT_DIM")
        dimension = int(dim_override) if dim_override else None
        logger.info("Using API embedder: model=%s, base_url=%s", model, base_url)
        return APIEmbedder(
            api_key=api_key,
            base_url=base_url,
            model=model,
            dimension=dimension,
        )

    logger.info(
        "Using hash-based embedder (set EMBEDDING_MODEL + EMBEDDING_API_KEY for real embeddings)"
    )
    return HashEmbedder()
