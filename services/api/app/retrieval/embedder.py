from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import requests

logger = logging.getLogger(__name__)


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
    "qwen3-embedding:0.6b": 1024,
    "qwen3-embedding:4b": 2560,
    "qwen3-embedding:8b": 4096,
    "qwen/qwen3-embedding-0.6b": 1024,
    "qwen/qwen3-embedding-4b": 2560,
    "qwen/qwen3-embedding-8b": 4096,
}

_QWEN3_QUERY_TASK = (
    "Given a user question, retrieve relevant passages from the knowledge base "
    "that answer the question"
)


def model_dimension(model: str) -> Optional[int]:
    """Return a known native embedding dimension without case sensitivity."""
    normalized = str(model or "").strip().lower()
    if not normalized:
        return None
    if normalized == "qwen3-embedding":
        # Ollama's unqualified library tag currently resolves to the 8B variant.
        return 4096
    return MODEL_DIMENSIONS.get(normalized)


def default_query_instruction(model: str) -> str:
    """Return a model-specific retrieval instruction when the model benefits from one."""
    normalized = str(model or "").strip().lower()
    if "qwen3-embedding" in normalized:
        return _QWEN3_QUERY_TASK
    return ""


@runtime_checkable
class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> List[float]: ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]: ...


class HashEmbedder:
    """Deterministic hash-based embedder for development and tests."""

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
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest, byteorder="big", signed=False) % self._dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_query(self, text: str) -> List[float]:
        return self.embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


class APIEmbedder:
    """OpenAI-compatible embedding client with strict vector dimensions.

    Document embeddings remain unmodified. Query embeddings can receive a
    retrieval-task instruction; Qwen3 Embedding gets its recommended query-side
    ``Instruct: ... / Query: ...`` shape by default while other models remain
    unchanged. Set ``EMBEDDING_QUERY_INSTRUCTION`` to override it, or to an
    empty string to disable an explicit configured instruction.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        dimension: Optional[int] = None,
        timeout: int = 30,
        batch_size: int = 100,
        query_instruction: Optional[str] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimension = dimension or model_dimension(model) or 1536
        self._timeout = timeout
        self._batch_size = batch_size
        self._query_instruction = (
            default_query_instruction(model)
            if query_instruction is None
            else str(query_instruction).strip()
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def query_instruction(self) -> str:
        return self._query_instruction

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_query(self, text: str) -> List[float]:
        value = text
        if self._query_instruction:
            value = f"Instruct: {self._query_instruction}\nQuery:{text}"
        return self._embed_raw_batch([value])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed knowledge documents without query instructions."""
        return self._embed_raw_batch(texts)

    def _embed_raw_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            all_vectors.extend(self._call_api(batch))
        return all_vectors

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        url = f"{self._base_url}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload: Dict[str, Any] = {"model": self._model, "input": texts}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
            items = sorted(data["data"], key=lambda x: x["index"])
            vectors: List[List[float]] = []
            for item in items:
                vec = item["embedding"]
                self._validate_dimension(vec)
                vectors.append(vec)
            if len(vectors) != len(texts):
                raise ValueError(
                    f"Embedding API returned {len(vectors)} vectors for {len(texts)} inputs"
                )
            return vectors
        except Exception as exc:
            logger.warning("Embedding API failed: %s", exc)
            raise

    def _validate_dimension(self, vec: List[float]) -> None:
        actual = len(vec)
        if actual != self._dimension:
            raise ValueError(
                "Embedding API vector dimension mismatch: "
                f"model={self._model}, actual={actual}, expected={self._dimension}. "
                "Set QDRANT_DIM to the real model dimension and reindex into a compatible collection."
            )


def build_embedder(dimension: Optional[int] = None) -> Embedder:
    """Build an Embedder from environment variables.

    Local OpenAI-compatible endpoints such as Ollama do not require a fake API
    key: setting ``EMBEDDING_MODEL`` + ``EMBEDDING_BASE_URL`` is sufficient.
    Development can still fall back to ``HashEmbedder``; production validation
    rejects that fallback.
    """
    model = os.getenv("EMBEDDING_MODEL", "").strip()
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    explicit_embedding_base = os.getenv("EMBEDDING_BASE_URL", "").strip()
    base_url = explicit_embedding_base or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com"
    dim_override = os.getenv("QDRANT_DIM")
    effective_dimension = int(dim_override) if dim_override else (dimension or model_dimension(model))
    query_instruction = os.getenv("EMBEDDING_QUERY_INSTRUCTION")

    if model and (api_key or explicit_embedding_base):
        logger.info("Using API embedder: model=%s, base_url=%s", model, base_url)
        return APIEmbedder(
            api_key=api_key,
            base_url=base_url,
            model=model,
            dimension=effective_dimension,
            query_instruction=query_instruction,
        )

    fallback_dimension = effective_dimension or 64
    logger.info(
        "Using hash-based embedder (dimension=%d); set EMBEDDING_MODEL + "
        "EMBEDDING_BASE_URL for a local endpoint or EMBEDDING_API_KEY/OPENAI_API_KEY "
        "for a hosted semantic embedding service",
        fallback_dimension,
    )
    return HashEmbedder(dim=fallback_dimension)
