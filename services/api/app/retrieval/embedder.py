from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import threading
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable
from urllib.parse import urlparse

from .embedding_contract import EmbeddingSpec
from .embedding_transport import EmbeddingTransport

# Compatibility hints only. The resolved EmbeddingSpec and actual provider
# response validation are authoritative; this table is not an index schema.
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
_LOCAL_EMBEDDING_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "host.docker.internal"})


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_local_embedding_endpoint(base_url: str) -> bool:
    try:
        hostname = (urlparse(base_url).hostname or "").lower()
    except ValueError:
        return False
    return hostname in _LOCAL_EMBEDDING_HOSTS


def model_dimension(model: str) -> Optional[int]:
    normalized = str(model or "").strip().lower()
    if not normalized:
        return None
    if normalized == "qwen3-embedding" or normalized.startswith("qwen3-embedding:8b"):
        return 4096
    if normalized.startswith("qwen3-embedding:4b"):
        return 2560
    if normalized.startswith("qwen3-embedding:0.6b"):
        return 1024
    if normalized.startswith("qwen/qwen3-embedding-8b"):
        return 4096
    if normalized.startswith("qwen/qwen3-embedding-4b"):
        return 2560
    if normalized.startswith("qwen/qwen3-embedding-0.6b"):
        return 1024
    return MODEL_DIMENSIONS.get(normalized)


def default_query_instruction(model: str) -> str:
    return _QWEN3_QUERY_TASK if "qwen3-embedding" in str(model or "").strip().lower() else ""


@runtime_checkable
class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...
    @property
    def dimension(self) -> int: ...
    @property
    def contract_id(self) -> str: ...
    def embed(self, text: str) -> List[float]: ...
    def embed_query(self, text: str) -> List[float]: ...
    def embed_batch(self, texts: List[str]) -> List[List[float]]: ...
    async def aembed_query(self, text: str) -> List[float]: ...
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]: ...


class MemoryEmbeddingCache:
    """Content-addressed vector cache; no tenant/ACL state is stored."""

    def __init__(self, max_entries: int = 10_000) -> None:
        self.max_entries = max(0, int(max_entries))
        self._values: OrderedDict[str, List[float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[List[float]]:
        if self.max_entries <= 0:
            return None
        with self._lock:
            value = self._values.get(key)
            if value is None:
                return None
            self._values.move_to_end(key)
            return list(value)

    def put(self, key: str, vector: List[float]) -> None:
        if self.max_entries <= 0:
            return
        with self._lock:
            self._values[key] = list(vector)
            self._values.move_to_end(key)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)


class HashEmbedder:
    """Deterministic hash-based embedder for development and tests."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self.spec = EmbeddingSpec(provider_id="hash", model=f"hash-{dim}", dimension=dim, normalize=True)

    @property
    def model_name(self) -> str:
        return self.spec.model

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def contract_id(self) -> str:
        return self.spec.contract_id

    def embed(self, text: str) -> List[float]:
        tokens = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
        vec = [0.0] * self._dim
        for tok in tokens:
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest, byteorder="big", signed=False) % self._dim
            vec[idx] += 1.0
        return _normalize(vec)

    def embed_query(self, text: str) -> List[float]:
        return self.embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_batch(texts)


class APIEmbedder:
    """OpenAI-compatible embedding client bound to one immutable EmbeddingSpec."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        dimension: Optional[int] = None,
        timeout: int = 30,
        batch_size: int = 100,
        query_instruction: Optional[str] = None,
        *,
        provider_id: str = "openai-compatible",
        revision: str = "",
        document_instruction: str = "",
        normalize: bool = False,
        max_batch_bytes: int = 1_000_000,
        max_attempts: int = 4,
        concurrency: int = 8,
        cache: Optional[MemoryEmbeddingCache] = None,
        transport: Optional[EmbeddingTransport] = None,
    ) -> None:
        resolved_dimension = dimension or model_dimension(model) or 1536
        resolved_query_instruction = default_query_instruction(model) if query_instruction is None else str(query_instruction).strip()
        self.spec = EmbeddingSpec(
            provider_id=provider_id,
            model=model,
            revision=revision,
            dimension=resolved_dimension,
            normalize=normalize,
            query_instruction=resolved_query_instruction,
            document_instruction=str(document_instruction or "").strip(),
            max_batch_items=int(batch_size),
            max_batch_bytes=int(max_batch_bytes),
            multilingual=("qwen" in model.lower() or "multilingual" in model.lower()),
        )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport = transport or EmbeddingTransport(
            timeout_seconds=timeout,
            max_attempts=max_attempts,
            concurrency=concurrency,
        )
        self._owns_transport = transport is None
        self._cache = cache
        self._stats = {"requests": 0, "inputs": 0, "bytes": 0, "retries": 0, "cache_hits": 0, "failures": 0, "provider_latency_ms": 0.0}
        self._stats_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self.spec.model

    @property
    def dimension(self) -> int:
        return self.spec.dimension

    @property
    def contract_id(self) -> str:
        return self.spec.contract_id

    @property
    def query_instruction(self) -> str:
        return self.spec.query_instruction

    def diagnostics(self) -> dict[str, Any]:
        with self._stats_lock:
            stats = dict(self._stats)
        return {"contract_id": self.contract_id, "spec": self.spec.as_public_dict(), "metrics": stats}

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_sync([text], role="query")[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self._embed_sync(texts, role="document")

    async def aembed_query(self, text: str) -> List[float]:
        return (await self._embed_async([text], role="query"))[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self._embed_async(texts, role="document")

    def _embed_sync(self, texts: List[str], *, role: str) -> List[List[float]]:
        if not texts:
            return []
        prepared = [self._prepare(text, role) for text in texts]
        output: list[Optional[List[float]]] = [None] * len(prepared)
        missing: list[tuple[int, str]] = []
        for index, value in enumerate(prepared):
            cached = self._cache_get(value, role)
            if cached is not None:
                output[index] = cached
                self._inc("cache_hits", 1)
            else:
                missing.append((index, value))
        for batch in _adaptive_batches(missing, self.spec.max_batch_items, self.spec.max_batch_bytes):
            indices = [index for index, _ in batch]
            values = [value for _, value in batch]
            vectors = self._call_api(values)
            for index, value, vector in zip(indices, values, vectors):
                output[index] = vector
                self._cache_put(value, role, vector)
        return [item for item in output if item is not None]

    async def _embed_async(self, texts: List[str], *, role: str) -> List[List[float]]:
        if not texts:
            return []
        prepared = [self._prepare(text, role) for text in texts]
        output: list[Optional[List[float]]] = [None] * len(prepared)
        missing: list[tuple[int, str]] = []
        for index, value in enumerate(prepared):
            cached = self._cache_get(value, role)
            if cached is not None:
                output[index] = cached
                self._inc("cache_hits", 1)
            else:
                missing.append((index, value))
        batches = list(_adaptive_batches(missing, self.spec.max_batch_items, self.spec.max_batch_bytes))
        results = await asyncio.gather(*[self._acall_api([value for _, value in batch]) for batch in batches])
        for batch, vectors in zip(batches, results):
            for (index, value), vector in zip(batch, vectors):
                output[index] = vector
                self._cache_put(value, role, vector)
        return [item for item in output if item is not None]

    def _prepare(self, text: str, role: str) -> str:
        instruction = self.spec.query_instruction if role == "query" else self.spec.document_instruction
        if not instruction:
            return text
        label = "Query" if role == "query" else "Document"
        return f"Instruct: {instruction}\n{label}:{text}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self.spec.model, "input": texts}
        try:
            data, retries, latency = self._transport.post_json(
                f"{self._base_url}/v1/embeddings", headers=self._headers(), payload=payload
            )
            self._observe(texts, retries, latency)
            return self._parse_vectors(data, len(texts))
        except Exception:
            self._inc("failures", 1)
            raise

    async def _acall_api(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self.spec.model, "input": texts}
        try:
            data, retries, latency = await self._transport.apost_json(
                f"{self._base_url}/v1/embeddings", headers=self._headers(), payload=payload
            )
            self._observe(texts, retries, latency)
            return self._parse_vectors(data, len(texts))
        except Exception:
            self._inc("failures", 1)
            raise

    def _parse_vectors(self, data: Dict[str, Any], expected: int) -> List[List[float]]:
        items = sorted(data["data"], key=lambda x: x["index"])
        vectors: List[List[float]] = []
        for item in items:
            vector = [float(value) for value in item["embedding"]]
            self._validate_dimension(vector)
            vectors.append(_normalize(vector) if self.spec.normalize else vector)
        if len(vectors) != expected:
            raise ValueError(f"Embedding API returned {len(vectors)} vectors for {expected} inputs")
        return vectors

    def _validate_dimension(self, vector: List[float]) -> None:
        if len(vector) != self.spec.dimension:
            raise ValueError(
                "Embedding API vector dimension mismatch: "
                f"model={self.spec.model}, actual={len(vector)}, expected={self.spec.dimension}. "
                "Build a compatible index version after changing embedding contracts."
            )

    def _cache_key(self, value: str, role: str) -> str:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return f"{self.contract_id}:{role}:{digest}"

    def _cache_get(self, value: str, role: str) -> Optional[List[float]]:
        return self._cache.get(self._cache_key(value, role)) if self._cache else None

    def _cache_put(self, value: str, role: str, vector: List[float]) -> None:
        if self._cache:
            self._cache.put(self._cache_key(value, role), vector)

    def _observe(self, texts: List[str], retries: int, latency: float) -> None:
        self._inc("requests", 1)
        self._inc("inputs", len(texts))
        self._inc("bytes", sum(len(text.encode("utf-8")) for text in texts))
        self._inc("retries", retries)
        self._inc("provider_latency_ms", latency)

    def _inc(self, key: str, amount: float) -> None:
        with self._stats_lock:
            self._stats[key] += amount

    def close(self) -> None:
        if self._owns_transport:
            self._transport.close()

    async def aclose(self) -> None:
        if self._owns_transport:
            await self._transport.aclose()


def _adaptive_batches(items: List[tuple[int, str]], max_items: int, max_bytes: int) -> Iterable[List[tuple[int, str]]]:
    batch: List[tuple[int, str]] = []
    size = 0
    for item in items:
        item_bytes = len(item[1].encode("utf-8"))
        if item_bytes > max_bytes:
            raise ValueError(f"Single embedding input exceeds max batch bytes: {item_bytes} > {max_bytes}")
        if batch and (len(batch) >= max_items or size + item_bytes > max_bytes):
            yield batch
            batch = []
            size = 0
        batch.append(item)
        size += item_bytes
    if batch:
        yield batch


def _normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def build_embedder(dimension: Optional[int] = None) -> Embedder:
    model = os.getenv("EMBEDDING_MODEL", "").strip()
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    explicit_embedding_base = os.getenv("EMBEDDING_BASE_URL", "").strip()
    base_url = explicit_embedding_base or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com"
    dim_override = os.getenv("QDRANT_DIM")
    effective_dimension = int(dim_override) if dim_override else (dimension or model_dimension(model))
    query_instruction = os.getenv("EMBEDDING_QUERY_INSTRUCTION")
    document_instruction = os.getenv("EMBEDDING_DOCUMENT_INSTRUCTION", "")
    anonymous_allowed = _is_local_embedding_endpoint(base_url) or _env_flag("EMBEDDING_ALLOW_ANONYMOUS", False)

    if model and (api_key or anonymous_allowed):
        timeout = int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30"))
        batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
        max_batch_bytes = int(os.getenv("EMBEDDING_MAX_BATCH_BYTES", "1000000"))
        max_attempts = int(os.getenv("EMBEDDING_MAX_ATTEMPTS", "4"))
        concurrency = int(os.getenv("EMBEDDING_CONCURRENCY", "8"))
        cache_size = int(os.getenv("EMBEDDING_CACHE_MAX_ENTRIES", "0"))
        cache = MemoryEmbeddingCache(cache_size) if cache_size > 0 else None
        return APIEmbedder(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider_id=os.getenv("EMBEDDING_PROVIDER", "openai-compatible").strip().lower() or "openai-compatible",
            revision=os.getenv("EMBEDDING_MODEL_REVISION", "").strip(),
            dimension=effective_dimension,
            timeout=timeout,
            batch_size=batch_size,
            max_batch_bytes=max_batch_bytes,
            max_attempts=max_attempts,
            concurrency=concurrency,
            query_instruction=query_instruction,
            document_instruction=document_instruction,
            normalize=_env_flag("EMBEDDING_NORMALIZE", False),
            cache=cache,
        )

    return HashEmbedder(dim=effective_dimension or 64)
