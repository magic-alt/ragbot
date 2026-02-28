"""In-memory caching for retrieval results and embeddings.

Provides TTL-based caching to reduce redundant LLM/retrieval calls
within short time windows.

Environment variables:
    RAGBOT_CACHE_ENABLED: enable caching (default: true)
    RAGBOT_CACHE_TTL_SECONDS: TTL for cache entries (default: 300)
    RAGBOT_CACHE_MAX_ENTRIES: max entries before eviction (default: 1000)
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached value with metadata."""

    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    hits: int = 0


class LRUCache:
    """Thread-safe LRU cache with TTL eviction.

    Usage::

        cache = LRUCache(max_entries=1000, ttl_seconds=300)
        cache.put("key", value)
        result = cache.get("key")
    """

    def __init__(
        self,
        max_entries: int = 1000,
        ttl_seconds: float = 300.0,
    ) -> None:
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache. Returns None on miss or expiry."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            # Check TTL
            if time.time() - entry.created_at > self._ttl:
                del self._store[key]
                self._misses += 1
                return None

            entry.hits += 1
            self._hits += 1
            # Move to end (most recently used)
            self._store.move_to_end(key)
            return entry.value

    def put(self, key: str, value: Any) -> None:
        """Store a value in the cache."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = CacheEntry(key=key, value=value)
            else:
                self._store[key] = CacheEntry(key=key, value=value)

            # Evict oldest if over capacity
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def invalidate(self, key: str) -> bool:
        """Remove a specific entry."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "max_entries": self._max_entries,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            }


def _cache_key(*parts: Any) -> str:
    """Build a deterministic cache key from parts."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class RetrievalCache:
    """Cache layer for retrieval results.

    Caches the mapping (query, filters, top_k) → chunks to avoid
    redundant vector searches for identical queries.
    """

    def __init__(self, cache: Optional[LRUCache] = None) -> None:
        self._cache = cache or LRUCache()

    def get(self, query: str, filters: Dict[str, Any], top_k: int) -> Optional[list]:
        """Look up cached retrieval results."""
        key = _cache_key(query, sorted(filters.items()), top_k)
        return self._cache.get(key)

    def put(self, query: str, filters: Dict[str, Any], top_k: int, results: list) -> None:
        """Cache retrieval results."""
        key = _cache_key(query, sorted(filters.items()), top_k)
        self._cache.put(key, results)

    def stats(self) -> Dict[str, Any]:
        return self._cache.stats()

    def clear(self) -> None:
        self._cache.clear()


class EmbeddingCache:
    """Cache layer for embedding vectors.

    Caches text → embedding vector to avoid redundant embedding calls.
    """

    def __init__(self, cache: Optional[LRUCache] = None) -> None:
        self._cache = cache or LRUCache(max_entries=5000, ttl_seconds=3600.0)

    def get(self, text: str) -> Optional[List[float]]:
        """Look up cached embedding."""
        key = _cache_key(text)
        return self._cache.get(key)

    def put(self, text: str, embedding: List[float]) -> None:
        """Cache an embedding."""
        key = _cache_key(text)
        self._cache.put(key, embedding)

    def stats(self) -> Dict[str, Any]:
        return self._cache.stats()

    def clear(self) -> None:
        self._cache.clear()


# Global caches
_retrieval_cache: Optional[RetrievalCache] = None
_embedding_cache: Optional[EmbeddingCache] = None
_caches_lock = threading.Lock()


def get_retrieval_cache() -> RetrievalCache:
    """Get or create the global retrieval cache."""
    global _retrieval_cache
    with _caches_lock:
        if _retrieval_cache is None:
            max_entries = int(os.getenv("RAGBOT_CACHE_MAX_ENTRIES", "1000"))
            ttl = float(os.getenv("RAGBOT_CACHE_TTL_SECONDS", "300"))
            _retrieval_cache = RetrievalCache(LRUCache(max_entries, ttl))
        return _retrieval_cache


def get_embedding_cache() -> EmbeddingCache:
    """Get or create the global embedding cache."""
    global _embedding_cache
    with _caches_lock:
        if _embedding_cache is None:
            _embedding_cache = EmbeddingCache()
        return _embedding_cache


def is_cache_enabled() -> bool:
    """Check if caching is enabled via environment."""
    return os.getenv("RAGBOT_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
