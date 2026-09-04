"""Local cache primitives retained for tests/experiments only.

These classes are intentionally NOT wired into Ragbot retrieval or embedding
runtime. A process-local RetrievalCache cannot be safely invalidated across API
replicas and ingestion workers, so it is not a production capability and has no
runtime environment flags or admin endpoint. A future cache must use a shared,
generation-aware invalidation contract before it can sit on the retrieval path.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    hits: int = 0


class LRUCache:
    """Thread-safe local LRU cache with TTL eviction for experiments/tests."""

    def __init__(self, max_entries: int = 1000, ttl_seconds: float = 300.0) -> None:
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if time.time() - entry.created_at > self._ttl:
                del self._store[key]
                self._misses += 1
                return None
            entry.hits += 1
            self._hits += 1
            self._store.move_to_end(key)
            return entry.value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = CacheEntry(key=key, value=value)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def invalidate(self, key: str) -> bool:
        with self._lock:
            if key not in self._store:
                return False
            del self._store[key]
            return True

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "max_entries": self._max_entries,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }


def _cache_key(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class RetrievalCache:
    """Experimental process-local retrieval cache; not used by runtime."""

    def __init__(self, cache: Optional[LRUCache] = None) -> None:
        self._cache = cache or LRUCache()

    def get(self, query: str, filters: Dict[str, Any], top_k: int) -> Optional[list]:
        return self._cache.get(_cache_key(query, sorted(filters.items()), top_k))

    def put(self, query: str, filters: Dict[str, Any], top_k: int, results: list) -> None:
        self._cache.put(_cache_key(query, sorted(filters.items()), top_k), results)

    def stats(self) -> Dict[str, Any]:
        return self._cache.stats()

    def clear(self) -> None:
        self._cache.clear()


class EmbeddingCache:
    """Experimental process-local embedding cache; not used by runtime."""

    def __init__(self, cache: Optional[LRUCache] = None) -> None:
        self._cache = cache or LRUCache(max_entries=5000, ttl_seconds=3600.0)

    def get(self, text: str) -> Optional[List[float]]:
        return self._cache.get(_cache_key(text))

    def put(self, text: str, embedding: List[float]) -> None:
        self._cache.put(_cache_key(text), embedding)

    def stats(self) -> Dict[str, Any]:
        return self._cache.stats()

    def clear(self) -> None:
        self._cache.clear()
