from __future__ import annotations

import asyncio
import email.utils
import random
import threading
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

import httpx


class EmbeddingTransportError(RuntimeError):
    def __init__(self, message: str, *, status_code: Optional[int] = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class EmbeddingTransport:
    """Pooled sync+async transport used during the staged async migration.

    Existing retrieval/worker callers can keep their synchronous Embedder API.
    New code should use the async methods; #57 can then remove the sync hot path
    without changing the embedding contract/provider implementation again.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        max_attempts: int = 4,
        concurrency: int = 8,
        sync_client: Optional[httpx.Client] = None,
        async_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if timeout_seconds <= 0 or max_attempts < 1 or concurrency < 1:
            raise ValueError("embedding transport timeout/attempts/concurrency must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self.max_attempts = int(max_attempts)
        self._async_semaphore = asyncio.Semaphore(int(concurrency))
        self._sync_semaphore = threading.BoundedSemaphore(int(concurrency))
        limits = httpx.Limits(max_connections=max(16, concurrency * 2), max_keepalive_connections=max(8, concurrency))
        self._sync = sync_client or httpx.Client(timeout=self.timeout_seconds, limits=limits)
        self._async = async_client or httpx.AsyncClient(timeout=self.timeout_seconds, limits=limits)
        self._owns_sync = sync_client is None
        self._owns_async = async_client is None

    def post_json(self, url: str, *, headers: Mapping[str, str], payload: Any) -> tuple[dict[str, Any], int, float]:
        retries = 0
        started = time.perf_counter()
        for attempt in range(1, self.max_attempts + 1):
            try:
                with self._sync_semaphore:
                    response = self._sync.post(url, headers=dict(headers), json=payload, timeout=self.timeout_seconds)
                if response.status_code < 400:
                    return response.json(), retries, (time.perf_counter() - started) * 1000
                retryable = _retryable(response.status_code)
                if not retryable or attempt >= self.max_attempts:
                    raise EmbeddingTransportError(
                        f"Embedding provider HTTP {response.status_code}",
                        status_code=response.status_code,
                        retryable=retryable,
                    )
                retries += 1
                time.sleep(_retry_delay(response, attempt))
            except EmbeddingTransportError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                if attempt >= self.max_attempts:
                    raise EmbeddingTransportError(f"Embedding transport failed: {type(exc).__name__}", retryable=True) from None
                retries += 1
                time.sleep(_backoff(attempt))
        raise EmbeddingTransportError("Embedding transport exhausted retries")

    async def apost_json(self, url: str, *, headers: Mapping[str, str], payload: Any) -> tuple[dict[str, Any], int, float]:
        retries = 0
        started = time.perf_counter()
        for attempt in range(1, self.max_attempts + 1):
            try:
                async with self._async_semaphore:
                    response = await self._async.post(url, headers=dict(headers), json=payload, timeout=self.timeout_seconds)
                if response.status_code < 400:
                    return response.json(), retries, (time.perf_counter() - started) * 1000
                retryable = _retryable(response.status_code)
                if not retryable or attempt >= self.max_attempts:
                    raise EmbeddingTransportError(
                        f"Embedding provider HTTP {response.status_code}",
                        status_code=response.status_code,
                        retryable=retryable,
                    )
                retries += 1
                await asyncio.sleep(_retry_delay(response, attempt))
            except EmbeddingTransportError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                if attempt >= self.max_attempts:
                    raise EmbeddingTransportError(f"Embedding transport failed: {type(exc).__name__}", retryable=True) from None
                retries += 1
                await asyncio.sleep(_backoff(attempt))
        raise EmbeddingTransportError("Embedding transport exhausted retries")

    def close(self) -> None:
        if self._owns_sync:
            self._sync.close()

    async def aclose(self) -> None:
        if self._owns_async:
            await self._async.aclose()


def _retryable(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429} or 500 <= status_code <= 599


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    raw = response.headers.get("Retry-After")
    if raw:
        try:
            return min(60.0, max(0.0, float(raw)))
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(raw)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return min(60.0, max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds()))
            except (TypeError, ValueError, OverflowError):
                pass
    return _backoff(attempt)


def _backoff(attempt: int) -> float:
    base = min(8.0, 0.5 * (2 ** max(0, attempt - 1)))
    return base * (0.75 + random.random() * 0.5)
