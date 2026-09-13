from __future__ import annotations

import asyncio
import email.utils
import random
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Mapping, Optional

import httpx


class ModelTransportError(RuntimeError):
    def __init__(self, message: str, *, status_code: Optional[int] = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class HttpModelTransport:
    """Shared pooled HTTP transport with bounded concurrency and retry policy."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 60.0,
        max_attempts: int = 3,
        concurrency: int = 16,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("model timeout_seconds must be > 0")
        if max_attempts < 1:
            raise ValueError("model max_attempts must be >= 1")
        if concurrency < 1:
            raise ValueError("model concurrency must be >= 1")
        self.timeout_seconds = float(timeout_seconds)
        self.max_attempts = int(max_attempts)
        self._semaphore = asyncio.Semaphore(int(concurrency))
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds),
            limits=httpx.Limits(max_connections=max(16, concurrency * 2), max_keepalive_connections=max(8, concurrency)),
        )
        self._owns_client = client is None

    async def post_json(
        self,
        url: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        json: Any = None,
        params: Optional[Mapping[str, str]] = None,
    ) -> dict[str, Any]:
        response = await self.request("POST", url, headers=headers, json=json, params=params)
        try:
            return response.json()
        except ValueError as exc:
            raise ModelTransportError("Model provider returned invalid JSON", status_code=response.status_code) from exc

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        json: Any = None,
        params: Optional[Mapping[str, str]] = None,
    ) -> httpx.Response:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                async with self._semaphore:
                    response = await self._client.request(
                        method,
                        url,
                        headers=dict(headers or {}),
                        json=json,
                        params=params,
                        timeout=self.timeout_seconds,
                    )
                if response.status_code < 400:
                    return response
                retryable = _retryable_status(response.status_code)
                if not retryable or attempt >= self.max_attempts:
                    raise ModelTransportError(
                        f"Model provider HTTP {response.status_code}",
                        status_code=response.status_code,
                        retryable=retryable,
                    )
                await asyncio.sleep(_retry_delay(response, attempt))
            except ModelTransportError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt >= self.max_attempts:
                    raise ModelTransportError(
                        f"Model transport failed: {type(exc).__name__}",
                        retryable=True,
                    ) from None
                await asyncio.sleep(_backoff(attempt))
        raise ModelTransportError(f"Model transport failed: {type(last_exc).__name__ if last_exc else 'unknown'}")

    async def stream_lines(
        self,
        url: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        json: Any = None,
        params: Optional[Mapping[str, str]] = None,
    ) -> AsyncIterator[str]:
        # Streaming requests are never transparently replayed after response bytes
        # have been exposed to the caller. We only retry connection/open failures.
        for attempt in range(1, self.max_attempts + 1):
            try:
                async with self._semaphore:
                    async with self._client.stream(
                        "POST",
                        url,
                        headers=dict(headers or {}),
                        json=json,
                        params=params,
                        timeout=self.timeout_seconds,
                    ) as response:
                        if response.status_code >= 400:
                            retryable = _retryable_status(response.status_code)
                            if retryable and attempt < self.max_attempts:
                                delay = _retry_delay(response, attempt)
                            else:
                                raise ModelTransportError(
                                    f"Model provider HTTP {response.status_code}",
                                    status_code=response.status_code,
                                    retryable=retryable,
                                )
                        else:
                            async for line in response.aiter_lines():
                                yield line
                            return
                await asyncio.sleep(delay)
            except ModelTransportError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                if attempt >= self.max_attempts:
                    raise ModelTransportError(
                        f"Model stream failed: {type(exc).__name__}", retryable=True
                    ) from None
                await asyncio.sleep(_backoff(attempt))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _retryable_status(status_code: int) -> bool:
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
