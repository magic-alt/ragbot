from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, Optional, TypeVar, cast

import httpx

from .types import ChatRequest, ChatResponse, JobPage, SSEEvent, SearchRequest, SearchResponse, Source, SourcePage

T = TypeVar("T")
_RETRYABLE_STATUS = {429, 502, 503, 504}


class RagbotApiError(RuntimeError):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        request_id: Optional[str] = None,
        retryable: bool = False,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = str(code)
        self.request_id = request_id
        self.retryable = bool(retryable)
        self.details = details


class RagbotClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "RagbotClient":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def search(self, payload: SearchRequest, *, timeout: Optional[float] = None) -> SearchResponse:
        return cast(
            SearchResponse,
            self._request("POST", "/v1/search", json_body=dict(payload), timeout=timeout, retryable=True),
        )

    def chat(self, payload: ChatRequest, *, timeout: Optional[float] = None) -> ChatResponse:
        return cast(
            ChatResponse,
            self._request("POST", "/v1/chat", json_body=dict(payload), timeout=timeout, retryable=True),
        )

    def get_source(self, source_id: str, *, timeout: Optional[float] = None) -> Source:
        return cast(Source, self._request("GET", f"/v1/sources/{source_id}", timeout=timeout))

    def list_sources(
        self,
        *,
        tenant_id: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> SourcePage:
        params = _params(tenant_id=tenant_id, limit=limit, cursor=cursor)
        return cast(SourcePage, self._request("GET", "/v1/sources", params=params, timeout=timeout))

    def iter_sources(
        self,
        *,
        tenant_id: Optional[str] = None,
        page_size: int = 100,
        timeout: Optional[float] = None,
    ) -> Iterator[Source]:
        cursor: Optional[str] = None
        while True:
            page = self.list_sources(
                tenant_id=tenant_id,
                limit=page_size,
                cursor=cursor,
                timeout=timeout,
            )
            yield from page["sources"]
            cursor = page.get("next_cursor")
            if not cursor:
                return

    def list_jobs(
        self,
        *,
        tenant_id: Optional[str] = None,
        source_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> JobPage:
        params = _params(
            tenant_id=tenant_id,
            source_id=source_id,
            status=status,
            limit=limit,
            cursor=cursor,
        )
        return cast(
            JobPage,
            self._request("GET", "/v1/catalog/jobs", params=params, timeout=timeout),
        )

    def chat_stream(
        self,
        payload: ChatRequest,
        *,
        timeout: Optional[float] = None,
    ) -> Iterator[SSEEvent]:
        headers = self._headers()
        body = {**dict(payload), "stream": True}
        with self._client.stream(
            "POST",
            f"{self.base_url}/v1/chat",
            headers=headers,
            json=body,
            timeout=timeout if timeout is not None else self.timeout,
        ) as response:
            if not response.is_success:
                raise _api_error(response)
            yield from _parse_sse_lines(response.iter_lines())

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
        retryable: bool = True,
        idempotency_key: Optional[str] = None,
    ) -> Any:
        attempts = self.max_retries + 1 if retryable or idempotency_key else 1
        last_error: Optional[RagbotApiError] = None
        for attempt in range(attempts):
            try:
                response = self._client.request(
                    method,
                    f"{self.base_url}{path}",
                    headers=self._headers(idempotency_key=idempotency_key),
                    json=json_body,
                    params=params,
                    timeout=timeout if timeout is not None else self.timeout,
                )
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as exc:
                if attempt + 1 >= attempts:
                    raise
                _sleep_backoff(attempt, None)
                continue
            if response.is_success:
                if response.status_code == 204:
                    return None
                return response.json()
            error = _api_error(response)
            last_error = error
            if not error.retryable or attempt + 1 >= attempts:
                raise error
            _sleep_backoff(attempt, response.headers.get("Retry-After"))
        if last_error is not None:  # pragma: no cover
            raise last_error
        raise RuntimeError("request failed without a response")  # pragma: no cover

    def _headers(self, *, idempotency_key: Optional[str] = None) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers


class AsyncRagbotClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=self.timeout)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "AsyncRagbotClient":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        await self.aclose()

    async def search(self, payload: SearchRequest, *, timeout: Optional[float] = None) -> SearchResponse:
        return cast(
            SearchResponse,
            await self._request("POST", "/v1/search", json_body=dict(payload), timeout=timeout, retryable=True),
        )

    async def chat(self, payload: ChatRequest, *, timeout: Optional[float] = None) -> ChatResponse:
        return cast(
            ChatResponse,
            await self._request("POST", "/v1/chat", json_body=dict(payload), timeout=timeout, retryable=True),
        )

    async def list_sources(
        self,
        *,
        tenant_id: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> SourcePage:
        return cast(
            SourcePage,
            await self._request(
                "GET",
                "/v1/sources",
                params=_params(tenant_id=tenant_id, limit=limit, cursor=cursor),
                timeout=timeout,
            ),
        )

    async def iter_sources(
        self,
        *,
        tenant_id: Optional[str] = None,
        page_size: int = 100,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[Source]:
        cursor: Optional[str] = None
        while True:
            page = await self.list_sources(
                tenant_id=tenant_id,
                limit=page_size,
                cursor=cursor,
                timeout=timeout,
            )
            for source in page["sources"]:
                yield source
            cursor = page.get("next_cursor")
            if not cursor:
                return

    async def list_jobs(
        self,
        *,
        tenant_id: Optional[str] = None,
        source_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> JobPage:
        return cast(
            JobPage,
            await self._request(
                "GET",
                "/v1/catalog/jobs",
                params=_params(
                    tenant_id=tenant_id,
                    source_id=source_id,
                    status=status,
                    limit=limit,
                    cursor=cursor,
                ),
                timeout=timeout,
            ),
        )

    async def chat_stream(
        self,
        payload: ChatRequest,
        *,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[SSEEvent]:
        body = {**dict(payload), "stream": True}
        async with self._client.stream(
            "POST",
            f"{self.base_url}/v1/chat",
            headers=self._headers(),
            json=body,
            timeout=timeout if timeout is not None else self.timeout,
        ) as response:
            if not response.is_success:
                raw = await response.aread()
                raise _api_error_from_bytes(response.status_code, response.headers, raw)
            async for event in _parse_sse_async(response.aiter_lines()):
                yield event

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
        retryable: bool = True,
        idempotency_key: Optional[str] = None,
    ) -> Any:
        attempts = self.max_retries + 1 if retryable or idempotency_key else 1
        last_error: Optional[RagbotApiError] = None
        for attempt in range(attempts):
            try:
                response = await self._client.request(
                    method,
                    f"{self.base_url}{path}",
                    headers=self._headers(idempotency_key=idempotency_key),
                    json=json_body,
                    params=params,
                    timeout=timeout if timeout is not None else self.timeout,
                )
            except asyncio.CancelledError:
                raise
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout):
                if attempt + 1 >= attempts:
                    raise
                await asyncio.sleep(_backoff_seconds(attempt, None))
                continue
            if response.is_success:
                if response.status_code == 204:
                    return None
                return response.json()
            error = _api_error(response)
            last_error = error
            if not error.retryable or attempt + 1 >= attempts:
                raise error
            await asyncio.sleep(_backoff_seconds(attempt, response.headers.get("Retry-After")))
        if last_error is not None:  # pragma: no cover
            raise last_error
        raise RuntimeError("request failed without a response")  # pragma: no cover

    def _headers(self, *, idempotency_key: Optional[str] = None) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers


def _api_error(response: httpx.Response) -> RagbotApiError:
    return _api_error_from_bytes(response.status_code, response.headers, response.content)


def _api_error_from_bytes(status_code: int, headers: Any, raw: bytes) -> RagbotApiError:
    try:
        payload = json.loads(raw.decode("utf-8")) if raw else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = {}
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        return RagbotApiError(
            status_code=status_code,
            code=str(error.get("code") or "http_error"),
            message=str(error.get("message") or f"HTTP {status_code}"),
            request_id=error.get("request_id") or headers.get("X-Request-ID"),
            retryable=bool(error.get("retryable", status_code in _RETRYABLE_STATUS)),
            details=error.get("details"),
        )
    detail = payload.get("detail") if isinstance(payload, dict) else None
    return RagbotApiError(
        status_code=status_code,
        code="http_error",
        message=str(detail or f"HTTP {status_code}"),
        request_id=headers.get("X-Request-ID"),
        retryable=status_code in _RETRYABLE_STATUS,
        details=detail,
    )


def _params(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _sleep_backoff(attempt: int, retry_after: Optional[str]) -> None:
    time.sleep(_backoff_seconds(attempt, retry_after))


def _backoff_seconds(attempt: int, retry_after: Optional[str]) -> float:
    if retry_after:
        try:
            return max(0.0, min(float(retry_after), 30.0))
        except ValueError:
            pass
    return min(5.0, 0.25 * (2**attempt)) + random.random() * 0.05


def _parse_sse_lines(lines: Iterator[str]) -> Iterator[SSEEvent]:
    current_event = "message"
    data_lines: list[str] = []
    for line in lines:
        if line == "":
            if data_lines:
                yield _sse_event(current_event, data_lines)
            current_event = "message"
            data_lines = []
            continue
        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield _sse_event(current_event, data_lines)


async def _parse_sse_async(lines: AsyncIterator[str]) -> AsyncIterator[SSEEvent]:
    current_event = "message"
    data_lines: list[str] = []
    async for line in lines:
        if line == "":
            if data_lines:
                yield _sse_event(current_event, data_lines)
            current_event = "message"
            data_lines = []
            continue
        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield _sse_event(current_event, data_lines)


def _sse_event(event: str, data_lines: list[str]) -> SSEEvent:
    raw = "\n".join(data_lines)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"raw": raw}
    if not isinstance(data, dict):
        data = {"value": data}
    return {"event": event, "data": data}
