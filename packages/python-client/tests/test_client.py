from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from ragbot_client import AsyncRagbotClient, RagbotApiError, RagbotClient


def test_sync_client_parses_typed_error_and_request_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/search"
        return httpx.Response(
            409,
            headers={"X-Request-ID": "req-header"},
            json={
                "error": {
                    "code": "conflict",
                    "message": "plan not available",
                    "request_id": "req-body",
                    "retryable": False,
                    "details": {"plan": "qdrant_dense_sparse"},
                }
            },
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as raw:
        client = RagbotClient("http://ragbot.test", client=raw)
        with pytest.raises(RagbotApiError) as excinfo:
            client.search({"query": "x", "tenant_id": "t", "user_id": "u"})
    assert excinfo.value.status_code == 409
    assert excinfo.value.code == "conflict"
    assert excinfo.value.request_id == "req-body"
    assert excinfo.value.retryable is False


def test_sync_source_iterator_follows_keyset_cursor() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        cursor = request.url.params.get("cursor")
        if cursor is None:
            return httpx.Response(
                200,
                json={
                    "total": 2,
                    "next_cursor": "cursor-2",
                    "sources": [{"source_id": "s1", "tenant_id": "t", "source_type": "pdf", "name": "one", "config": {}}],
                },
            )
        assert cursor == "cursor-2"
        return httpx.Response(
            200,
            json={
                "total": 2,
                "next_cursor": None,
                "sources": [{"source_id": "s2", "tenant_id": "t", "source_type": "pdf", "name": "two", "config": {}}],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as raw:
        client = RagbotClient("http://ragbot.test", client=raw)
        source_ids = [item["source_id"] for item in client.iter_sources(tenant_id="t", page_size=1)]
    assert source_ids == ["s1", "s2"]
    assert len(calls) == 2


def test_async_client_propagates_task_cancellation() -> None:
    started = asyncio.Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.sleep(10)
        return httpx.Response(200, json={})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as raw:
            client = AsyncRagbotClient("http://ragbot.test", client=raw)
            task = asyncio.create_task(
                client.search({"query": "x", "tenant_id": "t", "user_id": "u"})
            )
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())


def test_sync_sse_parser_handles_multiline_data() -> None:
    payload = (
        'event: token\n'
        'data: {"request_id":"r1",\n'
        'data: "delta":"hello"}\n\n'
        'event: final\n'
        'data: {"request_id":"r1","answer":"hello","citations":[],"confidence":"high","followups":[]}\n\n'
    ).encode()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload, headers={"content-type": "text/event-stream"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as raw:
        client = RagbotClient("http://ragbot.test", client=raw)
        events = list(client.chat_stream({"query": "x", "tenant_id": "t", "user_id": "u"}))
    assert events[0]["event"] == "token"
    assert events[0]["data"]["delta"] == "hello"
    assert events[1]["event"] == "final"
