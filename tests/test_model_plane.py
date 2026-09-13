from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import httpx
import pytest

from services.api.app.llm.anthropic import AnthropicAdapter
from services.api.app.llm.bedrock import BedrockAdapter
from services.api.app.llm.contracts import ModelCapabilities, ModelEndpoint, ModelUsage, UsageMixin
from services.api.app.llm.gemini import GeminiAdapter
from services.api.app.llm.provider import endpoint_from_env
from services.api.app.llm.router import CostTracker, ModelCapabilityError, ModelRouter, model_task_scope
from services.api.app.llm.transport import HttpModelTransport


class FakeProvider(UsageMixin):
    def __init__(self, provider_id: str, model_id: str, capabilities: ModelCapabilities, result=None):
        super().__init__()
        self.provider_id = provider_id
        self.model_id = model_id
        self.capabilities = capabilities
        self.enabled = True
        self.result = result or {"ok": True}
        self.calls = []

    async def chat_json(self, system, user, schema, temperature=0.2, max_output_tokens=None):
        self.calls.append(("chat_json", system, user))
        self._record_usage(ModelUsage(input_tokens=10, output_tokens=4, cached_input_tokens=2))
        return dict(self.result)

    async def stream_text(self, system, user, temperature=0.2, max_output_tokens=None):
        self.calls.append(("stream_text", system, user))
        self._record_usage(ModelUsage(input_tokens=7, output_tokens=2))
        yield "ok"

    async def web_search(self, query, allowed_domains=None, recency_days=None):
        self.calls.append(("web_search", query))
        self._record_usage(ModelUsage(input_tokens=3, output_tokens=1))
        return [{"url": "https://example.com"}]


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def post_json(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response

    async def stream_lines(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if False:
            yield ""

    async def aclose(self):
        return None


def test_router_uses_request_context_task_without_mutating_provider_config():
    fast = FakeProvider("fast-provider", "fast-model", ModelCapabilities(structured_output=True, streaming=True, web_search=True))
    strong = FakeProvider("strong-provider", "strong-model", ModelCapabilities(structured_output=True, streaming=True))
    router = ModelRouter(fast, strong, routing_enabled=True)

    async def run():
        with model_task_scope(router, "synthesize"):
            result = await router.chat_json("system", "user", {"type": "object"})
        assert result == {"ok": True}

    asyncio.run(run())
    assert len(strong.calls) == 1
    assert not fast.calls
    summary = router.cost_tracker.summary()
    assert summary["total_tokens"] == 14
    assert "strong-provider:strong-model" in summary["by_model"]


def test_router_falls_back_by_capability_and_fails_before_request_when_unavailable():
    fast = FakeProvider("fast", "f", ModelCapabilities(structured_output=True, web_search=True))
    strong = FakeProvider("strong", "s", ModelCapabilities(structured_output=True, web_search=False))
    router = ModelRouter(fast, strong, routing_enabled=True)

    async def run():
        with model_task_scope(router, "synthesize"):
            result = await router.web_search("query")
        assert result[0]["url"] == "https://example.com"

    asyncio.run(run())
    assert fast.calls[0][0] == "web_search"

    no_web = ModelRouter(
        FakeProvider("a", "a", ModelCapabilities(structured_output=True)),
        FakeProvider("b", "b", ModelCapabilities(structured_output=True)),
        routing_enabled=True,
    )
    with pytest.raises(ModelCapabilityError):
        no_web.get_provider("web_search", capability="web_search")


def test_endpoint_from_env_supports_tier_specific_provider_without_environment_mutation(monkeypatch):
    monkeypatch.setenv("RAGBOT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "baseline")
    monkeypatch.setenv("RAGBOT_MODEL_STRONG_PROVIDER", "anthropic")
    monkeypatch.setenv("RAGBOT_MODEL_STRONG", "claude-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    before = dict(__import__("os").environ)
    endpoint = endpoint_from_env("STRONG")
    after = dict(__import__("os").environ)
    assert endpoint.provider_id == "anthropic"
    assert endpoint.model_id == "claude-test"
    assert before == after


def test_anthropic_adapter_normalizes_usage_and_json():
    transport = FakeTransport({
        "content": [{"type": "text", "text": '{"route":"doc_rag"}'}],
        "usage": {"input_tokens": 12, "output_tokens": 3, "cache_read_input_tokens": 5},
    })
    endpoint = ModelEndpoint(
        provider_id="anthropic",
        model_id="claude-test",
        base_url="https://api.anthropic.test",
        api_key="key",
        api_version="2023-06-01",
        capabilities=ModelCapabilities(structured_output=True, streaming=True),
    )
    adapter = AnthropicAdapter(endpoint, transport=transport)  # type: ignore[arg-type]
    result = asyncio.run(adapter.chat_json("system", "user", {"type": "object"}))
    assert result["route"] == "doc_rag"
    usage = adapter.consume_usage()
    assert usage == ModelUsage(input_tokens=12, output_tokens=3, cached_input_tokens=5)
    assert transport.calls[0][0].endswith("/v1/messages")


def test_gemini_adapter_uses_response_json_schema_and_usage_metadata():
    transport = FakeTransport({
        "candidates": [{"content": {"parts": [{"text": '{"ok":true}'}]}}],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 2, "thoughtsTokenCount": 1},
    })
    endpoint = ModelEndpoint(
        provider_id="gemini",
        model_id="gemini-test",
        base_url="https://gemini.test",
        api_key="key",
        capabilities=ModelCapabilities(structured_output=True, json_schema=True, streaming=True),
    )
    adapter = GeminiAdapter(endpoint, transport=transport)  # type: ignore[arg-type]
    result = asyncio.run(adapter.chat_json("system", "user", {"type": "object"}))
    assert result == {"ok": True}
    _url, kwargs = transport.calls[0]
    assert kwargs["json"]["generationConfig"]["responseJsonSchema"] == {"type": "object"}
    assert adapter.consume_usage() == ModelUsage(input_tokens=8, output_tokens=2, reasoning_tokens=1)


def test_bedrock_adapter_normalizes_converse_response():
    class Client:
        def converse(self, **kwargs):
            return {
                "output": {"message": {"content": [{"text": '{"ok":true}'}]}},
                "usage": {"inputTokens": 5, "outputTokens": 2},
            }

    endpoint = ModelEndpoint(
        provider_id="bedrock",
        model_id="bedrock-test",
        region="us-east-1",
        capabilities=ModelCapabilities(structured_output=True, streaming=False),
    )
    adapter = BedrockAdapter(endpoint, client=Client())
    result = asyncio.run(adapter.chat_json("system", "user", {"type": "object"}))
    assert result == {"ok": True}
    assert adapter.consume_usage() == ModelUsage(input_tokens=5, output_tokens=2)


def test_http_transport_retries_429_using_retry_after_zero():
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "rate"})
        return httpx.Response(200, json={"ok": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = HttpModelTransport(client=client, max_attempts=2, concurrency=1)

    async def run():
        try:
            assert await transport.post_json("https://example.test", json={}) == {"ok": True}
        finally:
            await client.aclose()

    asyncio.run(run())
    assert calls == 2


def test_cost_tracker_pricing_is_model_specific():
    provider = FakeProvider("openai", "priced", ModelCapabilities())
    tracker = CostTracker(pricing={"openai:priced": __import__("services.api.app.llm.router", fromlist=["ModelPrice"]).ModelPrice(input_per_million=1.0, output_per_million=2.0, cached_input_per_million=0.5)})
    record = tracker.record(
        task="synthesize",
        tier="strong",
        provider=provider,
        usage=ModelUsage(input_tokens=1000, cached_input_tokens=200, output_tokens=500),
    )
    assert record.estimated_cost_usd == pytest.approx(0.0019)
