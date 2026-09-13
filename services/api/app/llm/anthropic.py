from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from .contracts import ModelCapabilities, ModelEndpoint, ModelUsage, UsageMixin
from .transport import HttpModelTransport


class AnthropicAdapter(UsageMixin):
    def __init__(self, endpoint: ModelEndpoint, *, transport: Optional[HttpModelTransport] = None) -> None:
        super().__init__()
        self.endpoint = endpoint
        self._transport = transport or HttpModelTransport(
            timeout_seconds=endpoint.timeout_seconds,
            max_attempts=endpoint.max_attempts,
            concurrency=endpoint.concurrency,
        )
        self._owns_transport = transport is None

    @property
    def provider_id(self) -> str:
        return "anthropic"

    @property
    def model_id(self) -> str:
        return self.endpoint.model_id

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.endpoint.capabilities

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint.api_key)

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.endpoint.api_key,
            "anthropic-version": self.endpoint.api_version or "2023-06-01",
            "content-type": "application/json",
        }

    async def chat_json(
        self,
        system: str,
        user: str,
        schema: Dict[str, Any],
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        structured_system = (
            f"{system}\n\nReturn only JSON matching this JSON Schema:\n{json.dumps(schema, separators=(',', ':'))}"
        )
        payload = {
            "model": self.model_id,
            "system": structured_system,
            "messages": [{"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_output_tokens or 4096,
        }
        data = await self._transport.post_json(
            f"{self.endpoint.base_url}/v1/messages",
            headers=self._headers(),
            json=payload,
        )
        self._record_usage(_usage(data.get("usage")))
        text = "".join(
            str(block.get("text") or "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        return _extract_json(text)

    async def stream_text(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        if not self.enabled:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        payload = {
            "model": self.model_id,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_output_tokens or 4096,
            "stream": True,
        }
        input_tokens = 0
        output_tokens = 0
        async for line in self._transport.stream_lines(
            f"{self.endpoint.base_url}/v1/messages",
            headers=self._headers(),
            json=payload,
        ):
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            event_type = event.get("type")
            if event_type == "message_start":
                input_tokens = int(((event.get("message") or {}).get("usage") or {}).get("input_tokens", 0) or 0)
            elif event_type == "message_delta":
                output_tokens = int((event.get("usage") or {}).get("output_tokens", output_tokens) or output_tokens)
            elif event_type == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield str(delta["text"])
        self._record_usage(ModelUsage(input_tokens=input_tokens, output_tokens=output_tokens))

    async def web_search(
        self,
        query: str,
        allowed_domains: Optional[List[str]] = None,
        recency_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return []

    async def aclose(self) -> None:
        if self._owns_transport:
            await self._transport.aclose()


def _usage(raw: Any) -> Optional[ModelUsage]:
    if not isinstance(raw, dict):
        return None
    cache = int(raw.get("cache_read_input_tokens", 0) or 0)
    return ModelUsage(
        input_tokens=int(raw.get("input_tokens", 0) or 0),
        output_tokens=int(raw.get("output_tokens", 0) or 0),
        cached_input_tokens=cache,
    )


def _extract_json(text: str) -> Dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1] if "\n" in value else value
        value = value.rsplit("```", 1)[0].strip()
    return json.loads(value)
