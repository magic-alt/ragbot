from __future__ import annotations

import json
import os
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from .contracts import ModelCapabilities, ModelEndpoint, ModelUsage, UsageMixin
from .transport import HttpModelTransport

_REASONING_EFFORTS = {"none", "low", "medium", "high", "max"}


class OllamaAdapter(UsageMixin):
    """Ollama adapter over the OpenAI-compatible chat-completions surface."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        *,
        endpoint: Optional[ModelEndpoint] = None,
        transport: Optional[HttpModelTransport] = None,
    ) -> None:
        super().__init__()
        if endpoint is None:
            configured_reasoning = reasoning_effort
            if configured_reasoning is None:
                configured_reasoning = os.getenv("OLLAMA_REASONING_EFFORT", "").strip() or None
            endpoint = ModelEndpoint(
                provider_id="ollama",
                model_id=model or os.getenv("OLLAMA_MODEL", "llama3"),
                base_url=(base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/"),
                timeout_seconds=float(timeout if timeout is not None else os.getenv("OLLAMA_TIMEOUT_SECONDS", "300")),
                reasoning_effort=configured_reasoning,
                capabilities=ModelCapabilities(structured_output=True, json_schema=True, streaming=True, reasoning=True),
            )
        configured_reasoning = endpoint.reasoning_effort
        if configured_reasoning is not None:
            configured_reasoning = configured_reasoning.lower()
            if configured_reasoning not in _REASONING_EFFORTS:
                allowed = ", ".join(sorted(_REASONING_EFFORTS))
                raise ValueError(
                    f"Unsupported OLLAMA_REASONING_EFFORT={configured_reasoning!r}; expected one of: {allowed}"
                )
        self.endpoint = endpoint
        self.base_url = endpoint.base_url.rstrip("/")
        self.model = endpoint.model_id
        self.timeout = endpoint.timeout_seconds
        self.reasoning_effort = configured_reasoning
        self._transport = transport or HttpModelTransport(
            timeout_seconds=endpoint.timeout_seconds,
            max_attempts=endpoint.max_attempts,
            concurrency=endpoint.concurrency,
        )
        self._owns_transport = transport is None

    @property
    def provider_id(self) -> str:
        return "ollama"

    @property
    def model_id(self) -> str:
        return self.model

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.endpoint.capabilities

    @property
    def enabled(self) -> bool:
        return True

    def _apply_completion_options(self, payload: Dict[str, Any], max_output_tokens: Optional[int]) -> None:
        if max_output_tokens:
            payload["max_tokens"] = max_output_tokens
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

    async def chat_json(
        self,
        system: str,
        user: str,
        schema: Dict[str, Any],
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        system_with_schema = (
            f"{system}\n\nYou MUST respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        )
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_with_schema},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
            "stream": False,
        }
        self._apply_completion_options(payload, max_output_tokens)
        data = await self._post_json("/v1/chat/completions", payload)
        self._record_usage(_usage(data.get("usage")))
        return _extract_json(data["choices"][0]["message"]["content"])

    async def stream_text(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        self._apply_completion_options(payload, max_output_tokens)
        async for line in self._transport.stream_lines(
            f"{self.base_url}/v1/chat/completions", json=payload
        ):
            if not line or not line.startswith("data: "):
                continue
            raw = line[len("data: "):].strip()
            if raw == "[DONE]":
                break
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                self._record_usage(_usage(event.get("usage")))
            if not event.get("choices"):
                continue
            content = event["choices"][0].get("delta", {}).get("content")
            if content:
                yield content

    async def web_search(
        self,
        query: str,
        allowed_domains: Optional[List[str]] = None,
        recency_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return []

    async def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._transport.post_json(f"{self.base_url}{path}", json=payload)

    async def aclose(self) -> None:
        if self._owns_transport:
            await self._transport.aclose()


def _usage(raw: Any) -> Optional[ModelUsage]:
    if not isinstance(raw, dict):
        return None
    return ModelUsage(
        input_tokens=int(raw.get("prompt_tokens", 0) or 0),
        output_tokens=int(raw.get("completion_tokens", 0) or 0),
    )


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)
