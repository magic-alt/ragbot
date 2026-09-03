from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_REASONING_EFFORTS = {"none", "low", "medium", "high", "max"}


class OllamaAdapter:
    """LLM adapter for Ollama using its OpenAI-compatible API.

    Ragbot uses structured JSON for several agent nodes. Ollama's native API
    accepts ``format``/``options`` fields, while the OpenAI-compatible
    ``/v1/chat/completions`` endpoint accepts ``response_format`` and
    ``max_tokens``. Keep those request shapes separate so structured routing and
    synthesis work reliably with current Ollama models such as Qwen3.8.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")
        if timeout is None:
            timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
        self.timeout = float(timeout)

        configured_reasoning = reasoning_effort
        if configured_reasoning is None:
            configured_reasoning = os.getenv("OLLAMA_REASONING_EFFORT", "").strip() or None
        if configured_reasoning is not None:
            configured_reasoning = configured_reasoning.lower()
            if configured_reasoning not in _REASONING_EFFORTS:
                allowed = ", ".join(sorted(_REASONING_EFFORTS))
                raise ValueError(
                    f"Unsupported OLLAMA_REASONING_EFFORT={configured_reasoning!r}; "
                    f"expected one of: {allowed}"
                )
        self.reasoning_effort = configured_reasoning

    @property
    def enabled(self) -> bool:
        return True

    def _apply_completion_options(
        self,
        payload: Dict[str, Any],
        max_output_tokens: Optional[int],
    ) -> None:
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
            f"{system}\n\nYou MUST respond with valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}"
        )
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_with_schema},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": schema},
            },
            "stream": False,
        }
        self._apply_completion_options(payload, max_output_tokens)
        data = await self._post_json("/v1/chat/completions", payload)
        content = data["choices"][0]["message"]["content"]
        return _extract_json(content)

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
        }
        self._apply_completion_options(payload, max_output_tokens)
        url = f"{self.base_url}/v1/chat/completions"
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST", url, json=payload, timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[len("data: ") :].strip()
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if not event.get("choices"):
                            continue
                        delta = event["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama stream failed: {type(exc).__name__}") from None

    async def web_search(
        self,
        query: str,
        allowed_domains: Optional[List[str]] = None,
        recency_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        # Ollama does not support web search natively.
        return []

    async def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama API request failed: {type(exc).__name__}") from None
        return response.json()


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response text, handling markdown fences."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)
