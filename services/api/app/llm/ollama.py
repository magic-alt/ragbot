from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional

import requests

logger = logging.getLogger(__name__)


class OllamaAdapter:
    """LLM adapter for Ollama using the OpenAI-compatible API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return True

    def chat_json(
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
            "format": "json",
            "stream": False,
        }
        if max_output_tokens:
            payload["options"] = {"num_predict": max_output_tokens}
        data = self._post_json("/v1/chat/completions", payload)
        content = data["choices"][0]["message"]["content"]
        return _extract_json(content)

    def stream_text(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "stream": True,
        }
        if max_output_tokens:
            payload["options"] = {"num_predict": max_output_tokens}
        url = f"{self.base_url}/v1/chat/completions"
        try:
            response = requests.post(url, json=payload, stream=True, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama stream failed: {type(exc).__name__}") from None

        with response:
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):].strip()
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

    def web_search(
        self,
        query: str,
        allowed_domains: Optional[List[str]] = None,
        recency_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        # Ollama does not support web search natively
        return []

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama API request failed: {type(exc).__name__}") from None
        return response.json()


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response text, handling markdown fences."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)
