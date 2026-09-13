from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional
from urllib.parse import quote

from .contracts import ModelCapabilities, ModelEndpoint, ModelUsage, UsageMixin
from .transport import HttpModelTransport


class GeminiAdapter(UsageMixin):
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
        return self.endpoint.provider_id

    @property
    def model_id(self) -> str:
        return self.endpoint.model_id

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.endpoint.capabilities

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint.api_key)

    def _request_target(self, method: str) -> tuple[str, dict[str, str], dict[str, str]]:
        if self.endpoint.provider_id == "vertex":
            if not self.endpoint.project:
                raise RuntimeError("Vertex AI requires GOOGLE_CLOUD_PROJECT")
            location = self.endpoint.region or "global"
            root = self.endpoint.base_url.rstrip("/")
            if root == "https://generativelanguage.googleapis.com":
                root = f"https://{location}-aiplatform.googleapis.com"
            url = (
                f"{root}/v1/projects/{quote(self.endpoint.project)}/locations/{quote(location)}"
                f"/publishers/google/models/{quote(self.model_id)}:{method}"
            )
            return url, {"Authorization": f"Bearer {self.endpoint.api_key}", "Content-Type": "application/json"}, {}
        url = f"{self.endpoint.base_url.rstrip('/')}/v1beta/models/{quote(self.model_id)}:{method}"
        return url, {"Content-Type": "application/json"}, {"key": self.endpoint.api_key}

    async def chat_json(
        self,
        system: str,
        user: str,
        schema: Dict[str, Any],
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Gemini/Vertex credential is not configured")
        url, headers, params = self._request_target("generateContent")
        generation_config: Dict[str, Any] = {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
        }
        if max_output_tokens:
            generation_config["maxOutputTokens"] = max_output_tokens
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation_config,
        }
        data = await self._transport.post_json(url, headers=headers, params=params, json=payload)
        self._record_usage(_usage(data.get("usageMetadata")))
        return json.loads(_text(data))

    async def stream_text(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        if not self.enabled:
            raise RuntimeError("Gemini/Vertex credential is not configured")
        url, headers, params = self._request_target("streamGenerateContent")
        params = {**params, "alt": "sse"}
        generation_config: Dict[str, Any] = {"temperature": temperature}
        if max_output_tokens:
            generation_config["maxOutputTokens"] = max_output_tokens
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation_config,
        }
        async for line in self._transport.stream_lines(url, headers=headers, params=params, json=payload):
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if event.get("usageMetadata"):
                self._record_usage(_usage(event["usageMetadata"]))
            text = _text(event)
            if text:
                yield text

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


def _text(data: Dict[str, Any]) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    return "".join(str(part.get("text") or "") for part in parts)


def _usage(raw: Any) -> Optional[ModelUsage]:
    if not isinstance(raw, dict):
        return None
    return ModelUsage(
        input_tokens=int(raw.get("promptTokenCount", 0) or 0),
        output_tokens=int(raw.get("candidatesTokenCount", 0) or 0),
        cached_input_tokens=int(raw.get("cachedContentTokenCount", 0) or 0),
        reasoning_tokens=int(raw.get("thoughtsTokenCount", 0) or 0),
    )
