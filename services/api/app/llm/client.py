from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

from .contracts import ModelCapabilities, ModelEndpoint, ModelUsage, UsageMixin
from .transport import HttpModelTransport


class OpenAIClient(UsageMixin):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        web_model: Optional[str] = None,
        timeout: int = 30,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        *,
        endpoint: Optional[ModelEndpoint] = None,
        transport: Optional[HttpModelTransport] = None,
    ) -> None:
        super().__init__()
        if endpoint is None:
            endpoint = ModelEndpoint(
                provider_id="openai",
                model_id=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                base_url=(base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/"),
                api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
                organization=organization or os.getenv("OPENAI_ORGANIZATION", ""),
                project=project or os.getenv("OPENAI_PROJECT", ""),
                timeout_seconds=float(timeout),
                capabilities=ModelCapabilities(
                    structured_output=True,
                    json_schema=True,
                    streaming=True,
                    tools=True,
                    web_search=True,
                    vision=True,
                    reasoning=True,
                    batch=True,
                ),
            )
        self.endpoint = endpoint
        self.api_key = endpoint.api_key
        self.base_url = endpoint.base_url.rstrip("/")
        self.model = endpoint.model_id
        self.web_model = web_model or os.getenv("OPENAI_WEB_MODEL", self.model)
        self.organization = endpoint.organization
        self.project = endpoint.project
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
        return bool(self.api_key)

    async def chat_json(
        self,
        system: str,
        user: str,
        schema: Dict[str, Any],
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("OPENAI_API_KEY is not set")
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "rag_response",
                    "schema": schema,
                    "strict": True,
                },
            },
        }
        if max_output_tokens:
            payload["max_tokens"] = max_output_tokens
        data = await self._post_json("/v1/chat/completions", payload)
        self._record_usage(_openai_usage(data.get("usage")))
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)

    async def stream_text(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        if not self.enabled:
            raise RuntimeError("OPENAI_API_KEY is not set")
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
        if max_output_tokens:
            payload["max_tokens"] = max_output_tokens
        async for delta in self._stream_chat(payload):
            if delta:
                yield delta

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project
        return headers

    async def web_search(
        self,
        query: str,
        allowed_domains: Optional[List[str]] = None,
        recency_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            raise RuntimeError("OPENAI_API_KEY is not set")
        tool: Dict[str, Any] = {"type": "web_search"}
        if allowed_domains:
            tool["filters"] = {"allowed_domains": allowed_domains}
        payload = {
            "model": self.web_model,
            "tools": [tool],
            "tool_choice": "auto",
            "input": query,
            "include": ["web_search_call.action.sources"],
        }
        data = await self._post_json("/v1/responses", payload)
        self._record_usage(_openai_usage(data.get("usage")))
        return _extract_web_sources(data)

    async def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._transport.post_json(
            f"{self.base_url}{path}",
            headers=self._build_headers(),
            json=payload,
        )

    async def _stream_chat(self, payload: Dict[str, Any]) -> AsyncIterator[str]:
        async for line in self._transport.stream_lines(
            f"{self.base_url}/v1/chat/completions",
            headers=self._build_headers(),
            json=payload,
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
                self._record_usage(_openai_usage(event.get("usage")))
            if not event.get("choices"):
                continue
            delta = event["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content

    async def aclose(self) -> None:
        if self._owns_transport:
            await self._transport.aclose()


def _openai_usage(raw: Any) -> Optional[ModelUsage]:
    if not isinstance(raw, dict):
        return None
    input_details = raw.get("prompt_tokens_details") or raw.get("input_tokens_details") or {}
    output_details = raw.get("completion_tokens_details") or raw.get("output_tokens_details") or {}
    return ModelUsage(
        input_tokens=int(raw.get("prompt_tokens", raw.get("input_tokens", 0)) or 0),
        output_tokens=int(raw.get("completion_tokens", raw.get("output_tokens", 0)) or 0),
        cached_input_tokens=int(input_details.get("cached_tokens", 0) or 0),
        reasoning_tokens=int(output_details.get("reasoning_tokens", 0) or 0),
    )


def _extract_web_sources(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for item in data.get("output", []):
        if item.get("type") == "web_search_call":
            action = item.get("action") or {}
            for source in action.get("sources", []) or []:
                sources.append(_normalize_source(source))
        if item.get("type") == "message":
            for content in item.get("content", []) or []:
                text = content.get("text") or ""
                for ann in content.get("annotations", []) or []:
                    if ann.get("type") != "url_citation":
                        continue
                    sources.append(_normalize_source({
                        "url": ann.get("url", ""),
                        "title": ann.get("title", ""),
                        "snippet": text,
                        "published_at": ann.get("published_at") or ann.get("date"),
                        "score": ann.get("score"),
                    }))
    return _dedupe_sources(sources)


def _normalize_source(source: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "url": source.get("url", ""),
        "title": source.get("title", ""),
        "snippet": source.get("snippet", ""),
        "score": source.get("score"),
        "published_at": source.get("published_at") or source.get("date"),
    }


def _dedupe_sources(sources: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output: List[Dict[str, Any]] = []
    for source in sources:
        url = source.get("url") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        output.append(source)
    return output
