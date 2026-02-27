from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, Iterator, List, Optional

import requests

logger = logging.getLogger(__name__)


class OpenAIClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        web_model: Optional[str] = None,
        timeout: int = 30,
        organization: Optional[str] = None,
        project: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.web_model = web_model or os.getenv("OPENAI_WEB_MODEL", self.model)
        self.timeout = timeout
        self.organization = organization or os.getenv("OPENAI_ORGANIZATION")
        self.project = project or os.getenv("OPENAI_PROJECT")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def chat_json(
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
        data = self._post_json("/v1/chat/completions", payload)
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)

    def stream_text(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> Iterator[str]:
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
        }
        if max_output_tokens:
            payload["max_tokens"] = max_output_tokens
        for delta in self._stream_chat(payload):
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

    def web_search(
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
        data = self._post_json("/v1/responses", payload)
        return _extract_web_sources(data)

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = self._build_headers()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"LLM API request failed: {type(exc).__name__}: {_sanitize_error(exc)}") from None
        return response.json()

    def _stream_chat(self, payload: Dict[str, Any]) -> Iterator[str]:
        url = f"{self.base_url}/v1/chat/completions"
        headers = self._build_headers()
        try:
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"LLM API stream failed: {type(exc).__name__}: {_sanitize_error(exc)}") from None

        with response:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if not line.startswith("data: "):
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
                    source = {
                        "url": ann.get("url", ""),
                        "title": ann.get("title", ""),
                        "snippet": text,
                        "published_at": ann.get("published_at") or ann.get("date"),
                        "score": ann.get("score"),
                    }
                    sources.append(_normalize_source(source))
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


def _sanitize_error(exc: Exception) -> str:
    msg = str(exc)
    if hasattr(exc, "response") and exc.response is not None:
        return f"HTTP {exc.response.status_code}"
    for prefix in ("Bearer ", "sk-", "key-"):
        while prefix in msg:
            start = msg.index(prefix)
            end = msg.find(" ", start + len(prefix))
            if end == -1:
                end = msg.find("'", start + len(prefix))
            if end == -1:
                end = len(msg)
            msg = msg[:start] + "[REDACTED]" + msg[end:]
    return msg

