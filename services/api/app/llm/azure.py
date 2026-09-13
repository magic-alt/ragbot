from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from .client import OpenAIClient
from .contracts import ModelEndpoint
from .transport import HttpModelTransport


class AzureOpenAIAdapter(OpenAIClient):
    def __init__(self, endpoint: ModelEndpoint, *, transport: Optional[HttpModelTransport] = None) -> None:
        if not endpoint.base_url or not endpoint.model_id:
            raise ValueError("Azure OpenAI requires endpoint base_url and deployment/model ID")
        super().__init__(endpoint=endpoint, transport=transport)

    @property
    def provider_id(self) -> str:
        return "azure-openai"

    def _build_headers(self) -> Dict[str, str]:
        return {"api-key": self.endpoint.api_key, "Content-Type": "application/json"}

    def _chat_url(self) -> str:
        return (
            f"{self.base_url}/openai/deployments/{self.model_id}/chat/completions"
            f"?api-version={self.endpoint.api_version or '2024-10-21'}"
        )

    async def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Ragbot currently uses chat-completions for structured generation. The
        # native OpenAI Responses web-search path is not assumed for Azure.
        if path != "/v1/chat/completions":
            raise RuntimeError(f"Azure OpenAI adapter does not expose compatibility path: {path}")
        return await self._transport.post_json(
            self._chat_url(),
            headers=self._build_headers(),
            json=payload,
        )

    async def _stream_chat(self, payload: Dict[str, Any]) -> AsyncIterator[str]:
        import json

        async for line in self._transport.stream_lines(
            self._chat_url(),
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
                from .client import _openai_usage
                self._record_usage(_openai_usage(event.get("usage")))
            if event.get("choices"):
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
