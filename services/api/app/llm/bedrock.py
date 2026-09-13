from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional

from .contracts import ModelCapabilities, ModelEndpoint, ModelUsage, UsageMixin


class BedrockAdapter(UsageMixin):
    """Provider-neutral adapter over Bedrock Runtime Converse."""

    def __init__(self, endpoint: ModelEndpoint, *, client: Any = None) -> None:
        super().__init__()
        self.endpoint = endpoint
        if client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise RuntimeError("Bedrock provider requires boto3/botocore (install ragbot[s3])") from exc
            client = boto3.client(
                "bedrock-runtime",
                region_name=endpoint.region or "us-east-1",
                config=Config(
                    connect_timeout=endpoint.timeout_seconds,
                    read_timeout=endpoint.timeout_seconds,
                    retries={"max_attempts": endpoint.max_attempts, "mode": "adaptive"},
                    max_pool_connections=max(10, endpoint.concurrency),
                ),
            )
        self._client = client

    @property
    def provider_id(self) -> str:
        return "bedrock"

    @property
    def model_id(self) -> str:
        return self.endpoint.model_id

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.endpoint.capabilities

    @property
    def enabled(self) -> bool:
        return bool(self.model_id)

    async def chat_json(
        self,
        system: str,
        user: str,
        schema: Dict[str, Any],
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("BEDROCK_MODEL is not set")
        system_text = (
            f"{system}\n\nReturn only JSON matching this JSON Schema:\n{json.dumps(schema, separators=(',', ':'))}"
        )
        kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "system": [{"text": system_text}],
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {"temperature": temperature, "maxTokens": max_output_tokens or 4096},
        }
        response = await asyncio.to_thread(self._client.converse, **kwargs)
        self._record_usage(_usage(response.get("usage")))
        return json.loads(_response_text(response))

    async def stream_text(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        # Keep the interface source-compatible. Capability-aware routing rejects
        # streaming before selecting Bedrock until a converse_stream adapter is promoted.
        response = await asyncio.to_thread(
            self._client.converse,
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"temperature": temperature, "maxTokens": max_output_tokens or 4096},
        )
        self._record_usage(_usage(response.get("usage")))
        text = _response_text(response)
        if text:
            yield text

    async def web_search(
        self,
        query: str,
        allowed_domains: Optional[List[str]] = None,
        recency_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return []

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


def _response_text(response: Dict[str, Any]) -> str:
    content = (((response.get("output") or {}).get("message") or {}).get("content") or [])
    return "".join(str(block.get("text") or "") for block in content if "text" in block)


def _usage(raw: Any) -> Optional[ModelUsage]:
    if not isinstance(raw, dict):
        return None
    return ModelUsage(
        input_tokens=int(raw.get("inputTokens", 0) or 0),
        output_tokens=int(raw.get("outputTokens", 0) or 0),
    )
