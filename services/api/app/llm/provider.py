from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, runtime_checkable

from typing import Protocol


@runtime_checkable
class ModelProvider(Protocol):
    @property
    def enabled(self) -> bool: ...

    def chat_json(
        self,
        system: str,
        user: str,
        schema: Dict[str, Any],
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]: ...

    def stream_text(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> Iterator[str]: ...

    def web_search(
        self,
        query: str,
        allowed_domains: Optional[List[str]] = None,
        recency_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]: ...


@dataclass
class ModelCapabilities:
    supports_json_schema: bool = True
    supports_streaming: bool = True
    supports_web_search: bool = False
    max_context_tokens: int = 128000


def build_model_provider() -> ModelProvider:
    """Build the appropriate ModelProvider based on environment configuration."""
    provider = os.getenv("RAGBOT_LLM_PROVIDER", "openai").lower()
    if provider == "ollama":
        from .ollama import OllamaAdapter
        return OllamaAdapter()
    from .client import OpenAIClient
    return OpenAIClient()
