from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ModelCapabilities:
    structured_output: bool = True
    json_schema: bool = False
    streaming: bool = True
    tools: bool = False
    web_search: bool = False
    vision: bool = False
    reasoning: bool = False
    batch: bool = False
    max_context_tokens: int = 128_000
    max_output_tokens: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "structured_output": self.structured_output,
            "json_schema": self.json_schema,
            "streaming": self.streaming,
            "tools": self.tools,
            "web_search": self.web_search,
            "vision": self.vision,
            "reasoning": self.reasoning,
            "batch": self.batch,
            "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
        }


@dataclass(frozen=True)
class ModelEndpoint:
    provider_id: str
    model_id: str
    base_url: str = ""
    api_key: str = ""
    api_version: str = ""
    organization: str = ""
    project: str = ""
    region: str = ""
    timeout_seconds: float = 60.0
    max_attempts: int = 3
    concurrency: int = 16
    reasoning_effort: Optional[str] = None
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    @property
    def identity(self) -> str:
        return f"{self.provider_id}:{self.model_id}"

    def public_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "api_version": self.api_version,
            "region": self.region,
            "timeout_seconds": self.timeout_seconds,
            "max_attempts": self.max_attempts,
            "concurrency": self.concurrency,
            "reasoning_effort": self.reasoning_effort,
            "capabilities": self.capabilities.as_dict(),
        }


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class UsageMixin:
    """Concurrency-safe provider usage handoff to the routing layer."""

    def __init__(self) -> None:
        self._usage_var: ContextVar[Optional[ModelUsage]] = ContextVar(
            f"ragbot_model_usage_{id(self)}",
            default=None,
        )

    def _record_usage(self, usage: Optional[ModelUsage]) -> None:
        self._usage_var.set(usage)

    def consume_usage(self) -> Optional[ModelUsage]:
        usage = self._usage_var.get()
        self._usage_var.set(None)
        return usage
