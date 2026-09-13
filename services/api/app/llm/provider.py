from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable

from .contracts import ModelCapabilities, ModelEndpoint, ModelUsage


@runtime_checkable
class ModelProvider(Protocol):
    @property
    def enabled(self) -> bool: ...

    @property
    def provider_id(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    @property
    def capabilities(self) -> ModelCapabilities: ...

    async def chat_json(
        self,
        system: str,
        user: str,
        schema: Dict[str, Any],
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]: ...

    async def stream_text(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]: ...

    async def web_search(
        self,
        query: str,
        allowed_domains: Optional[List[str]] = None,
        recency_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]: ...

    def consume_usage(self) -> Optional[ModelUsage]: ...


def endpoint_from_env(tier: Optional[str] = None) -> ModelEndpoint:
    """Resolve one immutable model endpoint without mutating process environment."""
    normalized_tier = str(tier or "").strip().upper()
    prefix = f"RAGBOT_MODEL_{normalized_tier}_" if normalized_tier else ""
    default_provider = os.getenv("RAGBOT_LLM_PROVIDER", "openai").strip().lower() or "openai"
    provider = os.getenv(f"{prefix}PROVIDER", default_provider).strip().lower() or default_provider

    model_override = os.getenv(f"RAGBOT_MODEL_{normalized_tier}", "").strip() if normalized_tier else ""
    model = os.getenv(f"{prefix}MODEL", "").strip() or model_override
    timeout = float(os.getenv(f"{prefix}TIMEOUT_SECONDS", os.getenv("RAGBOT_MODEL_TIMEOUT_SECONDS", "60")))
    max_attempts = int(os.getenv(f"{prefix}MAX_ATTEMPTS", os.getenv("RAGBOT_MODEL_MAX_ATTEMPTS", "3")))
    concurrency = int(os.getenv(f"{prefix}CONCURRENCY", os.getenv("RAGBOT_MODEL_CONCURRENCY", "16")))

    if provider in {"openai", "openai-compatible"}:
        return ModelEndpoint(
            provider_id=provider,
            model_id=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv(f"{prefix}BASE_URL", "").strip() or os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/"),
            api_key=os.getenv(f"{prefix}API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", ""),
            organization=os.getenv("OPENAI_ORGANIZATION", ""),
            project=os.getenv("OPENAI_PROJECT", ""),
            timeout_seconds=timeout,
            max_attempts=max_attempts,
            concurrency=concurrency,
            capabilities=ModelCapabilities(structured_output=True, json_schema=True, streaming=True, tools=True, web_search=provider == "openai", vision=True, reasoning=True, batch=True),
        )
    if provider == "ollama":
        reasoning_effort = os.getenv(f"{prefix}REASONING_EFFORT", "").strip() or os.getenv("OLLAMA_REASONING_EFFORT", "").strip() or None
        return ModelEndpoint(
            provider_id="ollama",
            model_id=model or os.getenv("OLLAMA_MODEL", "llama3"),
            base_url=os.getenv(f"{prefix}BASE_URL", "").strip() or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            timeout_seconds=float(os.getenv(f"{prefix}TIMEOUT_SECONDS", os.getenv("OLLAMA_TIMEOUT_SECONDS", str(timeout)))),
            max_attempts=max_attempts,
            concurrency=concurrency,
            reasoning_effort=reasoning_effort,
            capabilities=ModelCapabilities(structured_output=True, json_schema=True, streaming=True, reasoning=True),
        )
    if provider == "anthropic":
        return ModelEndpoint(
            provider_id="anthropic",
            model_id=model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            base_url=os.getenv(f"{prefix}BASE_URL", "").strip() or os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/"),
            api_key=os.getenv(f"{prefix}API_KEY", "").strip() or os.getenv("ANTHROPIC_API_KEY", ""),
            api_version=os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
            timeout_seconds=timeout,
            max_attempts=max_attempts,
            concurrency=concurrency,
            capabilities=ModelCapabilities(structured_output=True, streaming=True, tools=True, vision=True, reasoning=True),
        )
    if provider in {"gemini", "google", "vertex"}:
        vertex = provider == "vertex"
        base_url = os.getenv(f"{prefix}BASE_URL", "").strip() or os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")
        return ModelEndpoint(
            provider_id="vertex" if vertex else "gemini",
            model_id=model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            base_url=base_url,
            api_key=os.getenv(f"{prefix}API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "") or os.getenv("GOOGLE_OAUTH_ACCESS_TOKEN", ""),
            project=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
            region=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            timeout_seconds=timeout,
            max_attempts=max_attempts,
            concurrency=concurrency,
            capabilities=ModelCapabilities(structured_output=True, json_schema=True, streaming=True, tools=True, vision=True, reasoning=True),
        )
    if provider in {"azure", "azure-openai"}:
        return ModelEndpoint(
            provider_id="azure-openai",
            model_id=model or os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
            base_url=os.getenv(f"{prefix}BASE_URL", "").strip() or os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/"),
            api_key=os.getenv(f"{prefix}API_KEY", "").strip() or os.getenv("AZURE_OPENAI_API_KEY", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            timeout_seconds=timeout,
            max_attempts=max_attempts,
            concurrency=concurrency,
            capabilities=ModelCapabilities(structured_output=True, json_schema=True, streaming=True, tools=True, vision=True, reasoning=True),
        )
    if provider == "bedrock":
        return ModelEndpoint(
            provider_id="bedrock",
            model_id=model or os.getenv("BEDROCK_MODEL", ""),
            region=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
            timeout_seconds=timeout,
            max_attempts=max_attempts,
            concurrency=concurrency,
            capabilities=ModelCapabilities(structured_output=True, tools=True, vision=True, reasoning=True, streaming=False),
        )
    raise ValueError(f"Unsupported RAGBOT model provider: {provider}")


def build_model_provider(endpoint: Optional[ModelEndpoint] = None) -> ModelProvider:
    endpoint = endpoint or endpoint_from_env()
    from ..runtime_registry import runtime_component_registry
    return runtime_component_registry().build(
        "llm",
        endpoint.provider_id,
        {"endpoint": endpoint},
    )
