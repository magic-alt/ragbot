from __future__ import annotations

import inspect
import json
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

from .contracts import ModelCapabilities, ModelUsage
from .provider import ModelProvider, build_model_provider, endpoint_from_env

ModelTier = Literal["fast", "strong"]

TASK_TIER_MAP: Dict[str, ModelTier] = {
    "route": "fast",
    "retrieve": "fast",
    "sql_query": "fast",
    "code_search": "fast",
    "web_search": "fast",
    "synthesize": "strong",
    "verify": "fast",
    "finalize": "fast",
    "open_file": "fast",
    "apply_patch": "strong",
    "explain_error": "strong",
}
_LEGACY_TIER_COST_PER_MILLION = {"fast": 0.50, "strong": 10.0}


class ModelCapabilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float = 0.0
    output_per_million: float = 0.0
    cached_input_per_million: Optional[float] = None


@dataclass
class CostRecord:
    task: str
    tier: str
    provider: str = "legacy"
    model: str = "legacy"
    request_id: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    usage_source: str = "estimated"
    timestamp: float = field(default_factory=time.time)

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens

    @property
    def completion_tokens(self) -> int:
        return self.output_tokens


class CostTracker:
    def __init__(self, max_history: int = 10000, pricing: Optional[Dict[str, ModelPrice]] = None) -> None:
        self._lock = threading.Lock()
        self._records: List[CostRecord] = []
        self._max_history = max_history
        self._pricing = pricing if pricing is not None else _pricing_from_env()

    def record(
        self,
        task: str,
        tier: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        *,
        provider: Optional[ModelProvider] = None,
        usage: Optional[ModelUsage] = None,
        usage_source: str = "provider",
        request_id: Optional[str] = None,
    ) -> CostRecord:
        if usage is None:
            usage = ModelUsage(input_tokens=int(prompt_tokens), output_tokens=int(completion_tokens))
        if provider is None:
            rate = _LEGACY_TIER_COST_PER_MILLION.get(tier, 5.0)
            estimated = usage.total_tokens / 1_000_000 * rate
            provider_id = "legacy"
            model_id = tier
            usage_source = "estimated"
        else:
            provider_id = getattr(provider, "provider_id", "unknown")
            model_id = getattr(provider, "model_id", "unknown")
            identity = f"{provider_id}:{model_id}"
            price = self._pricing.get(identity) or self._pricing.get(model_id) or ModelPrice()
            cached_rate = (
                price.cached_input_per_million
                if price.cached_input_per_million is not None
                else price.input_per_million
            )
            uncached_input = max(0, usage.input_tokens - usage.cached_input_tokens)
            estimated = (
                uncached_input * price.input_per_million
                + usage.cached_input_tokens * cached_rate
                + usage.output_tokens * price.output_per_million
            ) / 1_000_000
        rec = CostRecord(
            task=task,
            tier=tier,
            provider=provider_id,
            model=model_id,
            request_id=request_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=estimated,
            usage_source=usage_source,
        )
        with self._lock:
            self._records.append(rec)
            if len(self._records) > self._max_history:
                self._records = self._records[-self._max_history:]
        return rec

    def records_for_request(self, request_id: str) -> List[CostRecord]:
        with self._lock:
            return [item for item in self._records if item.request_id == request_id]

    def summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            records = list(self._records)
        if last_n:
            records = records[-last_n:]
        by_tier: Dict[str, Dict[str, Any]] = {}
        by_model: Dict[str, Dict[str, Any]] = {}
        by_task: Dict[str, Dict[str, Any]] = {}
        for record in records:
            identity = f"{record.provider}:{record.model}"
            for bucket, key in (
                (by_tier, record.tier),
                (by_model, identity),
                (by_task, record.task),
            ):
                value = bucket.setdefault(key, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
                value["calls"] += 1
                value["tokens"] += record.total_tokens
                value["cost_usd"] += record.estimated_cost_usd
        for bucket in (by_tier, by_model, by_task):
            for value in bucket.values():
                value["cost_usd"] = round(float(value["cost_usd"]), 6)
        return {
            "total_calls": len(records),
            "total_tokens": sum(item.total_tokens for item in records),
            "total_cost_usd": round(sum(item.estimated_cost_usd for item in records), 6),
            "by_tier": by_tier,
            "by_model": by_model,
            "by_task": by_task,
        }

    def reset(self) -> None:
        with self._lock:
            self._records.clear()


class _TaskBoundModel:
    def __init__(self, router: "ModelRouter", task: str) -> None:
        self.router = router
        self.task = task

    @property
    def enabled(self) -> bool:
        return self.router.enabled

    async def chat_json(self, *args, **kwargs):
        return await self.router.chat_json(*args, task=self.task, **kwargs)

    async def stream_text(self, *args, **kwargs):
        async for item in self.router.stream_text(*args, task=self.task, **kwargs):
            yield item

    async def web_search(self, *args, **kwargs):
        return await self.router.web_search(*args, task=self.task, **kwargs)


class ModelRouter:
    """Task-aware provider router with capability validation and fallback."""

    def __init__(
        self,
        fast_provider: Optional[ModelProvider],
        strong_provider: Optional[ModelProvider] = None,
        routing_enabled: bool = False,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self.fast = fast_provider
        self.strong = strong_provider or fast_provider
        self.routing_enabled = routing_enabled
        self.cost_tracker = cost_tracker or CostTracker()
        self._task_context: ContextVar[str] = ContextVar(
            f"ragbot_model_task_{id(self)}", default="default"
        )
        self._request_context: ContextVar[Optional[str]] = ContextVar(
            f"ragbot_model_request_{id(self)}", default=None
        )

    @property
    def provider_id(self) -> str:
        return "router"

    @property
    def model_id(self) -> str:
        return f"{getattr(self.fast, 'model_id', 'fast')}|{getattr(self.strong, 'model_id', 'strong')}"

    @property
    def capabilities(self) -> ModelCapabilities:
        providers = [item for item in (self.fast, self.strong) if item is not None]
        if not providers:
            return ModelCapabilities(structured_output=False, streaming=False)
        return ModelCapabilities(
            structured_output=any(item.capabilities.structured_output for item in providers),
            json_schema=any(item.capabilities.json_schema for item in providers),
            streaming=any(item.capabilities.streaming for item in providers),
            tools=any(item.capabilities.tools for item in providers),
            web_search=any(item.capabilities.web_search for item in providers),
            vision=any(item.capabilities.vision for item in providers),
            reasoning=any(item.capabilities.reasoning for item in providers),
            batch=any(item.capabilities.batch for item in providers),
            max_context_tokens=max(item.capabilities.max_context_tokens for item in providers),
        )

    @property
    def enabled(self) -> bool:
        return any(
            bool(getattr(item, "enabled", False))
            for item in (self.fast, self.strong)
            if item is not None
        )

    @contextmanager
    def task_scope(self, task: str, request_id: Optional[str] = None):
        task_token = self._task_context.set(task)
        request_token = None
        if request_id is not None:
            request_token = self._request_context.set(request_id)
        try:
            yield
        finally:
            if request_token is not None:
                self._request_context.reset(request_token)
            self._task_context.reset(task_token)

    def for_task(self, task: str) -> _TaskBoundModel:
        return _TaskBoundModel(self, task)

    def _task(self, task: str) -> str:
        return self._task_context.get() if task == "default" else task

    def get_tier(self, task: str = "default") -> ModelTier:
        task = self._task(task)
        if not self.routing_enabled:
            return "fast"
        return TASK_TIER_MAP.get(task, "fast")

    def get_provider(
        self,
        task: str = "default",
        *,
        capability: Optional[str] = None,
    ) -> ModelProvider:
        task = self._task(task)
        tier = self.get_tier(task)
        preferred = self.strong if tier == "strong" else self.fast
        alternate = self.fast if preferred is self.strong else self.strong

        if capability is None:
            if preferred is not None:
                return preferred
            if alternate is not None:
                return alternate
            raise ModelCapabilityError(f"No model provider is configured for task={task!r}")

        seen: set[int] = set()
        for provider in (preferred, alternate):
            if provider is None or id(provider) in seen:
                continue
            seen.add(id(provider))
            if not getattr(provider, "enabled", False):
                continue
            capabilities = getattr(provider, "capabilities", None)
            if not capabilities or not bool(getattr(capabilities, capability, False)):
                continue
            return provider
        raise ModelCapabilityError(
            f"No enabled model provider satisfies task={task!r} capability={capability}"
        )

    async def chat_json(
        self,
        system: str,
        user: str,
        schema: Dict[str, Any],
        task: str = "default",
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        task = self._task(task)
        provider = self.get_provider(task, capability="structured_output")
        result = await provider.chat_json(
            system=system,
            user=user,
            schema=schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        self._record(provider, task, system, user, result)
        return result

    async def stream_text(
        self,
        system: str,
        user: str,
        task: str = "default",
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        task = self._task(task)
        provider = self.get_provider(task, capability="streaming")
        output: List[str] = []
        async for chunk in provider.stream_text(
            system=system,
            user=user,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ):
            output.append(chunk)
            yield chunk
        self._record(provider, task, system, user, "".join(output))

    async def web_search(
        self,
        query: str,
        allowed_domains: Optional[List[str]] = None,
        recency_days: Optional[int] = None,
        task: str = "default",
    ) -> List[Dict[str, Any]]:
        task = self._task("web_search" if task == "default" else task)
        provider = self.get_provider(task, capability="web_search")
        result = await provider.web_search(query, allowed_domains, recency_days)
        self._record(provider, task, "", query, result)
        return result

    def _record(
        self,
        provider: ModelProvider,
        task: str,
        system: str,
        user: str,
        result: Any,
    ) -> None:
        consume = getattr(provider, "consume_usage", None)
        usage = consume() if callable(consume) else None
        source = "provider"
        if usage is None:
            source = "estimated"
            usage = ModelUsage(
                input_tokens=(len(system) + len(user)) // 4,
                output_tokens=len(str(result)) // 4,
            )
        self.cost_tracker.record(
            task,
            self.get_tier(task),
            provider=provider,
            usage=usage,
            usage_source=source,
            request_id=self._request_context.get(),
        )

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "routing_enabled": self.routing_enabled,
            "fast": _provider_diagnostics(self.fast),
            "strong": _provider_diagnostics(self.strong),
        }

    async def aclose(self) -> None:
        seen: set[int] = set()
        for provider in (self.fast, self.strong):
            if provider is None or id(provider) in seen:
                continue
            seen.add(id(provider))
            close = getattr(provider, "aclose", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
            else:
                sync_close = getattr(provider, "close", None)
                if callable(sync_close):
                    sync_close()


def model_for_task(model: Any, task: str) -> Any:
    binder = getattr(model, "for_task", None)
    return binder(task) if callable(binder) else model


def model_task_scope(model: Any, task: str, request_id: Optional[str] = None):
    scope = getattr(model, "task_scope", None)
    if not callable(scope):
        return nullcontext()
    try:
        return scope(task, request_id=request_id)
    except TypeError:
        return scope(task)


def build_model_router() -> ModelRouter:
    routing_enabled = os.getenv("RAGBOT_MODEL_ROUTING", "false").lower() in {
        "true",
        "1",
        "yes",
        "on",
    }
    if routing_enabled:
        fast = build_model_provider(endpoint_from_env("FAST"))
        strong = build_model_provider(endpoint_from_env("STRONG"))
    else:
        fast = build_model_provider(endpoint_from_env())
        strong = fast
    return ModelRouter(
        fast_provider=fast,
        strong_provider=strong,
        routing_enabled=routing_enabled,
    )


def _provider_diagnostics(provider: Optional[ModelProvider]) -> Dict[str, Any]:
    if provider is None:
        return {"provider": None, "model": None, "enabled": False, "capabilities": {}}
    endpoint = getattr(provider, "endpoint", None)
    public = (
        endpoint.public_dict()
        if endpoint is not None and hasattr(endpoint, "public_dict")
        else {}
    )
    capabilities = getattr(provider, "capabilities", None)
    return {
        "provider": getattr(provider, "provider_id", type(provider).__name__),
        "model": getattr(provider, "model_id", "unknown"),
        "enabled": bool(getattr(provider, "enabled", False)),
        "capabilities": capabilities.as_dict() if capabilities is not None else {},
        "endpoint": public,
    }


def _pricing_from_env() -> Dict[str, ModelPrice]:
    raw = os.getenv("RAGBOT_MODEL_PRICING_JSON", "").strip()
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("RAGBOT_MODEL_PRICING_JSON must be valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("RAGBOT_MODEL_PRICING_JSON must be an object")
    output: Dict[str, ModelPrice] = {}
    for identity, value in decoded.items():
        if not isinstance(value, dict):
            raise ValueError(f"Model pricing for {identity!r} must be an object")
        output[str(identity)] = ModelPrice(
            input_per_million=float(value.get("input_per_million", 0.0)),
            output_per_million=float(value.get("output_per_million", 0.0)),
            cached_input_per_million=(
                float(value["cached_input_per_million"])
                if value.get("cached_input_per_million") is not None
                else None
            ),
        )
    return output
