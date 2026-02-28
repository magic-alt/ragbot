"""Model router: task-based routing between fast and strong models.

Routes agent tasks to the appropriate model tier:
- fast: route planning, simple synthesis, verification (low complexity)
- strong: complex synthesis, multi-hop reasoning, code generation (high complexity)

Configuration via environment:
    RAGBOT_MODEL_FAST: model name for fast tier (default: gpt-4o-mini)
    RAGBOT_MODEL_STRONG: model name for strong tier (default: gpt-4o)
    RAGBOT_MODEL_ROUTING: enable routing (default: false → use single model)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

ModelTier = Literal["fast", "strong"]

# Task → tier mapping
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


@dataclass
class CostRecord:
    """Token usage and cost for a single LLM call."""

    task: str
    tier: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)


# Approximate per-million-token pricing (input + output blended)
TIER_COST_PER_MILLION: Dict[str, float] = {
    "fast": 0.50,   # gpt-4o-mini class
    "strong": 10.0,  # gpt-4o class
}


class CostTracker:
    """Thread-safe tracker for LLM token usage and cost estimates."""

    def __init__(self, max_history: int = 10000) -> None:
        self._lock = threading.Lock()
        self._records: List[CostRecord] = []
        self._max_history = max_history

    def record(self, task: str, tier: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> CostRecord:
        """Record token usage for a task."""
        total = prompt_tokens + completion_tokens
        cost_per_m = TIER_COST_PER_MILLION.get(tier, 5.0)
        estimated = (total / 1_000_000) * cost_per_m

        rec = CostRecord(
            task=task,
            tier=tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            estimated_cost_usd=estimated,
        )
        with self._lock:
            self._records.append(rec)
            if len(self._records) > self._max_history:
                self._records = self._records[-self._max_history:]
        return rec

    def summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Aggregate cost summary."""
        with self._lock:
            records = list(self._records)
        if last_n:
            records = records[-last_n:]

        if not records:
            return {"total_tokens": 0, "total_cost_usd": 0.0, "by_tier": {}, "by_task": {}}

        total_tokens = sum(r.total_tokens for r in records)
        total_cost = sum(r.estimated_cost_usd for r in records)

        by_tier: Dict[str, Dict[str, Any]] = {}
        by_task: Dict[str, Dict[str, Any]] = {}

        for r in records:
            if r.tier not in by_tier:
                by_tier[r.tier] = {"calls": 0, "tokens": 0, "cost_usd": 0.0}
            by_tier[r.tier]["calls"] += 1
            by_tier[r.tier]["tokens"] += r.total_tokens
            by_tier[r.tier]["cost_usd"] += r.estimated_cost_usd

            if r.task not in by_task:
                by_task[r.task] = {"calls": 0, "tokens": 0, "cost_usd": 0.0}
            by_task[r.task]["calls"] += 1
            by_task[r.task]["tokens"] += r.total_tokens
            by_task[r.task]["cost_usd"] += r.estimated_cost_usd

        # Round floats
        for v in by_tier.values():
            v["cost_usd"] = round(v["cost_usd"], 6)
        for v in by_task.values():
            v["cost_usd"] = round(v["cost_usd"], 6)

        return {
            "total_calls": len(records),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "by_tier": by_tier,
            "by_task": by_task,
        }

    def reset(self) -> None:
        with self._lock:
            self._records.clear()


class ModelRouter:
    """Routes tasks to appropriate model tier with fallback.

    When routing is disabled, all tasks use the default provider.
    When enabled, tasks are mapped to fast/strong tiers.
    """

    def __init__(
        self,
        fast_provider=None,
        strong_provider=None,
        routing_enabled: bool = False,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self.fast = fast_provider
        self.strong = strong_provider or fast_provider
        self.routing_enabled = routing_enabled
        self.cost_tracker = cost_tracker or CostTracker()

    def get_provider(self, task: str = "default"):
        """Get the appropriate provider for a task."""
        if not self.routing_enabled or not self.strong:
            return self.fast

        tier = TASK_TIER_MAP.get(task, "fast")
        if tier == "strong" and self.strong and self.strong.enabled:
            return self.strong
        return self.fast

    def get_tier(self, task: str = "default") -> ModelTier:
        """Get the tier for a task."""
        if not self.routing_enabled:
            return "fast"
        return TASK_TIER_MAP.get(task, "fast")

    async def chat_json(
        self,
        system: str,
        user: str,
        schema: Dict[str, Any],
        task: str = "default",
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Route a chat_json call to the appropriate provider."""
        provider = self.get_provider(task)
        tier = self.get_tier(task)

        result = await provider.chat_json(
            system=system,
            user=user,
            schema=schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        # Estimate tokens (rough: 4 chars = 1 token)
        prompt_tokens = (len(system) + len(user)) // 4
        completion_tokens = len(str(result)) // 4
        self.cost_tracker.record(task, tier, prompt_tokens, completion_tokens)

        return result

    async def stream_text(
        self,
        system: str,
        user: str,
        task: str = "default",
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Route a stream_text call to the appropriate provider."""
        provider = self.get_provider(task)
        tier = self.get_tier(task)

        total_text = []
        async for chunk in provider.stream_text(
            system=system,
            user=user,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ):
            total_text.append(chunk)
            yield chunk

        prompt_tokens = (len(system) + len(user)) // 4
        completion_tokens = len("".join(total_text)) // 4
        self.cost_tracker.record(task, tier, prompt_tokens, completion_tokens)

    @property
    def enabled(self) -> bool:
        return self.fast is not None and self.fast.enabled

    async def web_search(
        self,
        query: str,
        allowed_domains: Optional[List[str]] = None,
        recency_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Web search always goes through the strong provider (if available)."""
        provider = self.strong if (self.strong and self.strong.enabled) else self.fast
        return await provider.web_search(query, allowed_domains, recency_days)


def build_model_router() -> ModelRouter:
    """Build a model router based on environment configuration."""
    from .provider import build_model_provider

    routing_enabled = os.getenv("RAGBOT_MODEL_ROUTING", "false").lower() in ("true", "1", "yes")

    fast_provider = build_model_provider()

    strong_provider = None
    if routing_enabled:
        strong_model = os.getenv("RAGBOT_MODEL_STRONG")
        if strong_model:
            # Build a separate provider for the strong tier
            original_model = os.getenv("OPENAI_MODEL")
            os.environ["OPENAI_MODEL"] = strong_model
            strong_provider = build_model_provider()
            if original_model:
                os.environ["OPENAI_MODEL"] = original_model
            else:
                os.environ.pop("OPENAI_MODEL", None)

    return ModelRouter(
        fast_provider=fast_provider,
        strong_provider=strong_provider,
        routing_enabled=routing_enabled,
    )
