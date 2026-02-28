from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default timeouts per tool (seconds)
DEFAULT_TIMEOUTS: Dict[str, float] = {
    "retrieve": 10.0,
    "sql_query": 5.0,
    "code_search": 8.0,
    "web_search": 15.0,
    "synthesize": 20.0,
    "verify": 15.0,
}

# Default retry config per tool
DEFAULT_RETRY: Dict[str, "RetryConfig"] = {}


@dataclass
class RetryConfig:
    max_retries: int = 2
    base_delay: float = 0.5
    max_delay: float = 5.0
    retryable_exceptions: tuple = (ConnectionError, TimeoutError, OSError)


@dataclass
class CircuitBreaker:
    """Simple circuit breaker: opens after consecutive failures, resets after cooldown."""

    failure_threshold: int = 3
    cooldown_seconds: float = 30.0
    _failures: int = field(default=0, init=False, repr=False)
    _opened_at: Optional[float] = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if time.monotonic() - self._opened_at >= self.cooldown_seconds:
                self._reset()
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.monotonic()
                logger.warning(
                    "Circuit breaker opened after %d failures, cooldown %.1fs",
                    self._failures,
                    self.cooldown_seconds,
                )

    def _reset(self) -> None:
        self._failures = 0
        self._opened_at = None


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open."""


class ToolTimeoutError(Exception):
    """Raised when a tool call exceeds its timeout."""


# Global circuit breakers per tool name
_breakers: Dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def _get_breaker(tool_name: str) -> CircuitBreaker:
    with _breakers_lock:
        if tool_name not in _breakers:
            _breakers[tool_name] = CircuitBreaker()
        return _breakers[tool_name]


async def safe_tool_call(tool_name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Unified async entry point: circuit breaker -> timeout -> execute.

    If `fn` returns a coroutine, it is awaited directly with asyncio.wait_for.
    If `fn` is a regular sync function, it is run in a thread via asyncio.to_thread.
    """
    breaker = _get_breaker(tool_name)
    if breaker.is_open:
        raise CircuitOpenError(f"Circuit breaker open for {tool_name}")

    timeout = DEFAULT_TIMEOUTS.get(tool_name, 15.0)
    retry_config = DEFAULT_RETRY.get(tool_name)

    try:
        if retry_config:
            result = await _async_with_retry(
                lambda: _async_with_timeout(fn, timeout, *args, **kwargs),
                retry_config,
            )
        else:
            result = await _async_with_timeout(fn, timeout, *args, **kwargs)
        breaker.record_success()
        return result
    except Exception:
        breaker.record_failure()
        raise


async def _async_with_timeout(fn: Callable[..., Any], timeout: float, *args: Any, **kwargs: Any) -> Any:
    """Execute fn with a timeout. Handles both sync and async callables."""
    result = fn(*args, **kwargs)
    if asyncio.iscoroutine(result):
        try:
            return await asyncio.wait_for(result, timeout=timeout)
        except asyncio.TimeoutError:
            raise ToolTimeoutError(f"Tool call timed out after {timeout:.1f}s")
    else:
        # Sync function — already returned
        return result


async def _async_with_retry(
    fn: Callable[..., Coroutine],
    config: RetryConfig,
) -> Any:
    """Execute an async callable with exponential backoff retry."""
    last_exc: Optional[Exception] = None
    delay = config.base_delay
    for attempt in range(config.max_retries + 1):
        try:
            return await fn()
        except config.retryable_exceptions as exc:
            last_exc = exc
            if attempt < config.max_retries:
                logger.info(
                    "Retry %d/%d after %s: %s",
                    attempt + 1,
                    config.max_retries,
                    type(exc).__name__,
                    exc,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, config.max_delay)
            else:
                raise
        except Exception:
            raise
    raise last_exc  # type: ignore[misc]
