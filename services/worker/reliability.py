"""Reliability helpers shared by ingestion connectors and durable workers."""
from __future__ import annotations

import logging
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Optional

import requests

logger = logging.getLogger(__name__)

RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_HTTP_STATUS_RE = re.compile(r"\b([1-5][0-9]{2})\b")


@dataclass(frozen=True)
class FailureClassification:
    failure_class: str
    retryable: bool


def provider_request(
    session: Any,
    method: str,
    url: str,
    *,
    max_attempts: Optional[int] = None,
    backoff_base_seconds: Optional[float] = None,
    backoff_max_seconds: Optional[float] = None,
    sleep: Callable[[float], None] = time.sleep,
    random_fn: Callable[[], float] = random.random,
    **kwargs: Any,
):
    """Execute one HTTP request with bounded retry for transient provider failures.

    The helper deliberately retries only transport failures and retryable HTTP
    statuses. Authentication/authorization/not-found errors remain immediate so
    bad credentials or bad Source configuration do not create retry storms.
    ``Retry-After`` is honored when present and capped by the configured maximum.
    """
    attempts = max_attempts or _positive_int("RAGBOT_PROVIDER_MAX_ATTEMPTS", 4)
    base = backoff_base_seconds or _positive_float("RAGBOT_PROVIDER_BACKOFF_BASE_SECONDS", 0.5)
    maximum = backoff_max_seconds or _positive_float("RAGBOT_PROVIDER_BACKOFF_MAX_SECONDS", 30.0)
    method_name = method.strip().lower()
    request = getattr(session, method_name)

    last_transport_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            response = request(url, **kwargs)
        except requests.RequestException as exc:
            last_transport_error = exc
            if attempt >= attempts:
                raise
            delay = _backoff_delay(attempt, base, maximum, random_fn=random_fn)
            logger.warning(
                "Provider transport retry: method=%s url=%s attempt=%d/%d delay=%.3fs error=%s",
                method_name.upper(), url, attempt, attempts, delay, type(exc).__name__,
            )
            sleep(delay)
            continue

        status = int(getattr(response, "status_code", 0) or 0)
        if status not in RETRYABLE_HTTP_STATUSES or attempt >= attempts:
            return response

        retry_after = _retry_after_seconds(getattr(response, "headers", {}) or {})
        delay = min(maximum, retry_after) if retry_after is not None else _backoff_delay(
            attempt, base, maximum, random_fn=random_fn
        )
        logger.warning(
            "Provider HTTP retry: method=%s url=%s status=%d attempt=%d/%d delay=%.3fs",
            method_name.upper(), url, status, attempt, attempts, delay,
        )
        sleep(delay)

    if last_transport_error is not None:  # pragma: no cover - defensive
        raise last_transport_error
    raise RuntimeError("provider_request exhausted without response")  # pragma: no cover


def classify_ingestion_error(exc: BaseException) -> FailureClassification:
    """Classify an ingestion exception for durable job-level retry decisions."""
    if isinstance(exc, requests.HTTPError):
        response = getattr(exc, "response", None)
        status = int(getattr(response, "status_code", 0) or 0)
        if status in RETRYABLE_HTTP_STATUSES:
            return FailureClassification(f"http_{status}", True)
        if status:
            return FailureClassification(f"http_{status}", False)
        return FailureClassification("http_error", True)
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return FailureClassification("provider_transport", True)
    if isinstance(exc, (FileNotFoundError, PermissionError)):
        return FailureClassification(type(exc).__name__.lower(), False)
    if isinstance(exc, ValueError):
        return FailureClassification("validation_error", False)
    return FailureClassification(type(exc).__name__.lower() or "ingestion_error", True)


def classify_persisted_failure(message: Optional[str]) -> FailureClassification:
    """Classify a pipeline failure after it has already been persisted as text.

    The pipeline intentionally sanitizes/serializes errors into the Job record.
    This conservative classifier prevents obvious permanent auth/config failures
    from being replayed repeatedly while leaving unknown runtime failures
    retryable under the bounded durable-attempt budget.
    """
    text = str(message or "").strip()
    lower = text.lower()
    for match in _HTTP_STATUS_RE.finditer(text):
        status = int(match.group(1))
        if status in RETRYABLE_HTTP_STATUSES:
            return FailureClassification(f"http_{status}", True)
        if 400 <= status < 500:
            return FailureClassification(f"http_{status}", False)
    permanent_markers = (
        "permission denied",
        "no such file",
        "not found:",
        "requires non-empty",
        "must not be empty",
        "invalid source",
        "unsupported source_type",
        "credential reference environment variable is not set",
        "api principal requires",
    )
    if any(marker in lower for marker in permanent_markers):
        return FailureClassification("configuration_error", False)
    return FailureClassification("pipeline_failure", True)


def durable_retry_delay(
    attempts: int,
    *,
    base_seconds: Optional[float] = None,
    max_seconds: Optional[float] = None,
) -> float:
    base = base_seconds or _positive_float("RAGBOT_WORKER_RETRY_BASE_SECONDS", 5.0)
    maximum = max_seconds or _positive_float("RAGBOT_WORKER_RETRY_MAX_SECONDS", 300.0)
    exponent = max(0, int(attempts) - 1)
    return min(maximum, base * (2 ** exponent))


def _backoff_delay(
    attempt: int,
    base: float,
    maximum: float,
    *,
    random_fn: Callable[[], float],
) -> float:
    raw = min(maximum, base * (2 ** max(0, attempt - 1)))
    # Equal jitter: avoids synchronized workers hammering a recovering provider.
    return min(maximum, raw * (0.5 + 0.5 * max(0.0, min(1.0, random_fn()))))


def _retry_after_seconds(headers: Any) -> Optional[float]:
    value = None
    try:
        value = headers.get("Retry-After")
    except AttributeError:
        return None
    if value in (None, ""):
        return None
    raw = str(value).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value
