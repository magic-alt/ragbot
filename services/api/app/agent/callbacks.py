"""Agent event callback system for real-time SSE streaming."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class AgentEvent:
    """A single event emitted during agent execution."""

    event_type: str  # tool_call, tool_result, evidence, token, final
    data: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class EventCallback(Protocol):
    """Protocol for agent event callbacks."""

    def emit(self, event: AgentEvent) -> None: ...

    def close(self) -> None: ...


class NullCallback:
    """No-op callback for non-streaming mode. Zero overhead."""

    def emit(self, event: AgentEvent) -> None:
        pass

    def close(self) -> None:
        pass


class QueueCallback:
    """Thread-safe callback that writes events to a queue.

    Used to bridge the sync agent thread with the async SSE endpoint.
    The agent thread calls emit() to push events; the async SSE generator
    reads from the queue via get().
    """

    _SENTINEL = object()

    def __init__(self, maxsize: int = 256) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._closed = threading.Event()

    def emit(self, event: AgentEvent) -> None:
        if self._closed.is_set():
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Drop oldest event to make room
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                pass

    def close(self) -> None:
        """Signal that no more events will be emitted."""
        self._closed.set()
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            pass

    def get(self, timeout: float = 1.0) -> Optional[AgentEvent]:
        """Get the next event, or None on timeout. Returns None after close."""
        try:
            item = self._queue.get(timeout=timeout)
            if item is self._SENTINEL:
                return None
            return item
        except queue.Empty:
            if self._closed.is_set():
                return None
            raise  # Let caller decide what to do on timeout

    @property
    def closed(self) -> bool:
        return self._closed.is_set() and self._queue.empty()
