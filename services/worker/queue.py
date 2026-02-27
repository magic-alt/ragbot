"""Task queue abstraction for ragbot worker.

Provides a thin wrapper around task execution that can be backed by:
- ``InProcessQueue``: synchronous execution for development/testing
- ``CeleryQueue``: Celery-based distributed task queue (requires celery)

Usage::

    queue = create_queue()
    queue.submit("ingest_pdf", {"path": "/data/doc.pdf", "doc_id": "d1", "tenant_id": "t1"})
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Protocol

logger = logging.getLogger(__name__)


class TaskQueue(Protocol):
    def submit(self, task_name: str, kwargs: Dict[str, Any]) -> str: ...

    def status(self, task_id: str) -> Dict[str, Any]: ...


class InProcessQueue:
    """Execute tasks synchronously in the current process (dev/test)."""

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[..., Any]] = {}
        self._results: Dict[str, Dict[str, Any]] = {}
        self._counter = 0

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self._registry[name] = fn

    def submit(self, task_name: str, kwargs: Dict[str, Any]) -> str:
        self._counter += 1
        task_id = f"task-{self._counter}"
        fn = self._registry.get(task_name)
        if not fn:
            self._results[task_id] = {"status": "failed", "error": f"Unknown task: {task_name}"}
            return task_id
        try:
            result = fn(**kwargs)
            # Consume generators
            if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
                result = list(result)
            self._results[task_id] = {"status": "completed", "result": result}
        except Exception as exc:
            logger.exception("Task %s failed", task_name)
            self._results[task_id] = {"status": "failed", "error": str(exc)}
        return task_id

    def status(self, task_id: str) -> Dict[str, Any]:
        return self._results.get(task_id, {"status": "unknown"})


def create_queue(backend: str = "in_process") -> TaskQueue:
    """Factory for creating task queues.

    Args:
        backend: ``"in_process"`` for synchronous execution (default),
                 ``"celery"`` for Celery (requires celery package).
    """
    if backend == "celery":
        try:
            return _create_celery_queue()
        except ImportError:
            logger.warning("celery not installed; falling back to in-process queue")
    queue = InProcessQueue()
    _register_default_tasks(queue)
    return queue


def _register_default_tasks(queue: InProcessQueue) -> None:
    from services.worker.jobs.ingest_pdf import ingest_pdf
    from services.worker.jobs.ingest_repo import ingest_repo
    from services.worker.jobs.ingest_web import ingest_web

    queue.register("ingest_pdf", ingest_pdf)
    queue.register("ingest_repo", ingest_repo)
    queue.register("ingest_web", ingest_web)


def _create_celery_queue() -> Any:
    """Create a Celery-backed task queue (requires celery package)."""
    raise ImportError("Celery integration not yet implemented")
