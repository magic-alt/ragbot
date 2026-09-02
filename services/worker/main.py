"""Durable PostgreSQL-backed ingestion worker.

Run with::

    python -m services.worker.main

The API persists jobs as ``pending``. Workers atomically claim jobs using the
repository lease contract, heartbeat while connectors/embedders run, and allow
another worker to recover work after a process crash or node restart.
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone

from services.api.app.factory import build_services_from_env
from services.worker.pipeline import run_ingest_pipeline

logger = logging.getLogger(__name__)
_STOP = threading.Event()


def main() -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _install_signal_handlers()

    poll_seconds = _positive_float("RAGBOT_WORKER_POLL_SECONDS", 1.0)
    lease_seconds = _positive_int("RAGBOT_WORKER_LEASE_SECONDS", 120)
    max_attempts = _positive_int("RAGBOT_WORKER_MAX_ATTEMPTS", 3)
    worker_id = os.getenv("RAGBOT_WORKER_ID") or f"{socket.gethostname()}:{os.getpid()}"

    services = build_services_from_env()
    claim = getattr(services.repo, "claim_next_job", None)
    heartbeat = getattr(services.repo, "heartbeat_job", None)
    release = getattr(services.repo, "release_job_lease", None)
    if not callable(claim) or not callable(heartbeat) or not callable(release):
        raise RuntimeError("Configured repository does not implement durable ingestion leases")

    logger.info(
        "Ingestion worker started: worker_id=%s lease=%ss max_attempts=%d",
        worker_id,
        lease_seconds,
        max_attempts,
    )
    try:
        while not _STOP.is_set():
            job = claim(worker_id, lease_seconds=lease_seconds, max_attempts=max_attempts)
            if job is None:
                _STOP.wait(poll_seconds)
                continue
            _execute_claimed_job(
                job,
                services,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
            )
    finally:
        _close_services(services)
        logger.info("Ingestion worker stopped: worker_id=%s", worker_id)
    return 0


def _execute_claimed_job(job, services, *, worker_id: str, lease_seconds: int) -> None:
    current_source = services.repo.get_source(job.source_id)
    if current_source is None or current_source.status != "active" or current_source.tenant_id != job.tenant_id:
        reason = "Source unavailable, inactive, or tenant-mismatched at execution time"
        services.repo.update_job(
            job.job_id,
            status="failed",
            error=reason,
            completed_at=datetime.now(timezone.utc).isoformat(),
            lease_owner=None,
            lease_expires_at=None,
        )
        logger.error("Rejecting claimed job %s: %s", job.job_id, reason)
        return

    # Connector configuration is part of the durable job contract. A Source may
    # be edited while a job waits in the queue; executing the queued job against
    # the mutable current config would make retries/non-immediate execution
    # nondeterministic. Keep current Source metadata/ACL state, but execute the
    # connector type/config snapshot captured when this job was submitted.
    source = replace(
        current_source,
        source_type=job.source_type,
        config=dict(job.source_config or {}),
    )

    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(services.repo, job.job_id, worker_id, lease_seconds, heartbeat_stop),
        name=f"ragbot-heartbeat-{job.job_id[:8]}",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        result = run_ingest_pipeline(
            source,
            services.repo,
            services.qdrant,
            job.job_id,
            services.embedder,
            True,
        )
        logger.info(
            "Ingestion job finished: job=%s status=%s attempts=%d",
            result.job_id,
            result.status,
            result.attempts,
        )
    except Exception as exc:  # pipeline normally persists failures itself
        logger.exception("Unexpected worker failure for job=%s", job.job_id)
        services.repo.update_job(
            job.job_id,
            status="failed",
            error=f"Worker execution failure: {exc}",
            completed_at=datetime.now(timezone.utc).isoformat(),
            lease_owner=None,
            lease_expires_at=None,
        )
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=max(1.0, lease_seconds / 3))
        services.repo.release_job_lease(job.job_id, worker_id)


def _heartbeat_loop(repo, job_id: str, worker_id: str, lease_seconds: int, stop: threading.Event) -> None:
    interval = max(1.0, lease_seconds / 3)
    while not stop.wait(interval):
        try:
            if not repo.heartbeat_job(job_id, worker_id, lease_seconds=lease_seconds):
                logger.warning("Lost lease while heartbeating job=%s worker=%s", job_id, worker_id)
                return
        except Exception:
            logger.exception("Heartbeat failed for job=%s worker=%s", job_id, worker_id)


def _install_signal_handlers() -> None:
    def _stop(signum, frame) -> None:  # pragma: no cover - OS signal path
        logger.info("Received signal %s; stopping after current operation", signum)
        _STOP.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, _stop)
        except (ValueError, OSError):
            pass


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


def _close_services(services) -> None:
    closed: set[int] = set()
    for resource in (services.sql_engine, services.repo, services.qdrant):
        if id(resource) in closed:
            continue
        close = getattr(resource, "close", None)
        if callable(close):
            close()
        closed.add(id(resource))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
