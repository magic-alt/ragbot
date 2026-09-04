"""Durable PostgreSQL-backed ingestion worker.

Run with::

    python -m services.worker.main

The API persists jobs as ``pending``. Workers atomically claim jobs using the
repository lease contract, heartbeat while connectors/embedders run, recover
expired leases, retry bounded ingestion failures with backoff, dead-letter
exhausted work, periodically enqueue due recurring Source syncs, and drain the
publication outbox used by staged PostgreSQL/Qdrant generation cutovers.
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from services.api.app.factory import build_services_from_env
from services.api.app.storage.generation_support import ensure_generation_repository
from services.worker.pipeline import run_ingest_pipeline
from services.worker.reliability import (
    classify_ingestion_error,
    classify_persisted_failure,
    durable_retry_delay,
)
from services.worker.scheduler import schedule_due_sources
from services.worker.source_fence import job_source_generation, source_generation

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
    retry_base_seconds = _positive_float("RAGBOT_WORKER_RETRY_BASE_SECONDS", 5.0)
    retry_max_seconds = _positive_float("RAGBOT_WORKER_RETRY_MAX_SECONDS", 300.0)
    scheduler_scan_seconds = _nonnegative_float("RAGBOT_SCHEDULER_SCAN_SECONDS", 30.0)
    reconcile_seconds = _nonnegative_float("RAGBOT_RECONCILE_SECONDS", 30.0)
    publication_scan_seconds = _nonnegative_float("RAGBOT_PUBLICATION_OUTBOX_SCAN_SECONDS", 5.0)
    publication_max_attempts = _positive_int("RAGBOT_PUBLICATION_OUTBOX_MAX_ATTEMPTS", 10)
    worker_id = os.getenv("RAGBOT_WORKER_ID") or f"{socket.gethostname()}:{os.getpid()}"

    services = build_services_from_env()
    ensure_generation_repository(services.repo)
    claim = getattr(services.repo, "claim_next_job", None)
    heartbeat = getattr(services.repo, "heartbeat_job", None)
    release = getattr(services.repo, "release_job_lease", None)
    add_if_absent = getattr(services.repo, "add_job_if_absent", None)
    if not all(callable(item) for item in (claim, heartbeat, release, add_if_absent)):
        raise RuntimeError("Configured repository does not implement durable ingestion/scheduling contracts")

    logger.info(
        "Ingestion worker started: worker_id=%s lease=%ss max_attempts=%d retry=%s..%ss scheduler_scan=%ss reconcile=%ss publication_scan=%ss",
        worker_id,
        lease_seconds,
        max_attempts,
        retry_base_seconds,
        retry_max_seconds,
        scheduler_scan_seconds,
        reconcile_seconds,
        publication_scan_seconds,
    )
    next_schedule_scan = 0.0
    next_reconcile = 0.0
    next_publication_scan = 0.0
    try:
        while not _STOP.is_set():
            monotonic_now = time.monotonic()
            if reconcile_seconds > 0 and monotonic_now >= next_reconcile:
                _reconcile_queue(services.repo, max_attempts=max_attempts)
                next_reconcile = monotonic_now + reconcile_seconds

            if publication_scan_seconds > 0 and monotonic_now >= next_publication_scan:
                _drain_publication_outbox(
                    services.repo,
                    services.qdrant,
                    worker_id=worker_id,
                    lease_seconds=lease_seconds,
                    max_attempts=publication_max_attempts,
                    retry_base_seconds=retry_base_seconds,
                    retry_max_seconds=retry_max_seconds,
                )
                next_publication_scan = monotonic_now + publication_scan_seconds

            if scheduler_scan_seconds > 0 and monotonic_now >= next_schedule_scan:
                try:
                    stats = schedule_due_sources(services.repo)
                    if stats["enqueued"] or stats["blocked_active"]:
                        logger.info("Scheduled source scan: %s", stats)
                except Exception:
                    logger.exception("Scheduled source scan failed")
                next_schedule_scan = monotonic_now + scheduler_scan_seconds

            job = claim(worker_id, lease_seconds=lease_seconds, max_attempts=max_attempts)
            if job is None:
                _STOP.wait(poll_seconds)
                continue
            _execute_claimed_job(
                job,
                services,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                max_attempts=max_attempts,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
            )
    finally:
        _close_services(services)
        logger.info("Ingestion worker stopped: worker_id=%s", worker_id)
    return 0


def _execute_claimed_job(
    job,
    services,
    *,
    worker_id: str,
    lease_seconds: int,
    max_attempts: int = 3,
    retry_base_seconds: float = 5.0,
    retry_max_seconds: float = 300.0,
) -> None:
    current_source = services.repo.get_source(job.source_id)
    if current_source is None or current_source.status != "active" or current_source.tenant_id != job.tenant_id:
        reason = "Source unavailable, inactive, or tenant-mismatched at execution time"
        _dead_letter_job(
            services.repo,
            job,
            error=reason,
            failure_class="source_unavailable",
        )
        logger.error("Dead-lettering claimed job %s: %s", job.job_id, reason)
        return

    expected_generation = job_source_generation(job)
    if expected_generation and source_generation(current_source) != expected_generation:
        reason = (
            "Source lifecycle generation changed before worker execution: "
            f"expected={expected_generation} actual={source_generation(current_source)}"
        )
        _dead_letter_job(
            services.repo,
            job,
            error=reason,
            failure_class="source_generation_mismatch",
        )
        logger.error("Dead-lettering claimed job %s: %s", job.job_id, reason)
        return

    # Connector configuration is part of the durable job contract. A Source may
    # be edited while a job waits in the queue; execute the immutable snapshot
    # captured when the Job was submitted while retaining current metadata/ACL.
    source = replace(
        current_source,
        source_type=job.source_type,
        config=deepcopy(job.source_config or {}),
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
        # Preserve the established six-argument pipeline call contract. The
        # pipeline reads the durable Job generation snapshot itself; avoiding a
        # new positional argument keeps tests/extensions that monkeypatch the
        # pipeline callable source-compatible.
        result = run_ingest_pipeline(
            source,
            services.repo,
            services.qdrant,
            job.job_id,
            services.embedder,
            True,
        )
        if result.status == "failed":
            classification = classify_persisted_failure(result.error)
            _retry_or_dead_letter(
                services.repo,
                result,
                error=result.error or "Ingestion pipeline failed",
                failure_class=classification.failure_class,
                retryable=classification.retryable,
                max_attempts=max_attempts,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
            )
            return
        logger.info(
            "Ingestion job finished: job=%s status=%s attempts=%d",
            result.job_id,
            result.status,
            result.attempts,
        )
    except Exception as exc:  # pipeline normally persists failures itself
        logger.exception("Unexpected worker failure for job=%s", job.job_id)
        classification = classify_ingestion_error(exc)
        latest = services.repo.get_job(job.job_id) or job
        _retry_or_dead_letter(
            services.repo,
            latest,
            error=f"Worker execution failure: {exc}",
            failure_class=classification.failure_class,
            retryable=classification.retryable,
            max_attempts=max_attempts,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
        )
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=max(1.0, lease_seconds / 3))
        services.repo.release_job_lease(job.job_id, worker_id)


def _drain_publication_outbox(
    repo,
    qdrant,
    *,
    worker_id: str,
    lease_seconds: int,
    max_attempts: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    limit: int = 10,
) -> int:
    claim = getattr(repo, "claim_publication_outbox", None)
    complete = getattr(repo, "complete_publication_outbox", None)
    retry = getattr(repo, "retry_publication_outbox", None)
    reconcile = getattr(repo, "reconcile_publication_outbox", None)
    if not all(callable(item) for item in (claim, complete, retry, reconcile)):
        return 0

    try:
        stats = reconcile(max_attempts=max_attempts)
        if any(int(value or 0) for value in stats.values()):
            logger.warning("Publication outbox reconciliation repaired state: %s", stats)
    except Exception:
        logger.exception("Publication outbox reconciliation failed")
        return 0

    processed = 0
    try:
        events = claim(worker_id, lease_seconds=lease_seconds, limit=limit)
    except Exception:
        logger.exception("Publication outbox claim failed")
        return 0

    for event in events:
        try:
            if event.event_type != "delete_qdrant_points":
                raise ValueError(f"Unsupported publication outbox event: {event.event_type}")
            point_ids = list(dict.fromkeys(str(item) for item in (event.payload.get("point_ids") or []) if item))
            delete = getattr(qdrant, "delete_points", None)
            if point_ids and not callable(delete):
                raise RuntimeError("Configured vector store cannot delete staged/retired points")
            if point_ids:
                delete(point_ids)
            if not complete(event.outbox_id, worker_id):
                raise RuntimeError(f"Lost publication outbox lease: {event.outbox_id}")
            processed += 1
        except Exception as exc:
            logger.exception("Publication outbox event failed: id=%s type=%s", event.outbox_id, event.event_type)
            delay = durable_retry_delay(
                int(event.attempts or 0),
                base_seconds=retry_base_seconds,
                max_seconds=retry_max_seconds,
            )
            try:
                retry(
                    event.outbox_id,
                    worker_id,
                    str(exc),
                    delay,
                    max_attempts=max_attempts,
                )
            except Exception:
                logger.exception("Failed to reschedule publication outbox event: %s", event.outbox_id)
    return processed


def _retry_or_dead_letter(
    repo,
    job,
    *,
    error: str,
    failure_class: str,
    retryable: bool,
    max_attempts: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
) -> None:
    if not retryable or int(job.attempts or 0) >= max_attempts:
        _dead_letter_job(repo, job, error=error, failure_class=failure_class)
        logger.error(
            "Ingestion job dead-lettered: job=%s attempts=%d class=%s retryable=%s",
            job.job_id,
            int(job.attempts or 0),
            failure_class,
            retryable,
        )
        return

    delay = durable_retry_delay(
        int(job.attempts or 0),
        base_seconds=retry_base_seconds,
        max_seconds=retry_max_seconds,
    )
    now = datetime.now(timezone.utc)
    stats = dict(job.stats or {})
    history = list(stats.get("attempt_failures") or [])[-19:]
    history.append(
        {
            "attempt": int(job.attempts or 0),
            "at": now.isoformat(),
            "failure_class": failure_class,
            "error": str(error)[:1000],
            "retry_delay_seconds": delay,
        }
    )
    stats["attempt_failures"] = history
    repo.update_job(
        job.job_id,
        status="pending",
        error=error,
        failure_class=failure_class,
        completed_at=None,
        dead_lettered_at=None,
        available_at=(now + timedelta(seconds=delay)).isoformat(),
        lease_owner=None,
        lease_expires_at=None,
        stats=stats,
    )
    logger.warning(
        "Ingestion job scheduled for durable retry: job=%s attempt=%d/%d class=%s delay=%.1fs",
        job.job_id,
        int(job.attempts or 0),
        max_attempts,
        failure_class,
        delay,
    )


def _dead_letter_job(repo, job, *, error: str, failure_class: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    stats = dict(job.stats or {})
    stats["dead_letter"] = {
        "at": now,
        "attempts": int(job.attempts or 0),
        "failure_class": failure_class,
    }
    repo.update_job(
        job.job_id,
        status="dead_lettered",
        error=error,
        failure_class=failure_class,
        completed_at=job.completed_at or now,
        dead_lettered_at=now,
        lease_owner=None,
        lease_expires_at=None,
        stats=stats,
    )


def _reconcile_queue(repo, *, max_attempts: int) -> None:
    reconcile = getattr(repo, "reconcile_ingestion_jobs", None)
    if not callable(reconcile):
        return
    try:
        stats = reconcile(max_attempts=max_attempts)
        if any(int(value or 0) for value in stats.values()):
            logger.warning("Ingestion queue reconciliation repaired state: %s", stats)
    except Exception:
        logger.exception("Ingestion queue reconciliation failed")


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


def _nonnegative_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
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
