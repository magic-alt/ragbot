from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

# One repository-wide gate coordinating very short lifecycle transitions.
# Worker job claims take a SHARED advisory lock, so workers still claim in
# parallel. Index alias cutover takes the EXCLUSIVE form, blocking new claims
# while it confirms no already-running ingestion can publish vectors prepared
# for the old physical index.
PUBLICATION_BARRIER_KEY = 1380007233


def _standalone_pg_connection(repo: Any):
    pool = getattr(repo, "_pool", None)
    conninfo = getattr(pool, "conninfo", None)
    if not conninfo:
        return None
    try:
        import psycopg
    except ImportError:
        return None
    return psycopg.connect(conninfo, autocommit=True)


@contextmanager
def shared_claim_barrier(repo: Any) -> Iterator[None]:
    """Allow concurrent worker claims, but wait behind an index cutover."""
    conn = _standalone_pg_connection(repo)
    if conn is None:
        # Index lifecycle is production-PostgreSQL only today. Keep a lightweight
        # local lock fallback for tests/custom built-ins without widening Repo.
        lock = getattr(repo, "_publication_claim_gate", None)
        if lock is None:
            lock = threading.RLock()
            setattr(repo, "_publication_claim_gate", lock)
        with lock:
            yield
        return

    try:
        conn.execute("SELECT pg_advisory_lock_shared(%s)", (PUBLICATION_BARRIER_KEY,))
        yield
    finally:
        try:
            conn.execute("SELECT pg_advisory_unlock_shared(%s)", (PUBLICATION_BARRIER_KEY,))
        finally:
            conn.close()


@contextmanager
def exclusive_cutover_barrier(repo: Any) -> Iterator[Optional[Any]]:
    """Block new worker claims while an IndexVersion performs alias cutover."""
    conn = _standalone_pg_connection(repo)
    if conn is None:
        lock = getattr(repo, "_publication_claim_gate", None)
        if lock is None:
            lock = threading.RLock()
            setattr(repo, "_publication_claim_gate", lock)
        with lock:
            yield None
        return

    try:
        conn.execute("SELECT pg_advisory_lock(%s)", (PUBLICATION_BARRIER_KEY,))
        yield conn
    finally:
        try:
            conn.execute("SELECT pg_advisory_unlock(%s)", (PUBLICATION_BARRIER_KEY,))
        finally:
            conn.close()


def ensure_worker_claim_gate(repo: Any) -> Any:
    """Wrap one repository instance's durable claim surface exactly once."""
    if getattr(repo, "_ragbot_publication_claim_gate_installed", False):
        return repo
    original = getattr(repo, "claim_next_job", None)
    if not callable(original):
        return repo

    def gated_claim(worker_id: str, lease_seconds: int = 120, max_attempts: int = 3):
        with shared_claim_barrier(repo):
            return original(
                worker_id,
                lease_seconds=lease_seconds,
                max_attempts=max_attempts,
            )

    setattr(repo, "claim_next_job", gated_claim)
    setattr(repo, "_ragbot_publication_claim_gate_installed", True)
    return repo


def running_ingestion_count(repo: Any, barrier_connection: Optional[Any] = None) -> int:
    """Return already-running durable jobs while new claims are gated."""
    if barrier_connection is not None:
        row = barrier_connection.execute(
            "SELECT COUNT(*) AS n FROM ingestion_jobs WHERE status = 'running'"
        ).fetchone()
        if row is None:
            return 0
        if hasattr(row, "keys"):
            return int(dict(row).get("n") or 0)
        return int(row[0])

    list_jobs = getattr(repo, "list_jobs", None)
    if not callable(list_jobs):
        return 0
    return sum(1 for job in list_jobs() if getattr(job, "status", None) == "running")
