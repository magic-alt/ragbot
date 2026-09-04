from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional


TERMINAL_SUCCESS = frozenset({"completed"})
TERMINAL_FAILURE = frozenset({"failed", "dead_lettered"})


def wait_for_job(
    request_fn: Callable[..., Dict[str, Any]],
    server: str,
    job_id: str,
    *,
    headers: Dict[str, str],
    timeout: float,
    poll_interval: float,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Poll one ingestion Job and treat DLQ as an immediate terminal failure."""
    deadline = time.monotonic() + timeout
    previous_status: Optional[str] = None
    while True:
        job = request_fn(server, "GET", f"/ingest/jobs/{job_id}", headers=headers, timeout=60)
        status = str(job.get("status", "unknown"))
        if not quiet and status != previous_status:
            details = []
            if job.get("doc_count") is not None:
                details.append(f"docs={job.get('doc_count', 0)}")
            if job.get("chunk_count") is not None:
                details.append(f"chunks={job.get('chunk_count', 0)}")
            suffix = f" ({', '.join(details)})" if details else ""
            print(f"Ingestion {job_id}: {status}{suffix}")
            previous_status = status

        if status in TERMINAL_SUCCESS:
            return job
        if status in TERMINAL_FAILURE:
            failure_class = str(job.get("failure_class") or "").strip()
            prefix = f"[{failure_class}] " if failure_class else ""
            raise RuntimeError(
                prefix + str(job.get("error") or f"Ingestion job {status}: {job_id}")
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for ingestion job {job_id} after {timeout:.0f}s")
        time.sleep(max(0.1, poll_interval))
