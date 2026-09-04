from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional


TERMINAL_SUCCESS = frozenset({"completed"})
TERMINAL_FAILURE = frozenset({"failed", "dead_lettered"})


def job_chunk_stats(job: Dict[str, Any]) -> Dict[str, int]:
    """Return unambiguous ingestion chunk counts for one Job response.

    ``job.chunk_count`` is the historical count of chunks written by this Job,
    while the ingestion pipeline stores the resulting knowledge snapshot size in
    ``stats.chunks_total`` and incremental reuse in ``stats.chunks_reused``.
    Keep those semantics intact and normalize them for human-facing CLI output.
    """
    stats = job.get("stats") if isinstance(job.get("stats"), dict) else {}
    written = int(stats.get("chunks_ingested", job.get("chunk_count", 0)) or 0)
    total = int(stats.get("chunks_total", written) or 0)
    reused = int(stats.get("chunks_reused", max(0, total - written)) or 0)
    return {"total": total, "written": written, "reused": reused}


def format_job_knowledge(job: Dict[str, Any]) -> str:
    counts = job_chunk_stats(job)
    docs = int(job.get("doc_count", 0) or 0)
    return (
        f"docs={docs}, chunks={counts['total']}, "
        f"written={counts['written']}, reused={counts['reused']}"
    )


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
            print(f"Ingestion {job_id}: {status} ({format_job_knowledge(job)})")
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
