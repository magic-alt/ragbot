from __future__ import annotations

import argparse
import json
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Iterable

from services.api.app.storage.managed_pg_repo import ManagedPostgresRepo
from services.api.app.storage.models import Chunk, Document, IngestionJob, Source
from services.api.app.storage.postgres_performance import ensure_database_performance
from services.api.app.storage.query_support import ensure_query_repository


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _chunks(prefix: str, doc_id: str, tenant_id: str, source_id: str, count: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{prefix}-chunk-{index}",
            doc_id=doc_id,
            tenant_id=tenant_id,
            chunk_index=index,
            text=(f"postgres scale benchmark servo ethercat row {index} " * 2).strip(),
            checksum=f"{prefix}-checksum-{index}",
            source_id=source_id,
            metadata={"source_type": "local_fs", "tags": ["postgres-scale"]},
        )
        for index in range(count)
    ]


def _seed_doc(repo: Any, prefix: str, tenant_id: str, source_id: str) -> str:
    now = datetime.now(timezone.utc).isoformat()
    doc_id = f"{prefix}-doc"
    repo.add_document(
        Document(
            doc_id=doc_id,
            tenant_id=tenant_id,
            source_type="local_fs",
            title=prefix,
            uri=f"source://{source_id}/{prefix}",
            version="1",
            doc_updated_at=now,
            ingested_at=now,
            source_id=source_id,
        )
    )
    return doc_id


def _node_types(plan: Any) -> list[str]:
    result: list[str] = []
    if isinstance(plan, dict):
        node_type = plan.get("Node Type")
        if node_type:
            result.append(str(node_type))
        for value in plan.values():
            result.extend(_node_types(value))
    elif isinstance(plan, list):
        for item in plan:
            result.extend(_node_types(item))
    return result


def _explain(conn: Any, sql: str, params: tuple[Any, ...]) -> dict[str, Any]:
    with conn.transaction():
        # With sequential scans disabled, a missing usable index still produces
        # a Seq Scan node with prohibitive cost. This removes cardinality/runner
        # variance from the regression assertion while preserving plan truth.
        conn.execute("SET LOCAL enable_seqscan = off")
        row = conn.execute(
            "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql,
            params,
        ).fetchone()
    payload = row["QUERY PLAN"] if isinstance(row, dict) else row[0]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload[0]


def _assert_indexed_plan(name: str, plan: dict[str, Any]) -> dict[str, Any]:
    node_types = _node_types(plan)
    if "Seq Scan" in node_types:
        raise AssertionError(f"{name} unexpectedly requires Seq Scan: {node_types}")
    root = plan.get("Plan") or {}
    return {
        "node_types": node_types,
        "planning_ms": round(float(plan.get("Planning Time", 0.0)), 3),
        "execution_ms": round(float(plan.get("Execution Time", 0.0)), 3),
        "rows": int(root.get("Actual Rows", 0) or 0),
    }


def _seed_jobs(repo: Any, source: Source, count: int, prefix: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for index in range(count):
        repo.add_job(
            IngestionJob(
                job_id=f"{prefix}-job-{index}",
                tenant_id=source.tenant_id,
                source_id=source.source_id,
                source_type=source.source_type,
                source_config={},
                status="pending",
                created_at=now,
                available_at=now,
            )
        )


def _claim_benchmark(repo: Any, workers: int, jobs: int) -> dict[str, Any]:
    suffix = uuid.uuid4().hex[:10]
    source = Source(
        source_id=f"claim-source-{suffix}",
        tenant_id=f"claim-tenant-{suffix}",
        source_type="local_fs",
        name="claim",
        config={"path": "/tmp"},
    )
    repo.add_source(source)
    _seed_jobs(repo, source, jobs, f"claim-{workers}-{suffix}")
    latencies: list[float] = []

    def claim_one(index: int) -> str | None:
        started = time.perf_counter()
        job = repo.claim_next_job(
            f"claim-worker-{workers}-{index}", lease_seconds=300, max_attempts=3
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        return job.job_id if job else None

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        claimed = list(executor.map(claim_one, range(jobs)))
    elapsed = max(1e-9, time.perf_counter() - started)
    claimed_count = sum(1 for item in claimed if item)
    if claimed_count != jobs:
        raise AssertionError(f"claim benchmark lost work: {claimed_count}/{jobs}")
    return {
        "workers": workers,
        "jobs": jobs,
        "throughput_jobs_s": round(claimed_count / elapsed, 2),
        "p50_ms": round(_percentile(latencies, 0.50), 3),
        "p95_ms": round(_percentile(latencies, 0.95), 3),
        "max_ms": round(max(latencies or [0.0]), 3),
    }


def run(
    dsn: str,
    *,
    chunks: int,
    plan_jobs: int,
    claim_jobs: int,
    workers: Iterable[int],
    min_copy_speedup: float,
    max_claim_p95_ms: float,
) -> dict[str, Any]:
    repo = ManagedPostgresRepo(dsn, pool_min=2, pool_max=max(40, max(workers, default=1) + 4))
    ensure_query_repository(repo)
    ensure_database_performance(repo)
    repo._ragbot_pg_copy_min_rows = 1
    suffix = uuid.uuid4().hex[:10]
    tenant_id = f"pg-bench-tenant-{suffix}"
    source = Source(
        source_id=f"pg-bench-source-{suffix}",
        tenant_id=tenant_id,
        source_type="local_fs",
        name="Postgres benchmark",
        config={"path": "/tmp/pg-bench"},
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    repo.add_source(source)

    try:
        legacy_doc = _seed_doc(repo, f"legacy-{suffix}", tenant_id, source.source_id)
        legacy_rows = _chunks(f"legacy-{suffix}", legacy_doc, tenant_id, source.source_id, chunks)
        started = time.perf_counter()
        repo._ragbot_legacy_add_chunks(legacy_rows)
        legacy_seconds = max(1e-9, time.perf_counter() - started)

        copy_doc = _seed_doc(repo, f"copy-{suffix}", tenant_id, source.source_id)
        copy_rows = _chunks(f"copy-{suffix}", copy_doc, tenant_id, source.source_id, chunks)
        started = time.perf_counter()
        repo.add_chunks(copy_rows)
        copy_seconds = max(1e-9, time.perf_counter() - started)
        speedup = legacy_seconds / copy_seconds
        if min_copy_speedup > 0 and speedup < min_copy_speedup:
            raise AssertionError(
                f"COPY bulk path speedup {speedup:.2f}x < required {min_copy_speedup:.2f}x"
            )

        plan_source = Source(
            source_id=f"plan-source-{suffix}",
            tenant_id=f"plan-tenant-{suffix}",
            source_type="local_fs",
            name="plan",
            config={"path": "/tmp"},
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        repo.add_source(plan_source)
        _seed_jobs(repo, plan_source, plan_jobs, f"plan-{suffix}")
        with repo._pool.connection() as conn:
            conn.execute("ANALYZE ingestion_jobs")
            conn.execute("ANALYZE documents")
            conn.execute("ANALYZE sources")
            plans = {
                "queue_claim": _assert_indexed_plan(
                    "queue_claim",
                    _explain(
                        conn,
                        """
                        SELECT job_id FROM ingestion_jobs
                        WHERE status = 'pending' AND available_at <= NOW() AND attempts < %s
                        ORDER BY available_at, created_at, job_id
                        LIMIT 1
                        """,
                        (3,),
                    ),
                ),
                "job_keyset": _assert_indexed_plan(
                    "job_keyset",
                    _explain(
                        conn,
                        """
                        SELECT job_id FROM ingestion_jobs
                        WHERE tenant_id = %s
                        ORDER BY created_at DESC, job_id DESC
                        LIMIT 100
                        """,
                        (plan_source.tenant_id,),
                    ),
                ),
                "source_documents": _assert_indexed_plan(
                    "source_documents",
                    _explain(
                        conn,
                        """
                        SELECT doc_id FROM documents
                        WHERE source_id = %s
                        ORDER BY ingested_at DESC, doc_id DESC
                        LIMIT 100
                        """,
                        (source.source_id,),
                    ),
                ),
            }

        claim_results = [
            _claim_benchmark(repo, int(worker_count), claim_jobs)
            for worker_count in workers
        ]
        if max_claim_p95_ms > 0:
            slowest = max((row["p95_ms"] for row in claim_results), default=0.0)
            if slowest > max_claim_p95_ms:
                raise AssertionError(
                    f"claim p95 {slowest:.2f}ms exceeds {max_claim_p95_ms:.2f}ms"
                )

        return {
            "chunks": chunks,
            "bulk": {
                "legacy_seconds": round(legacy_seconds, 3),
                "copy_seconds": round(copy_seconds, 3),
                "legacy_rows_s": round(chunks / legacy_seconds, 2),
                "copy_rows_s": round(chunks / copy_seconds, 2),
                "copy_speedup": round(speedup, 3),
            },
            "plans": plans,
            "claims": claim_results,
            "runtime_metrics": repo.database_runtime_metrics(),
        }
    finally:
        repo.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Ragbot PostgreSQL hot paths")
    parser.add_argument("--dsn", default=os.getenv("POSTGRES_TEST_DSN") or os.getenv("POSTGRES_DSN"))
    parser.add_argument("--chunks", type=int, default=int(os.getenv("RAGBOT_PG_BENCH_CHUNKS", "10000")))
    parser.add_argument("--plan-jobs", type=int, default=int(os.getenv("RAGBOT_PG_BENCH_PLAN_JOBS", "10000")))
    parser.add_argument("--claim-jobs", type=int, default=int(os.getenv("RAGBOT_PG_BENCH_CLAIM_JOBS", "128")))
    parser.add_argument("--workers", default=os.getenv("RAGBOT_PG_BENCH_WORKERS", "1,8"))
    parser.add_argument(
        "--min-copy-speedup",
        type=float,
        default=float(os.getenv("RAGBOT_PG_BENCH_MIN_COPY_SPEEDUP", "0")),
    )
    parser.add_argument(
        "--max-claim-p95-ms",
        type=float,
        default=float(os.getenv("RAGBOT_PG_BENCH_MAX_CLAIM_P95_MS", "1000")),
    )
    args = parser.parse_args()
    if not args.dsn:
        parser.error("--dsn or POSTGRES_TEST_DSN/POSTGRES_DSN is required")
    workers = [int(value) for value in str(args.workers).split(",") if value.strip()]
    result = run(
        args.dsn,
        chunks=max(1, args.chunks),
        plan_jobs=max(1, args.plan_jobs),
        claim_jobs=max(1, args.claim_jobs),
        workers=workers or [1],
        min_copy_speedup=max(0.0, args.min_copy_speedup),
        max_claim_p95_ms=max(0.0, args.max_claim_p95_ms),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
