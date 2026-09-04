from __future__ import annotations

import pytest

from cli.job_wait import format_job_knowledge, job_chunk_stats, wait_for_job


def test_job_chunk_stats_prefers_pipeline_snapshot_stats() -> None:
    job = {
        "doc_count": 1,
        "chunk_count": 0,
        "stats": {
            "chunks_total": 287,
            "chunks_ingested": 0,
            "chunks_reused": 287,
        },
    }

    assert job_chunk_stats(job) == {"total": 287, "written": 0, "reused": 287}
    assert format_job_knowledge(job) == "docs=1, chunks=287, written=0, reused=287"


def test_job_chunk_stats_falls_back_to_legacy_written_count() -> None:
    job = {"doc_count": 2, "chunk_count": 14}

    assert job_chunk_stats(job) == {"total": 14, "written": 14, "reused": 0}
    assert format_job_knowledge(job) == "docs=2, chunks=14, written=14, reused=0"


def test_wait_for_job_prints_total_written_and_reused(capsys) -> None:
    completed = {
        "status": "completed",
        "doc_count": 1,
        "chunk_count": 0,
        "stats": {
            "chunks_total": 42,
            "chunks_ingested": 0,
            "chunks_reused": 42,
        },
    }

    def request_fn(*args, **kwargs):
        return completed

    result = wait_for_job(
        request_fn,
        "http://127.0.0.1:8000",
        "job-1",
        headers={},
        timeout=1,
        poll_interval=0.1,
    )

    assert result is completed
    output = capsys.readouterr().out
    assert "completed (docs=1, chunks=42, written=0, reused=42)" in output


def test_wait_for_job_failure_includes_durable_source_snapshot() -> None:
    failed = {
        "job_id": "job-path",
        "status": "failed",
        "doc_count": 0,
        "chunk_count": 0,
        "attempts": 1,
        "source_type": "pdf",
        "source_config": {"path": "ragbot-data:///manual.pdf"},
        "error": "Local source is outside RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS",
        "failure_class": "configuration_error",
        "stats": {},
    }

    def request_fn(*args, **kwargs):
        return failed

    with pytest.raises(RuntimeError) as exc_info:
        wait_for_job(
            request_fn,
            "http://127.0.0.1:8000",
            "job-path",
            headers={},
            timeout=1,
            poll_interval=0.1,
            quiet=True,
        )

    message = str(exc_info.value)
    assert "job_id=job-path" in message
    assert "attempts=1" in message
    assert '"path": "ragbot-data:///manual.pdf"' in message
