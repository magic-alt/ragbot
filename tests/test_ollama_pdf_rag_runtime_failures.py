from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from services.worker.reliability import classify_persisted_failure
import services.worker.jobs.ingest_pdf as ingest_pdf_module

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ollama_pdf_rag_test.py"

_spec = importlib.util.spec_from_file_location("ollama_pdf_rag_test_runtime_failures", SCRIPT)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)


def _job(status: str, **overrides):
    job = {
        "job_id": "job-smoke",
        "status": status,
        "doc_count": 0,
        "chunk_count": 0,
        "attempts": 1,
        "lease_owner": None,
        "failure_class": None,
        "source_type": "pdf",
        "source_config": {"path": "/data/sample.pdf"},
        "error": None,
    }
    job.update(overrides)
    return job


def test_wait_job_ignores_transient_failed_state_until_completion() -> None:
    responses = [
        _job("failed", error="temporary pipeline failure"),
        _job("pending", error="temporary pipeline failure"),
        _job("completed", doc_count=1, chunk_count=12, error=None),
    ]

    with patch.object(mod._impl, "_request_json", side_effect=responses), patch.object(
        mod.time, "sleep", return_value=None
    ):
        result = mod._wait_job(
            "http://127.0.0.1:8000",
            "job-smoke",
            headers={},
            timeout=30.0,
        )

    assert result["status"] == "completed"
    assert result["chunk_count"] == 12


def test_wait_job_reports_dead_letter_and_stale_worker_hint() -> None:
    response = _job(
        "dead_lettered",
        failure_class="configuration_error",
        error="Local source is outside RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS",
    )

    with patch.object(mod._impl, "_request_json", return_value=response):
        with pytest.raises(mod.UserError, match="stale or foreign worker") as exc_info:
            mod._wait_job(
                "http://127.0.0.1:8000",
                "job-smoke",
                headers={},
                timeout=30.0,
            )

    message = str(exc_info.value)
    assert "attempts=1" in message
    assert "failure_class='configuration_error'" in message
    assert '"path": "/data/sample.pdf"' in message


def test_running_job_requires_durable_claim_metadata() -> None:
    job = _job("running", attempts=0, lease_owner=None)
    message = mod._claim_invariant_error(job)
    assert message is not None
    assert "attempts>=1" in message
    assert "inline execution or a stale/legacy executor" in message


def test_running_job_rejects_foreign_worker_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_EXPECTED_WORKER_ID", "smoke-worker")
    job = _job("running", attempts=1, lease_owner="other-worker")
    message = mod._claim_invariant_error(job)
    assert message is not None
    assert "Foreign worker claimed" in message
    assert "other-worker" in message


def test_running_job_accepts_expected_worker_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_EXPECTED_WORKER_ID", "smoke-worker")
    job = _job("running", attempts=1, lease_owner="smoke-worker")
    assert mod._claim_invariant_error(job) is None


def test_macos_python_client_on_isolated_postgres_port_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "_SMOKE_POSTGRES_PORT", 55432)
    fake_lsof = SimpleNamespace(
        returncode=0,
        stdout=(
            "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n"
            "Python 5151 user 10u IPv4 0t0 TCP 127.0.0.1:60000->127.0.0.1:55432 (ESTABLISHED)\n"
        ),
    )

    with patch.object(mod.sys, "platform", "darwin"), patch.object(
        mod.subprocess, "run", return_value=fake_lsof
    ):
        with pytest.raises(mod.UserError, match="Competing host PostgreSQL Python process detected"):
            mod._assert_no_competing_host_worker()


def test_macos_python_client_on_default_postgres_port_does_not_block_isolated_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "_SMOKE_POSTGRES_PORT", 55432)
    fake_lsof = SimpleNamespace(returncode=1, stdout="")

    with patch.object(mod.sys, "platform", "darwin"), patch.object(
        mod.subprocess, "run", return_value=fake_lsof
    ) as run_mock:
        mod._assert_no_competing_host_worker()

    command = run_mock.call_args.args[0]
    assert "-iTCP:55432" in command


def test_compose_env_assigns_unique_worker_and_isolated_postgres_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = SimpleNamespace(
        model="qwen3.8:27b-mlx",
        ollama_timeout=300.0,
        reasoning_effort="none",
        docker_ollama_url="http://host.docker.internal:11434",
        embedding_model="qwen3-embedding:8b",
        embedding_dim=4096,
        collection="rag_chunks_smoke_qwen3_embedding_8b_4096",
        data_dir=tmp_path,
        port=8000,
    )
    monkeypatch.delenv("RAGBOT_SMOKE_POSTGRES_PORT", raising=False)

    env = mod._compose_env(args)

    assert env["RAGBOT_INGESTION_MODE"] == "worker"
    assert env["RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS"] == "/data"
    assert env["RAGBOT_WORKER_ID"].startswith("ollama-pdf-smoke-")
    assert mod._EXPECTED_WORKER_ID == env["RAGBOT_WORKER_ID"]
    assert env["RAGBOT_POSTGRES_PORT"] == "55432"
    assert mod._SMOKE_POSTGRES_PORT == 55432


def test_compose_env_allows_custom_smoke_postgres_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = SimpleNamespace(
        model="qwen3.8:27b-mlx",
        ollama_timeout=300.0,
        reasoning_effort="none",
        docker_ollama_url="http://host.docker.internal:11434",
        embedding_model="qwen3-embedding:8b",
        embedding_dim=4096,
        collection="rag_chunks_smoke_qwen3_embedding_8b_4096",
        data_dir=tmp_path,
        port=8000,
    )
    monkeypatch.setenv("RAGBOT_SMOKE_POSTGRES_PORT", "56432")

    env = mod._compose_env(args)

    assert env["RAGBOT_POSTGRES_PORT"] == "56432"
    assert mod._SMOKE_POSTGRES_PORT == 56432


def test_local_path_policy_failure_is_permanent_configuration_error() -> None:
    result = classify_persisted_failure(
        "PDF local path rejected: requested='/data/sample.pdf'; "
        "Local source is outside RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS"
    )

    assert result.failure_class == "configuration_error"
    assert result.retryable is False


def test_pdf_ingestion_wraps_local_path_failure_with_runtime_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-test")
    monkeypatch.setenv("RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS", "/data")

    with patch.object(
        ingest_pdf_module,
        "fetch_pdf",
        side_effect=ValueError("Local source is outside RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS"),
    ):
        with pytest.raises(ValueError, match="PDF local path rejected") as exc_info:
            list(
                ingest_pdf_module.ingest_pdf(
                    str(pdf),
                    doc_id="doc-smoke",
                    tenant_id="tenant-smoke",
                )
            )

    message = str(exc_info.value)
    assert f"requested={str(pdf)!r}" in message
    assert "allowed_roots='/data'" in message
    assert "resolved=" in message
