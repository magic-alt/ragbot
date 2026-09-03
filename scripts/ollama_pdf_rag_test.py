#!/usr/bin/env python3
"""Stable entrypoint for the Ollama PDF RAG smoke-test implementation."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
IMPL = SCRIPT_DIR / "ollama_pdf_rag_test_impl.py"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
_FAILED_SETTLE_SECONDS = 3.0
_EXPECTED_WORKER_ID: str | None = None
_SMOKE_POSTGRES_PORT = 55432


def _installed_embedding_models() -> list[str]:
    """Return locally installed Ollama models that look embedding-specific."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    models: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        fields = line.split()
        if not fields:
            continue
        name = fields[0]
        lowered = name.lower()
        if "embedding" in lowered or lowered.startswith(("nomic-embed", "mxbai-embed", "bge-")):
            models.append(name)
    return models


def _resolve_embedding_default() -> str:
    """Choose an Ollama embedding default without inheriting OpenAI .env defaults."""
    explicit = os.getenv("OLLAMA_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL")
    if explicit:
        return explicit
    installed = _installed_embedding_models()
    if DEFAULT_OLLAMA_EMBEDDING_MODEL in installed:
        return DEFAULT_OLLAMA_EMBEDDING_MODEL
    if len(installed) == 1:
        return installed[0]
    return DEFAULT_OLLAMA_EMBEDDING_MODEL


def _load_impl() -> ModuleType:
    if "--embedding-model" not in sys.argv and not os.getenv("EMBEDDING_MODEL"):
        os.environ["EMBEDDING_MODEL"] = _resolve_embedding_default()
    spec = importlib.util.spec_from_file_location("ollama_pdf_rag_test_impl", IMPL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load smoke-test implementation: {IMPL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_contract_code(service: str, expected_worker_id: str | None = None) -> str:
    label = repr(service)
    worker_assert = ""
    if service == "worker" and expected_worker_id:
        worker_assert = (
            f"assert os.environ.get('RAGBOT_WORKER_ID') == {expected_worker_id!r}, "
            f"f\"worker RAGBOT_WORKER_ID={{os.environ.get('RAGBOT_WORKER_ID')!r}}, "
            f"expected {expected_worker_id!r}\"\n"
        )
    return f"""
import os
from pathlib import Path
from services.api.app.routes.ingest import _use_durable_worker
from services.worker.connectors.security import validate_local_source_path

service = {label}
mode = os.environ.get("RAGBOT_INGESTION_MODE")
allowed = os.environ.get("RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS")
dsn = os.environ.get("POSTGRES_DSN")
assert mode == "worker", f"{{service}} RAGBOT_INGESTION_MODE={{mode!r}}, expected 'worker'"
assert dsn, f"{{service}} POSTGRES_DSN is empty; durable ingestion cannot be used"
assert allowed == "/data", f"{{service}} RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS={{allowed!r}}, expected '/data'"
assert _use_durable_worker() is True, f"{{service}} API runtime resolved ingestion to inline mode"
{worker_assert}root = Path("/data")
assert root.is_dir(), f"{{service}} /data is not a mounted directory"
root_resolved = root.resolve()
pdfs = sorted(
    (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
    key=lambda path: path.as_posix().lower(),
)
assert pdfs, f"{{service}} sees no PDF files below /data"
print(f"{{service}}-contract mode={{mode}} allowed={{allowed}} root={{root_resolved}} pdfs={{len(pdfs)}}")
for path in pdfs:
    requested = str(path)
    resolved = validate_local_source_path(requested)
    assert Path(resolved).is_file(), f"{{service}} validated path is not a file: {{resolved}}"
    print(f"{{service}}-pdf-ok requested={{requested!r}} resolved={{resolved!r}}")
"""


def _failure_detail(job: Dict[str, Any]) -> str:
    error = str(job.get("error") or "Ingestion failed")
    source_config = job.get("source_config") or {}
    detail = (
        f"Ingestion {job.get('job_id')}: status={job.get('status')}, "
        f"attempts={job.get('attempts', 0)}, lease_owner={job.get('lease_owner')!r}, "
        f"failure_class={job.get('failure_class')!r}, source_type={job.get('source_type')!r}, "
        f"source_config={json.dumps(source_config, ensure_ascii=False)}; error={error}"
    )
    if "Local source is outside RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS" in error:
        if "PDF local path rejected:" not in error:
            detail += (
                ". The API/worker preflight passed the mounted /data PDFs, but this failure lacks "
                "the current PDF path diagnostics. This strongly suggests a stale or foreign worker "
                "is consuming the same PostgreSQL ingestion queue."
            )
    return detail


def _host_postgres_python_clients(port: int) -> list[str]:
    """Return host Python processes connected to the smoke stack's PostgreSQL port."""
    if sys.platform != "darwin":
        return []
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:ESTABLISHED"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode not in {0, 1}:
        return []
    matches: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        command = stripped.split(None, 1)[0].lower()
        if command.startswith(("python", "pypy")):
            matches.append(stripped)
    return matches


def _assert_no_competing_host_worker() -> None:
    """Reject host-side Python clients connected to the isolated smoke PostgreSQL port."""
    if sys.platform != "darwin":
        return
    postgres_python = _host_postgres_python_clients(_SMOKE_POSTGRES_PORT)
    if postgres_python:
        preview = " | ".join(postgres_python[:8])
        raise _impl.UserError(
            "Competing host PostgreSQL Python process detected on the isolated smoke port. "
            "It can claim jobs without the Docker /data mount. Stop it before running this smoke "
            f"test: {preview}"
        )
    print(
        "host-worker-contract ok: no host Python client connected to "
        f"PostgreSQL port {_SMOKE_POSTGRES_PORT}"
    )


_impl = _load_impl()
_impl_compose_env = _impl._compose_env
_impl_verify_container_contract = _impl._verify_container_contract


def _compose_env(args):
    """Force an isolated PostgreSQL host port and uniquely identifiable durable worker."""
    global _EXPECTED_WORKER_ID, _SMOKE_POSTGRES_PORT
    env = _impl_compose_env(args)
    env["RAGBOT_INGESTION_MODE"] = "worker"
    env["RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS"] = "/data"
    _EXPECTED_WORKER_ID = f"ollama-pdf-smoke-{os.getpid()}-{int(time.time())}"
    env["RAGBOT_WORKER_ID"] = _EXPECTED_WORKER_ID
    try:
        _SMOKE_POSTGRES_PORT = int(os.getenv("RAGBOT_SMOKE_POSTGRES_PORT", "55432"))
    except ValueError as exc:
        raise _impl.UserError("RAGBOT_SMOKE_POSTGRES_PORT must be an integer") from exc
    if not 1 <= _SMOKE_POSTGRES_PORT <= 65535:
        raise _impl.UserError("RAGBOT_SMOKE_POSTGRES_PORT must be between 1 and 65535")
    env["RAGBOT_POSTGRES_PORT"] = str(_SMOKE_POSTGRES_PORT)
    return env


def _verify_container_contract(args, env) -> None:
    """Validate API/worker runtime and isolated queue ownership before ingestion."""
    _impl_verify_container_contract(args, env)
    for service in ("api", "worker"):
        _impl._run(
            _impl._compose_command()
            + [
                "exec",
                "-T",
                service,
                "python",
                "-c",
                _runtime_contract_code(service, _EXPECTED_WORKER_ID),
            ],
            env=env,
        )
    _assert_no_competing_host_worker()
    print(
        f"queue-claim-contract expected_worker_id={_EXPECTED_WORKER_ID} "
        f"postgres_host_port={_SMOKE_POSTGRES_PORT}"
    )


def _claim_invariant_error(job: Dict[str, Any]) -> str | None:
    if str(job.get("status") or "") != "running":
        return None
    attempts = int(job.get("attempts") or 0)
    lease_owner = str(job.get("lease_owner") or "")
    if attempts < 1 or not lease_owner:
        return (
            "Durable queue invariant violated: a running ingestion job must have attempts>=1 and "
            f"a non-empty lease_owner, got attempts={attempts}, lease_owner={lease_owner!r}. "
            "This indicates inline execution or a stale/legacy executor bypassing claim_next_job()."
        )
    if _EXPECTED_WORKER_ID and lease_owner != _EXPECTED_WORKER_ID:
        return (
            "Foreign worker claimed the smoke-test job: "
            f"lease_owner={lease_owner!r}, expected={_EXPECTED_WORKER_ID!r}. The smoke stack uses "
            f"isolated PostgreSQL host port {_SMOKE_POSTGRES_PORT}; inspect extra Docker workers if "
            "this still occurs."
        )
    return None


def _wait_job(
    server: str,
    job_id: str,
    *,
    headers: Dict[str, str],
    timeout: float,
) -> Dict[str, Any]:
    """Wait through durable transitions while enforcing worker-claim invariants."""
    deadline = time.monotonic() + timeout
    previous = None
    failed_since: float | None = None
    while True:
        job = _impl._request_json(
            "GET",
            f"{server.rstrip('/')}/ingest/jobs/{job_id}",
            headers=headers,
            timeout=min(60.0, timeout),
        )
        status = str(job.get("status") or "unknown")
        if status != previous:
            print(
                f"Ingestion {job_id}: {status} "
                f"(docs={job.get('doc_count', 0)}, chunks={job.get('chunk_count', 0)}, "
                f"attempts={job.get('attempts', 0)}, lease_owner={job.get('lease_owner')!r}, "
                f"failure_class={job.get('failure_class')!r})"
            )
            previous = status

        invariant_error = _claim_invariant_error(job)
        if invariant_error:
            raise _impl.UserError(f"{invariant_error} Job snapshot: {_failure_detail(job)}")

        if status == "completed":
            return job
        if status == "dead_lettered":
            raise _impl.UserError(_failure_detail(job))

        now = time.monotonic()
        if status == "failed":
            if failed_since is None:
                failed_since = now
            if now - failed_since >= _FAILED_SETTLE_SECONDS:
                raise _impl.UserError(_failure_detail(job))
            time.sleep(0.2)
            continue
        failed_since = None

        if now >= deadline:
            raise _impl.UserError(f"Timed out waiting for ingestion job {job_id}; last status={status}")
        time.sleep(1.0)


# The implementation's main() resolves helpers from its own module globals, so
# replace them there as well as exporting them through this stable entrypoint.
_impl._compose_env = _compose_env
_impl._verify_container_contract = _verify_container_contract
_impl._wait_job = _wait_job

for _name, _value in vars(_impl).items():
    if _name not in {"__name__", "__file__", "__spec__", "__loader__", "__package__"}:
        globals().setdefault(_name, _value)


if __name__ == "__main__":
    raise SystemExit(_impl.main())
