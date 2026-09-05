#!/usr/bin/env python3
"""Recursively upload PDFs below Ragbot's ./data directory for ingestion.

The host discovers files, but client filesystem paths never cross the API
boundary. PDF bytes are streamed to Ragbot's server-managed upload surface;
Source and durable Job state persist only ``ragbot-upload:///`` object URIs.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TMP_DIR = ROOT / "tmp"
STATE_FILE = TMP_DIR / "ragbot-runtime.json"
LOCAL_PID = TMP_DIR / "ragbot-local.pid"
MAX_BATCH_SIZE = 100
UPLOAD_ENDPOINT = "/ingest/upload/pdf"


class UserError(RuntimeError):
    """Expected user/setup error with an actionable message."""


def _read_runtime_state() -> dict:
    if not STATE_FILE.exists():
        raise UserError(
            "Ragbot runtime state was not found. Start Ragbot first with "
            "`python scripts/ragbot.py up --mode auto`."
        )
    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UserError(f"Could not read runtime state: {STATE_FILE}") from exc
    if not isinstance(raw, dict) or raw.get("mode") not in {"local", "docker"}:
        raise UserError("Ragbot runtime state does not contain a valid local/docker mode")
    return raw


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _local_runtime_running() -> bool:
    if not LOCAL_PID.exists():
        return False
    try:
        return _pid_alive(int(LOCAL_PID.read_text(encoding="utf-8").strip()))
    except (OSError, ValueError):
        return False


def _docker_stack_running() -> bool:
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--status", "running", "--services"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
    except OSError:
        return False
    if result.returncode != 0:
        return False
    services = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return "api" in services and "worker" in services


def _resolve_runtime_mode(state: dict) -> str:
    saved = str(state.get("mode") or "")
    docker_running = _docker_stack_running()
    local_running = _local_runtime_running()
    if saved == "docker" and docker_running:
        return "docker"
    if saved == "local" and local_running:
        return "local"
    if docker_running and not local_running:
        return "docker"
    if local_running and not docker_running:
        return "local"
    if docker_running and local_running and saved in {"local", "docker"}:
        return saved
    raise UserError(
        "Saved Ragbot runtime state is stale. Restart with "
        "`python scripts/ragbot.py up --mode auto`."
    )


def _resolve_root(value: str) -> Path:
    path = Path(value)
    path = ((ROOT / path) if not path.is_absolute() else path).resolve()
    data_root = DATA_DIR.resolve()
    try:
        path.relative_to(data_root)
    except ValueError as exc:
        raise UserError(f"PDF corpus must be below {DATA_DIR}") from exc
    if not path.is_dir():
        raise UserError(f"PDF corpus directory does not exist: {path}")
    return path


def _discover_pdfs(root: Path, *, recursive: bool = True) -> List[Path]:
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        (path.resolve() for path in iterator if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.as_posix().lower(),
    )


def _request_json(url: str, *, api_key: str | None = None, timeout: float = 60.0) -> dict:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        with urlopen(Request(url, headers=headers), timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except Exception as exc:
        raise UserError(f"Could not query Ragbot API: {url}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UserError(f"Expected JSON from {url}") from exc
    if not isinstance(data, dict):
        raise UserError(f"Expected JSON object from {url}")
    return data


def _runtime_restart_command(mode: str) -> str:
    resolved = mode if mode in {"local", "docker"} else "auto"
    return f"python3 scripts/ragbot.py restart --mode {resolved}"


def _assert_upload_transport_available(
    server: str,
    *,
    api_key: str | None,
    mode: str,
) -> None:
    """Fail before reading PDF bodies when the running API is from an older revision.

    Readiness alone proves that dependencies are healthy; it does not prove that
    a controller/CLI and a long-lived API container expose the same product
    contract. OpenAPI is the authoritative runtime surface and is available on
    both local and Docker FastAPI deployments.
    """
    openapi_url = f"{server.rstrip('/')}/openapi.json"
    try:
        spec = _request_json(openapi_url, api_key=api_key, timeout=15.0)
    except UserError as exc:
        raise UserError(
            "Ragbot is reachable but its API contract could not be verified before PDF upload. "
            f"Recreate the {mode} runtime with `{_runtime_restart_command(mode)}` and retry. "
            f"Contract probe error: {exc}"
        ) from exc

    paths = spec.get("paths")
    route = paths.get(UPLOAD_ENDPOINT) if isinstance(paths, dict) else None
    if not isinstance(route, dict) or "post" not in {str(method).lower() for method in route}:
        raise UserError(
            "Ragbot API is READY but does not expose POST /ingest/upload/pdf. "
            "The local controller/CLI and the reached API are from different revisions. "
            "This can be a stale image or a local API shadowing Docker's published port on macOS. "
            "Check `lsof -nP -iTCP:8000 -sTCP:LISTEN` (use your server port) for a conflicting Python API. "
            f"Rebuild and recreate it with `{_runtime_restart_command(mode)}`, then retry ingestion."
        )


def _upload_pdf(
    server: str,
    path: Path,
    *,
    tenant: str,
    api_key: str | None,
    tags: Sequence[str],
    chunk_size: int | None,
    chunk_overlap: int | None,
    timeout: float,
) -> dict:
    parsed = urlsplit(server)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UserError(f"Invalid Ragbot server URL: {server}")
    query: list[tuple[str, str]] = [("tenant_id", tenant), ("filename", path.name)]
    for tag in tags:
        query.append(("tag", tag))
    if chunk_size is not None:
        query.append(("chunk_size", str(chunk_size)))
    if chunk_overlap is not None:
        query.append(("chunk_overlap", str(chunk_overlap)))
    endpoint = f"{parsed.path.rstrip('/')}{UPLOAD_ENDPOINT}?{urlencode(query)}"
    connection_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_cls(parsed.hostname, parsed.port, timeout=timeout)
    size = path.stat().st_size
    try:
        connection.putrequest("POST", endpoint)
        connection.putheader("Content-Type", "application/pdf")
        connection.putheader("Accept", "application/json")
        connection.putheader("Content-Length", str(size))
        if api_key:
            connection.putheader("X-API-Key", api_key)
        connection.endheaders()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                connection.send(chunk)
        response = connection.getresponse()
        raw = response.read().decode("utf-8", errors="replace")
        if response.status == 404:
            raise UserError(
                "PDF upload endpoint disappeared after the preflight contract check. "
                "The API was likely replaced by a stale/different replica; recreate the runtime and retry."
            )
        if response.status >= 400:
            raise UserError(f"PDF upload failed: HTTP {response.status}: {raw[:1200]}")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise UserError("PDF upload returned a non-object JSON payload")
        return data
    finally:
        connection.close()


def _wait_job(
    server: str,
    job_id: str,
    *,
    api_key: str | None,
    timeout: float,
    poll_interval: float,
) -> dict:
    deadline = time.monotonic() + timeout
    previous = None
    while True:
        job = _request_json(
            f"{server.rstrip('/')}/ingest/jobs/{job_id}",
            api_key=api_key,
            timeout=min(60.0, timeout),
        )
        status = str(job.get("status") or "unknown")
        if status != previous:
            stats = job.get("stats") if isinstance(job.get("stats"), dict) else {}
            chunks = int(stats.get("chunks_total", job.get("chunk_count", 0)) or 0)
            print(f"Ingestion {job_id}: {status} (docs={job.get('doc_count', 0)}, chunks={chunks})")
            previous = status
        if status == "completed":
            return job
        if status in {"failed", "dead_lettered"}:
            raise UserError(
                f"Ingestion {job_id} {status}: {job.get('error') or 'unknown error'}; "
                f"source_config={json.dumps(job.get('source_config') or {}, ensure_ascii=False)}"
            )
        if time.monotonic() >= deadline:
            raise UserError(f"Timed out waiting for ingestion {job_id}")
        time.sleep(max(0.1, poll_interval))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload and ingest PDFs below Ragbot ./data")
    parser.add_argument("directory", nargs="?", default="data")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--user", default="cli-user")
    parser.add_argument("--server")
    parser.add_argument("--api-key")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--batch-size", type=int, default=MAX_BATCH_SIZE, help="Progress grouping only; uploads are streamed per file")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
            raise UserError(f"--batch-size must be between 1 and {MAX_BATCH_SIZE}")
        if args.max_files is not None and args.max_files <= 0:
            raise UserError("--max-files must be greater than zero")
        state = _read_runtime_state()
        mode = _resolve_runtime_mode(state)
        root = _resolve_root(args.directory)
        pdfs = _discover_pdfs(root, recursive=not args.no_recursive)
        if args.max_files is not None:
            pdfs = pdfs[: args.max_files]
        server = str(args.server or state.get("server") or "http://127.0.0.1:8000").rstrip("/")
        print(f"Runtime mode: {mode}")
        print(f"PDF root: {root}")
        print(f"Discovered PDFs: {len(pdfs)}")
        print("Ingestion transport: server-managed upload")
        if not pdfs:
            print("No PDF files found; nothing to ingest.")
            return 0
        if args.dry_run:
            for path in pdfs:
                print(path.relative_to(DATA_DIR.resolve()).as_posix())
            return 0

        _assert_upload_transport_available(
            server,
            api_key=args.api_key,
            mode=mode,
        )
        print("Upload API contract: compatible")

        failures = 0
        completed = 0
        for index, path in enumerate(pdfs, 1):
            print(f"Uploading {index}/{len(pdfs)}: {path.relative_to(DATA_DIR.resolve()).as_posix()}")
            try:
                submission = _upload_pdf(
                    server,
                    path,
                    tenant=args.tenant,
                    api_key=args.api_key,
                    tags=args.tag,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    timeout=args.timeout,
                )
                print(
                    f"Uploaded -> {submission.get('location')} -> "
                    f"job {submission.get('job_id')} ({submission.get('status')})"
                )
                if not args.no_wait and submission.get("job_id"):
                    _wait_job(
                        server,
                        str(submission["job_id"]),
                        api_key=args.api_key,
                        timeout=args.timeout,
                        poll_interval=args.poll_interval,
                    )
                completed += 1
            except Exception as exc:
                failures += 1
                print(f"ERROR: {path.name}: {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    return 1
        if failures:
            print(f"PDF ingestion finished with failures: {failures}/{len(pdfs)}", file=sys.stderr)
            return 1
        print(f"PDF ingestion complete: {completed} PDF(s) uploaded and submitted.")
        return 0
    except UserError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
