#!/usr/bin/env python3
"""Recursively ingest every PDF below Ragbot's ./data directory.

The helper batches discovered PDFs into Ragbot manifests (maximum 100 sources per
HTTP batch request) and delegates submission/waiting to scripts/ragbot.py.  It
therefore works with both the no-Docker local runtime and the Docker Compose
runtime without duplicating authentication, venv, or readiness logic.

Examples:
    python scripts/ingest_pdfs.py
    python scripts/ingest_pdfs.py data/manuals --tenant engineering
    python scripts/ingest_pdfs.py data --tenant engineering --tag manuals --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TMP_DIR = ROOT / "tmp"
STATE_FILE = TMP_DIR / "ragbot-runtime.json"
LOCAL_PID = TMP_DIR / "ragbot-local.pid"
CONTROLLER = ROOT / "scripts" / "ragbot.py"
MAX_BATCH_SIZE = 100


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
        raise UserError(
            "Ragbot runtime state does not contain a valid local/docker mode. "
            "Restart with `python scripts/ragbot.py up --mode auto`."
        )
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
        pid = int(LOCAL_PID.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    return _pid_alive(pid)


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
    """Resolve the live runtime instead of trusting a potentially stale state file."""
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
        "Saved Ragbot runtime state is stale: neither the recorded runtime nor a replacement "
        "runtime could be verified. Restart with `python scripts/ragbot.py up --mode auto`."
    )


def _resolve_root(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    data_root = DATA_DIR.resolve()
    try:
        path.relative_to(data_root)
    except ValueError as exc:
        raise UserError(
            f"PDF corpus must be below {DATA_DIR}. Move/copy the corpus under ./data first."
        ) from exc
    if not path.exists():
        raise UserError(f"PDF corpus path does not exist: {path}")
    if not path.is_dir():
        raise UserError(f"PDF corpus path must be a directory: {path}")
    return path


def _discover_pdfs(root: Path, *, recursive: bool = True) -> List[Path]:
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        (path.resolve() for path in iterator if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.as_posix().lower(),
    )


def _runtime_location(path: Path, mode: str) -> str:
    resolved = path.resolve()
    relative = resolved.relative_to(DATA_DIR.resolve())
    if mode == "docker":
        return f"/data/{relative.as_posix()}"
    return str(resolved)


def _docker_source_contract(path: Path) -> None:
    """Verify that the running worker sees the same PDF under the /data contract."""
    runtime_path = _runtime_location(path, "docker")
    probe = r'''
import json
import os
import sys
from pathlib import Path

requested = sys.argv[1]
resolved = Path(requested).expanduser().resolve()
roots = [
    Path(value).expanduser().resolve()
    for value in os.getenv("RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS", "").split(os.pathsep)
    if value.strip()
]

def within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False

allowed = not roots or any(within(resolved, root) for root in roots)
result = {
    "requested": requested,
    "resolved": str(resolved),
    "allowed_roots": [str(root) for root in roots],
    "allowed": allowed,
    "is_file": resolved.is_file(),
}
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if allowed and result["is_file"] else 3)
'''
    try:
        result = subprocess.run(
            ["docker", "compose", "exec", "-T", "worker", "python", "-c", probe, runtime_path],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
    except OSError as exc:
        raise UserError("Docker Compose is unavailable while the saved runtime is Docker") from exc
    if result.returncode == 0:
        return
    detail = (result.stdout or result.stderr or "worker probe failed").strip()
    raise UserError(
        "Docker local-source contract mismatch. The host discovered the PDF, but the running "
        f"worker cannot read the mapped container path {runtime_path!r} inside its configured "
        f"RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS. Worker probe: {detail}. "
        "Recreate the controller-managed stack with "
        "`python scripts/ragbot.py restart --mode docker`, then retry ingestion."
    )


def _batches(items: Sequence[Path], size: int) -> Iterable[Sequence[Path]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _source_spec(
    path: Path,
    *,
    mode: str,
    tags: Sequence[str],
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> dict:
    config = {}
    if chunk_size is not None:
        config["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        config["chunk_overlap"] = chunk_overlap
    return {
        "location": _runtime_location(path, mode),
        "source_type": "pdf",
        "name": path.relative_to(DATA_DIR.resolve()).as_posix(),
        "tags": list(tags),
        "config": config,
    }


def _write_manifest(
    paths: Sequence[Path],
    *,
    batch_index: int,
    tenant: str,
    mode: str,
    tags: Sequence[str],
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "tenant_id": tenant,
        "sources": [
            _source_spec(
                path,
                mode=mode,
                tags=tags,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for path in paths
        ],
    }
    output = TMP_DIR / f"ragbot-pdf-batch-{batch_index:04d}.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def _controller_command(manifest: Path, args: argparse.Namespace, state: dict) -> List[str]:
    command = [
        sys.executable,
        str(CONTROLLER),
        "import",
        str(manifest),
        "--tenant",
        args.tenant,
        "--user",
        args.user,
    ]
    server = args.server or state.get("server")
    if server:
        command += ["--server", str(server)]
    if args.api_key:
        command += ["--api-key", args.api_key]
    if args.no_wait:
        command.append("--no-wait")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recursively discover and ingest every PDF below Ragbot ./data"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="data",
        help="Corpus directory below ./data (default: data)",
    )
    parser.add_argument("--tenant", default="default", help="Tenant ID (default: default)")
    parser.add_argument("--user", default="cli-user", help="User ID (default: cli-user)")
    parser.add_argument("--server", help="Override the server URL saved by scripts/ragbot.py")
    parser.add_argument("--api-key", help="Ragbot API key")
    parser.add_argument("--tag", action="append", default=[], help="Tag applied to every PDF; repeatable")
    parser.add_argument("--chunk-size", type=int, help="Override PDF chunk size")
    parser.add_argument("--chunk-overlap", type=int, help="Override PDF chunk overlap")
    parser.add_argument("--no-recursive", action="store_true", help="Scan only the selected directory")
    parser.add_argument("--no-wait", action="store_true", help="Submit jobs without waiting for indexing")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=MAX_BATCH_SIZE,
        help=f"Sources per manifest submission (default/max: {MAX_BATCH_SIZE})",
    )
    parser.add_argument("--max-files", type=int, help="Limit discovered PDFs for staged testing")
    parser.add_argument("--dry-run", action="store_true", help="Print discovered PDFs without submitting")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later batches if one batch fails",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
            raise UserError(f"--batch-size must be between 1 and {MAX_BATCH_SIZE}")
        if args.max_files is not None and args.max_files <= 0:
            raise UserError("--max-files must be greater than zero")

        state = _read_runtime_state()
        saved_mode = str(state["mode"])
        mode = _resolve_runtime_mode(state)
        root = _resolve_root(args.directory)
        pdfs = _discover_pdfs(root, recursive=not args.no_recursive)
        if args.max_files is not None:
            pdfs = pdfs[: args.max_files]

        if mode == saved_mode:
            print(f"Runtime mode: {mode}")
        else:
            print(f"Runtime mode: {mode} (recovered from stale saved mode: {saved_mode})")
        print(f"PDF root: {root}")
        print(f"Discovered PDFs: {len(pdfs)}")
        if not pdfs:
            print("No PDF files found; nothing to ingest.")
            return 0

        if args.dry_run:
            for path in pdfs:
                print(path.relative_to(DATA_DIR.resolve()).as_posix())
            return 0

        if mode == "docker":
            _docker_source_contract(pdfs[0])

        batches = list(_batches(pdfs, args.batch_size))
        failed_batches = 0
        for index, batch in enumerate(batches, 1):
            manifest = _write_manifest(
                batch,
                batch_index=index,
                tenant=args.tenant,
                mode=mode,
                tags=args.tag,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            print(f"Batch {index}/{len(batches)}: {len(batch)} PDF(s)")
            command = _controller_command(manifest, args, state)
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode != 0:
                failed_batches += 1
                print(f"ERROR: batch {index} failed with exit code {result.returncode}", file=sys.stderr)
                if not args.continue_on_error:
                    return result.returncode or 1

        if failed_batches:
            print(
                f"PDF ingestion finished with failures: {failed_batches}/{len(batches)} batch(es) failed.",
                file=sys.stderr,
            )
            return 1

        print(f"PDF ingestion complete: {len(pdfs)} PDF(s) submitted across {len(batches)} batch(es).")
        return 0
    except UserError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
