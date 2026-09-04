#!/usr/bin/env python3
"""Ragbot bootstrap/deployment controller.

This file owns only repository setup and runtime lifecycle operations. Product
knowledge commands are delegated to the canonical ``cli.rag`` implementation.
It intentionally uses the standard library so ``setup`` can repair/create the
repository-local virtual environment before Ragbot is installed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
TMP_DIR = ROOT / "tmp"
LOCAL_LOG = LOG_DIR / "ragbot-local.log"
LOCAL_PID = TMP_DIR / "ragbot-local.pid"
STATE_FILE = TMP_DIR / "ragbot-runtime.json"
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"
DEFAULT_SERVER = "http://127.0.0.1:8000"
_PDF_INGEST_PATH = Path(__file__).with_name("ingest_pdfs.py")
_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".csv", ".log"}
_EXCLUDED_DIRS = {
    ".git", "node_modules", "__pycache__", ".mypy_cache", ".venv", "venv",
    "dist", "build", ".tox", ".eggs",
}


class UserError(RuntimeError):
    pass


def _run(
    command: Iterable[str],
    *,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    cmd = [str(part) for part in command]
    print("+", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        check=check,
        text=True,
        capture_output=capture,
    )


def _venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def _copy_default_env() -> None:
    if ENV_FILE.exists():
        return
    if not ENV_EXAMPLE.exists():
        raise UserError(".env.example is missing")
    shutil.copyfile(ENV_EXAMPLE, ENV_FILE)
    print("Created .env from .env.example")
    print("Edit .env to configure LLM/embedding credentials for semantic RAG.")


def _parse_dotenv(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _base_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.update(_parse_dotenv(ENV_FILE))
    return env


def _local_env() -> Dict[str, str]:
    env = _base_env()
    env["RAGBOT_ENV"] = "development"
    env["RAGBOT_INGESTION_MODE"] = "inline"
    env.pop("POSTGRES_DSN", None)
    env.pop("QDRANT_URL", None)
    env["RAGBOT_DATA_DIR"] = str(DATA_DIR)
    env["RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS"] = str(DATA_DIR.resolve())
    return env


def _python_version_ok(executable: str) -> bool:
    result = subprocess.run(
        [executable, "-c", "import sys; print(int(sys.version_info >= (3, 10)))"],
        text=True,
        capture_output=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "1"


def _ensure_venv(force_install: bool = False, extras: str = "worker") -> Path:
    _ensure_dirs()
    python = _venv_python()
    if not python.exists():
        if not _python_version_ok(sys.executable):
            raise UserError("Ragbot requires Python >= 3.10")
        print(f"Creating virtual environment: {VENV}")
        _run([sys.executable, "-m", "venv", str(VENV)])

    pip_check = subprocess.run(
        [str(python), "-m", "pip", "--version"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if pip_check.returncode != 0:
        _run([str(python), "-m", "ensurepip", "--upgrade"])

    probe = subprocess.run(
        [str(python), "-c", "import fastapi, uvicorn, httpx, requests, pydantic, cli.rag"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if force_install or probe.returncode != 0:
        _run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        target = f".[{extras}]" if extras else "."
        _run([str(python), "-m", "pip", "install", "-e", target])
    return python


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    compose = subprocess.run(["docker", "compose", "version"], cwd=ROOT, text=True, capture_output=True)
    if compose.returncode != 0:
        return False
    daemon = subprocess.run(["docker", "info"], cwd=ROOT, text=True, capture_output=True)
    return daemon.returncode == 0


def _read_state() -> Dict[str, object]:
    if not STATE_FILE.exists():
        return {}
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_state(mode: str, server: str = DEFAULT_SERVER, **extra: object) -> None:
    _ensure_dirs()
    STATE_FILE.write_text(
        json.dumps({"mode": mode, "server": server, "updated_at": time.time(), **extra}, indent=2),
        encoding="utf-8",
    )


def _resolve_mode(requested: str) -> str:
    if requested in {"local", "docker"}:
        return requested
    if requested == "current":
        current = _read_state().get("mode")
        if current in {"local", "docker"}:
            return str(current)
    if requested in {"current", "auto"}:
        return "docker" if _docker_available() else "local"
    raise UserError(f"Unsupported mode: {requested}")


def _server_url(args: argparse.Namespace) -> str:
    explicit = getattr(args, "server", None)
    if explicit:
        return str(explicit).rstrip("/")
    return str(_read_state().get("server") or DEFAULT_SERVER).rstrip("/")


def _health(url: str, path: str = "/admin/ready", timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as response:
            if response.status != 200:
                return False
            data = json.loads(response.read().decode("utf-8"))
            expected = "ready" if path.endswith("/ready") else "ok"
            return data.get("status") == expected
    except Exception:
        return False


def _wait_ready(server: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _health(server):
            print(f"Ragbot is READY: {server}")
            return
        time.sleep(1)
    raise UserError(f"Ragbot did not become ready within {timeout:.0f}s")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _read_pid() -> Optional[int]:
    if not LOCAL_PID.exists():
        return None
    try:
        return int(LOCAL_PID.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _start_local(args: argparse.Namespace) -> None:
    python = _ensure_venv(force_install=args.force_install, extras="worker")
    _copy_default_env()
    server = f"http://{args.host}:{args.port}"
    existing = _read_pid()
    if existing and _pid_alive(existing):
        _write_state("local", server, pid=existing)
        _wait_ready(server, timeout=args.timeout)
        return

    _ensure_dirs()
    log_handle = LOCAL_LOG.open("a", encoding="utf-8")
    command = [str(python), "-m", "uvicorn", "services.api.app.api:app", "--host", args.host, "--port", str(args.port)]
    if args.reload:
        command.append("--reload")
    kwargs = {
        "cwd": ROOT,
        "env": _local_env(),
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "text": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **kwargs)
    log_handle.close()
    LOCAL_PID.write_text(str(process.pid), encoding="utf-8")
    _write_state("local", server, pid=process.pid)
    _wait_ready(server, timeout=args.timeout)
    print(f"Admin UI: {server}/admin/ui")
    print(f"Log file: {LOCAL_LOG}")


def _start_docker(args: argparse.Namespace) -> None:
    if not _docker_available():
        raise UserError("Docker Compose is unavailable; use --mode local")
    _copy_default_env()
    _ensure_dirs()
    _ensure_venv(force_install=args.force_install, extras="postgres,qdrant,worker,s3,saas")
    env = _base_env()
    env["RAGBOT_API_PORT"] = str(args.port)
    _run(["docker", "compose", "up", "-d", "--build"], env=env)
    server = f"http://{args.host}:{args.port}"
    _write_state("docker", server)
    _wait_ready(server, timeout=max(args.timeout, 120.0))
    print(f"Admin UI: {server}/admin/ui")


def _stop_local() -> None:
    pid = _read_pid()
    if not pid:
        print("No local Ragbot PID file found.")
        return
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and _pid_alive(pid):
                time.sleep(0.2)
            if _pid_alive(pid) and hasattr(signal, "SIGKILL"):
                os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    LOCAL_PID.unlink(missing_ok=True)


def _is_remote_location(location: str) -> bool:
    return location.lower().startswith(("http://", "https://", "s3://", "gdrive://", "notion://", "confluence://", "git@", "ssh://"))


def _docker_location(location: str) -> str:
    if _is_remote_location(location):
        return location
    path = Path(location)
    path = ((ROOT / path) if not path.is_absolute() else path).resolve()
    try:
        relative = path.relative_to(DATA_DIR.resolve())
    except ValueError as exc:
        raise UserError(f"Docker local sources must be below {DATA_DIR}") from exc
    posix = relative.as_posix()
    return "/data" if posix == "." else f"/data/{posix}"


def _local_location(location: str) -> str:
    if _is_remote_location(location):
        return location
    path = Path(location)
    path = ((ROOT / path) if not path.is_absolute() else path).resolve()
    try:
        path.relative_to(DATA_DIR.resolve())
    except ValueError as exc:
        raise UserError(f"Local filesystem/PDF sources must be below {DATA_DIR}") from exc
    return str(path)


def _cli_command(args: argparse.Namespace, tail: List[str]) -> None:
    python = _ensure_venv(force_install=False, extras="worker")
    command = [str(python), "-m", "cli.rag", "--server", _server_url(args)]
    if getattr(args, "api_key", None):
        command += ["--api-key", args.api_key]
    if getattr(args, "tenant", None):
        command += ["--tenant", args.tenant]
    if getattr(args, "user", None):
        command += ["--user", args.user]
    command += tail
    _run(command, env=_base_env())


def cmd_setup(args: argparse.Namespace) -> None:
    mode = _resolve_mode(args.mode)
    _copy_default_env()
    _ensure_dirs()
    extras = "postgres,qdrant,worker,s3,saas" if mode == "docker" else "worker"
    _ensure_venv(force_install=args.force_install, extras=extras)
    print(f"Setup complete ({mode}).")


def cmd_up(args: argparse.Namespace) -> None:
    mode = _resolve_mode(args.mode)
    (_start_docker if mode == "docker" else _start_local)(args)


def cmd_down(args: argparse.Namespace) -> None:
    mode = _resolve_mode(args.mode)
    if mode == "docker":
        command = ["docker", "compose", "down"]
        if args.volumes:
            command.append("-v")
        _run(command)
    else:
        _stop_local()
    STATE_FILE.unlink(missing_ok=True)


def cmd_restart(args: argparse.Namespace) -> None:
    mode = _resolve_mode(args.mode)
    cmd_down(argparse.Namespace(mode=mode, volumes=False))
    args.mode = mode
    cmd_up(args)


def cmd_status(args: argparse.Namespace) -> None:
    server = _server_url(args)
    state = _read_state()
    print(f"mode: {state.get('mode') or 'unknown'}")
    print(f"server: {server}")
    print(f"liveness: {'OK' if _health(server, '/admin/health') else 'DOWN'}")
    print(f"readiness: {'READY' if _health(server, '/admin/ready') else 'NOT READY'}")


def cmd_doctor(args: argparse.Namespace) -> None:
    _cli_command(args, ["doctor"])


def cmd_ingest(args: argparse.Namespace) -> None:
    mode = str(_read_state().get("mode") or _resolve_mode("auto"))
    location = _docker_location(args.location) if mode == "docker" else _local_location(args.location)
    tail = ["ingest", location]
    if args.type:
        tail += ["--type", args.type]
    if args.name:
        tail += ["--name", args.name]
    for tag in args.tag:
        tail += ["--tag", tag]
    if args.ref:
        tail += ["--ref", args.ref]
    if args.chunk_size:
        tail += ["--chunk-size", str(args.chunk_size)]
    if args.chunk_overlap is not None:
        tail += ["--chunk-overlap", str(args.chunk_overlap)]
    if not args.no_wait:
        tail.append("--wait")
    _cli_command(args, tail)


def cmd_import(args: argparse.Namespace) -> None:
    tail = ["import", str(Path(args.manifest).resolve())]
    if not args.no_wait:
        tail.append("--wait")
    _cli_command(args, tail)


def cmd_search(args: argparse.Namespace) -> None:
    _cli_command(args, ["search", args.query, "--top-k", str(args.top_k)])


def cmd_ask(args: argparse.Namespace) -> None:
    _cli_command(args, ["ask", args.query])


def _print_log_tail(lines: int) -> None:
    if not LOCAL_LOG.exists():
        print(f"No local log file found: {LOCAL_LOG}")
        return
    for line in LOCAL_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]:
        print(line)


def cmd_logs(args: argparse.Namespace) -> None:
    mode = str(_read_state().get("mode") or _resolve_mode("auto"))
    if mode == "docker":
        command = ["docker", "compose", "logs", "--tail", str(args.lines)]
        if args.follow:
            command.append("-f")
        _run(command)
        return
    _print_log_tail(args.lines)


def _add_common_server_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--server")
    parser.add_argument("--api-key")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--user", default="cli-user")


def _add_start_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["auto", "local", "docker"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--force-install", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python scripts/ragbot.py", description="Ragbot bootstrap and runtime controller")
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup")
    setup.add_argument("--mode", choices=["auto", "local", "docker"], default="auto")
    setup.add_argument("--force-install", action="store_true")
    setup.set_defaults(func=cmd_setup)

    up = sub.add_parser("up")
    _add_start_args(up)
    up.set_defaults(func=cmd_up)

    down = sub.add_parser("down")
    down.add_argument("--mode", choices=["current", "local", "docker"], default="current")
    down.add_argument("--volumes", action="store_true")
    down.set_defaults(func=cmd_down)

    restart = sub.add_parser("restart")
    _add_start_args(restart)
    restart.set_defaults(func=cmd_restart)

    status = sub.add_parser("status")
    status.add_argument("--server")
    status.set_defaults(func=cmd_status)

    doctor = sub.add_parser("doctor")
    _add_common_server_args(doctor)
    doctor.set_defaults(func=cmd_doctor)

    ingest = sub.add_parser("ingest")
    _add_common_server_args(ingest)
    ingest.add_argument("location")
    ingest.add_argument("--type", choices=["local_fs", "repo", "pdf", "web", "s3", "gdrive", "notion", "confluence"])
    ingest.add_argument("--name")
    ingest.add_argument("--tag", action="append", default=[])
    ingest.add_argument("--ref")
    ingest.add_argument("--chunk-size", type=int)
    ingest.add_argument("--chunk-overlap", type=int)
    ingest.add_argument("--no-wait", action="store_true")
    ingest.set_defaults(func=cmd_ingest)

    manifest = sub.add_parser("import")
    _add_common_server_args(manifest)
    manifest.add_argument("manifest")
    manifest.add_argument("--no-wait", action="store_true")
    manifest.set_defaults(func=cmd_import)

    search = sub.add_parser("search")
    _add_common_server_args(search)
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.set_defaults(func=cmd_search)

    ask = sub.add_parser("ask")
    _add_common_server_args(ask)
    ask.add_argument("query")
    ask.set_defaults(func=cmd_ask)

    logs = sub.add_parser("logs")
    logs.add_argument("--lines", type=int, default=100)
    logs.add_argument("--follow", "-f", action="store_true")
    logs.set_defaults(func=cmd_logs)
    return parser


def _normalize_ingest_argv(argv: Sequence[str]) -> List[str]:
    tokens = list(argv)
    try:
        index = tokens.index("ingest")
    except ValueError:
        return tokens
    start = index + 1
    if start >= len(tokens):
        return tokens
    end = start
    while end < len(tokens) and not tokens[end].startswith("--"):
        end += 1
    if end - start > 1:
        tokens[start:end] = [" ".join(tokens[start:end])]
    return tokens


def _has_option(tokens: Sequence[str], name: str) -> bool:
    return any(token == name or token.startswith(name + "=") for token in tokens)


def _option_value(tokens: Sequence[str], name: str) -> Optional[str]:
    for index, token in enumerate(tokens):
        if token == name:
            return tokens[index + 1] if index + 1 < len(tokens) else None
        if token.startswith(name + "="):
            return token[len(name) + 1 :]
    return None


def _option_values(tokens: Sequence[str], name: str) -> List[str]:
    values = []
    for index, token in enumerate(tokens):
        if token == name and index + 1 < len(tokens):
            values.append(tokens[index + 1])
        elif token.startswith(name + "="):
            values.append(token[len(name) + 1 :])
    return values


def _resolved_directory(location: str) -> Optional[Path]:
    if _is_remote_location(location):
        return None
    path = Path(location)
    path = ((ROOT / path) if not path.is_absolute() else path).resolve()
    return path if path.is_dir() else None


def _directory_inventory(directory: Path) -> tuple[int, int]:
    pdfs = text_files = 0
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(directory).parts[:-1]
        if any(part in _EXCLUDED_DIRS for part in relative_parts):
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pdfs += 1
        elif suffix in _TEXT_EXTENSIONS:
            text_files += 1
    return pdfs, text_files


def _pdf_directory_command(tokens: Sequence[str], directory: Path) -> List[str]:
    command = [sys.executable, str(_PDF_INGEST_PATH), str(directory)]
    for option in ("--tenant", "--user", "--server", "--api-key", "--chunk-size", "--chunk-overlap"):
        value = _option_value(tokens, option)
        if value is not None:
            command.extend([option, value])
    for tag in _option_values(tokens, "--tag"):
        command.extend(["--tag", tag])
    if _has_option(tokens, "--no-wait"):
        command.append("--no-wait")
    return command


class _BootstrapCore:
    """Compatibility seam for tests; this is not a second product CLI."""

    @staticmethod
    def main(argv: List[str]) -> int:
        return _controller_main(argv)


_impl = _BootstrapCore()


def _smart_directory_ingest(tokens: Sequence[str]) -> Optional[int]:
    if "ingest" not in tokens or _has_option(tokens, "--type"):
        return None
    index = list(tokens).index("ingest")
    if index + 1 >= len(tokens):
        return None
    directory = _resolved_directory(tokens[index + 1])
    if directory is None:
        return None
    pdf_count, text_count = _directory_inventory(directory)
    if pdf_count == 0:
        return None
    if text_count:
        result = _impl.main(list(tokens))
        if int(result or 0) != 0:
            return int(result)
    if not _PDF_INGEST_PATH.exists():
        raise RuntimeError(f"PDF corpus helper is missing: {_PDF_INGEST_PATH}")
    completed = subprocess.run(_pdf_directory_command(tokens, directory), cwd=ROOT)
    return int(completed.returncode)


def _controller_main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
        return 0
    except UserError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: command failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1
    except KeyboardInterrupt:
        return 130


def main(argv: Optional[List[str]] = None) -> int:
    raw = sys.argv[1:] if argv is None else argv
    normalized = _normalize_ingest_argv(raw)
    smart = _smart_directory_ingest(normalized)
    if smart is not None:
        return smart
    return _controller_main(normalized)


if __name__ == "__main__":
    raise SystemExit(main())
