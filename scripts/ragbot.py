#!/usr/bin/env python3
"""Cross-platform Ragbot bootstrap and operations helper.

This script intentionally uses only the Python standard library so it can repair
and provision a repository-local virtual environment before Ragbot itself is
installed.

Examples:
    python scripts/ragbot.py up --mode auto
    python scripts/ragbot.py ingest data/manuals --tenant engineering
    python scripts/ragbot.py search "ingestion worker lease" --tenant engineering
    python scripts/ragbot.py ask "Summarize the architecture" --tenant engineering
    python scripts/ragbot.py logs
    python scripts/ragbot.py down
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
from typing import Dict, Iterable, List, Optional

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


class UserError(RuntimeError):
    """Expected setup/runtime error with an actionable message."""


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
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


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
        key = key.strip()
        value = value.strip()
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
    py = _venv_python()

    if not py.exists():
        if not _python_version_ok(sys.executable):
            raise UserError("Ragbot requires Python >= 3.10")
        print(f"Creating virtual environment: {VENV}")
        _run([sys.executable, "-m", "venv", str(VENV)])

    pip_check = subprocess.run(
        [str(py), "-m", "pip", "--version"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if pip_check.returncode != 0:
        print("pip is missing from .venv; repairing it with ensurepip ...")
        _run([str(py), "-m", "ensurepip", "--upgrade"])

    dependency_probe = subprocess.run(
        [
            str(py),
            "-c",
            "import fastapi, uvicorn, httpx, requests, pydantic, cli.rag",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if force_install or dependency_probe.returncode != 0:
        _run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        target = f".[{extras}]" if extras else "."
        _run([str(py), "-m", "pip", "install", "-e", target])

    return py


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    compose = subprocess.run(
        ["docker", "compose", "version"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if compose.returncode != 0:
        return False
    daemon = subprocess.run(
        ["docker", "info"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return daemon.returncode == 0


def _resolve_mode(requested: str) -> str:
    if requested in {"local", "docker"}:
        return requested
    state = _read_state()
    if requested == "current" and state.get("mode") in {"local", "docker"}:
        return str(state["mode"])
    if requested == "current":
        return "docker" if _docker_available() else "local"
    if requested == "auto":
        return "docker" if _docker_available() else "local"
    raise UserError(f"Unsupported mode: {requested}")


def _write_state(mode: str, server: str = DEFAULT_SERVER, **extra: object) -> None:
    _ensure_dirs()
    payload = {"mode": mode, "server": server, "updated_at": time.time(), **extra}
    STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_state() -> Dict[str, object]:
    if not STATE_FILE.exists():
        return {}
    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _server_url(args: argparse.Namespace) -> str:
    if getattr(args, "server", None):
        return str(args.server).rstrip("/")
    state = _read_state()
    return str(state.get("server") or DEFAULT_SERVER).rstrip("/")


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
        if _health(server, "/admin/ready"):
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
    py = _ensure_venv(force_install=args.force_install, extras="worker")
    _copy_default_env()
    existing = _read_pid()
    server = f"http://{args.host}:{args.port}"
    if existing and _pid_alive(existing):
        print(f"Local Ragbot is already running (pid={existing})")
        _write_state("local", server, pid=existing)
        _wait_ready(server, timeout=args.timeout)
        return

    env = _local_env()
    command = [
        str(py),
        "-m",
        "uvicorn",
        "services.api.app.api:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.reload:
        command.append("--reload")

    _ensure_dirs()
    log_handle = LOCAL_LOG.open("a", encoding="utf-8")
    kwargs = {
        "cwd": ROOT,
        "env": env,
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "text": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    else:
        kwargs["start_new_session"] = True

    process = subprocess.Popen(command, **kwargs)
    log_handle.close()
    LOCAL_PID.write_text(str(process.pid), encoding="utf-8")
    _write_state("local", server, pid=process.pid)
    try:
        _wait_ready(server, timeout=args.timeout)
    except Exception:
        print(f"Startup log: {LOCAL_LOG}")
        _print_log_tail(80)
        raise

    if not env.get("EMBEDDING_API_KEY"):
        print(
            "WARNING: EMBEDDING_API_KEY is empty. Ragbot will use the development "
            "HashEmbedder fallback; configure a semantic embedding provider before "
            "evaluating Chinese/semantic retrieval quality."
        )
    print(f"Admin UI: {server}/admin/ui")
    print(f"Log file: {LOCAL_LOG}")


def _start_docker(args: argparse.Namespace) -> None:
    if not _docker_available():
        raise UserError(
            "Docker Compose is not available or the Docker daemon is not running. "
            "Use --mode local for the no-Docker development path."
        )
    _copy_default_env()
    _ensure_dirs()
    _ensure_venv(
        force_install=args.force_install,
        extras="postgres,qdrant,worker,s3,saas",
    )
    compose_env = _base_env()
    compose_env["RAGBOT_API_PORT"] = str(args.port)
    _run(["docker", "compose", "up", "-d", "--build"], env=compose_env)
    server = f"http://{args.host}:{args.port}"
    _write_state("docker", server)
    _wait_ready(server, timeout=max(args.timeout, 120.0))
    print(f"Admin UI: {server}/admin/ui")
    print("Data persistence: PostgreSQL + Qdrant Docker volumes")


def cmd_setup(args: argparse.Namespace) -> None:
    mode = _resolve_mode(args.mode)
    print(f"Setup mode: {mode}")
    _copy_default_env()
    _ensure_dirs()
    if mode == "docker":
        if not _docker_available():
            raise UserError("Docker is unavailable; rerun with --mode local")
        _ensure_venv(
            force_install=args.force_install,
            extras="postgres,qdrant,worker,s3,saas",
        )
    else:
        _ensure_venv(force_install=args.force_install, extras="worker")
    print("Setup complete.")
    print("Next: python scripts/ragbot.py up --mode " + mode)


def cmd_up(args: argparse.Namespace) -> None:
    mode = _resolve_mode(args.mode)
    print(f"Selected deployment mode: {mode}")
    if mode == "docker":
        _start_docker(args)
    else:
        _start_local(args)


def _stop_local() -> None:
    pid = _read_pid()
    if not pid:
        print("No local Ragbot PID file found.")
        return
    if _pid_alive(pid):
        print(f"Stopping local Ragbot (pid={pid}) ...")
        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and _pid_alive(pid):
                time.sleep(0.2)
            if _pid_alive(pid):
                os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    LOCAL_PID.unlink(missing_ok=True)
    print("Local Ragbot stopped.")


def cmd_down(args: argparse.Namespace) -> None:
    mode = _resolve_mode(args.mode)
    if mode == "docker":
        if shutil.which("docker"):
            command = ["docker", "compose", "down"]
            if args.volumes:
                command.append("-v")
            _run(command)
        else:
            raise UserError("docker executable not found")
    else:
        _stop_local()
    STATE_FILE.unlink(missing_ok=True)


def cmd_restart(args: argparse.Namespace) -> None:
    mode = _resolve_mode(args.mode)
    down_args = argparse.Namespace(mode=mode, volumes=False)
    cmd_down(down_args)
    args.mode = mode
    cmd_up(args)


def cmd_status(args: argparse.Namespace) -> None:
    server = _server_url(args)
    state = _read_state()
    mode = state.get("mode") or "unknown"
    print(f"mode: {mode}")
    print(f"server: {server}")
    print(f"liveness: {'OK' if _health(server, '/admin/health') else 'DOWN'}")
    print(f"readiness: {'READY' if _health(server, '/admin/ready') else 'NOT READY'}")
    if mode == "local":
        pid = _read_pid()
        print(f"pid: {pid or '-'}")
        print(f"log: {LOCAL_LOG}")


def _cli_command(args: argparse.Namespace, tail: List[str]) -> None:
    py = _ensure_venv(force_install=False, extras="worker")
    server = _server_url(args)
    command = [str(py), "-m", "cli.rag", "--server", server]
    if getattr(args, "api_key", None):
        command += ["--api-key", args.api_key]
    tenant = getattr(args, "tenant", None)
    if tenant:
        command += ["--tenant", tenant]
    user = getattr(args, "user", None)
    if user:
        command += ["--user", user]
    command += tail
    _run(command, env=_base_env())


def cmd_doctor(args: argparse.Namespace) -> None:
    _cli_command(args, ["doctor"])


def _is_remote_location(location: str) -> bool:
    lowered = location.lower()
    return lowered.startswith(
        (
            "http://",
            "https://",
            "s3://",
            "gdrive://",
            "notion://",
            "confluence://",
            "git@",
            "ssh://",
        )
    )


def _docker_location(location: str) -> str:
    if _is_remote_location(location):
        return location
    path = Path(location)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    data_root = DATA_DIR.resolve()
    try:
        relative = path.relative_to(data_root)
    except ValueError as exc:
        raise UserError(
            f"Docker local sources must be below {DATA_DIR}. "
            f"Move/copy the source under ./data first."
        ) from exc
    posix = relative.as_posix()
    return "/data" if posix == "." else f"/data/{posix}"


def _local_location(location: str) -> str:
    if _is_remote_location(location):
        return location
    path = Path(location)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    try:
        path.relative_to(DATA_DIR.resolve())
    except ValueError as exc:
        raise UserError(
            f"Local filesystem/PDF sources must be below {DATA_DIR}. "
            f"Move/copy the source under ./data first."
        ) from exc
    return str(path)


def cmd_ingest(args: argparse.Namespace) -> None:
    state = _read_state()
    mode = str(state.get("mode") or _resolve_mode("auto"))
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
    manifest = str(Path(args.manifest).resolve())
    tail = ["import", manifest]
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
    content = LOCAL_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in content[-lines:]:
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
    if not args.follow:
        return
    print(f"Following {LOCAL_LOG} (Ctrl+C to stop) ...")
    position = LOCAL_LOG.stat().st_size if LOCAL_LOG.exists() else 0
    try:
        while True:
            if LOCAL_LOG.exists():
                with LOCAL_LOG.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(position)
                    chunk = handle.read()
                    if chunk:
                        print(chunk, end="")
                        position = handle.tell()
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def _add_common_server_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--server", help=f"API server URL (default: runtime state or {DEFAULT_SERVER})")
    parser.add_argument("--api-key", help="Ragbot API key")
    parser.add_argument("--tenant", default="default", help="Tenant ID (default: default)")
    parser.add_argument("--user", default="cli-user", help="User ID (default: cli-user)")


def _add_start_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        choices=["auto", "local", "docker"],
        default="auto",
        help="auto prefers Docker when a healthy daemon is available, otherwise local",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload in local mode")
    parser.add_argument("--force-install", action="store_true", help="Reinstall Python dependencies")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/ragbot.py",
        description="Zero-friction Ragbot bootstrap, deployment and daily operations",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="Prepare .env, .venv and dependencies")
    setup.add_argument("--mode", choices=["auto", "local", "docker"], default="auto")
    setup.add_argument("--force-install", action="store_true")
    setup.set_defaults(func=cmd_setup)

    up = sub.add_parser("up", help="One-command install + start + readiness check")
    _add_start_args(up)
    up.set_defaults(func=cmd_up)

    down = sub.add_parser("down", help="Stop the active Ragbot deployment")
    down.add_argument("--mode", choices=["current", "local", "docker"], default="current")
    down.add_argument(
        "--volumes",
        action="store_true",
        help="Docker only: also remove Compose volumes (destructive)",
    )
    down.set_defaults(func=cmd_down)

    restart = sub.add_parser("restart", help="Restart the deployment")
    _add_start_args(restart)
    restart.set_defaults(func=cmd_restart)

    status = sub.add_parser("status", help="Show runtime mode and health")
    status.add_argument("--server")
    status.set_defaults(func=cmd_status)

    doctor = sub.add_parser("doctor", help="Run Ragbot's deployment doctor")
    _add_common_server_args(doctor)
    doctor.set_defaults(func=cmd_doctor)

    ingest = sub.add_parser("ingest", help="Ingest one local or remote source")
    _add_common_server_args(ingest)
    ingest.add_argument("location", help="Path/URL/source location")
    ingest.add_argument(
        "--type",
        choices=["local_fs", "repo", "pdf", "web", "s3", "gdrive", "notion", "confluence"],
    )
    ingest.add_argument("--name")
    ingest.add_argument("--tag", action="append", default=[])
    ingest.add_argument("--ref")
    ingest.add_argument("--chunk-size", type=int)
    ingest.add_argument("--chunk-overlap", type=int)
    ingest.add_argument("--no-wait", action="store_true")
    ingest.set_defaults(func=cmd_ingest)

    import_parser = sub.add_parser("import", help="Import a Ragbot JSON manifest")
    _add_common_server_args(import_parser)
    import_parser.add_argument("manifest")
    import_parser.add_argument("--no-wait", action="store_true")
    import_parser.set_defaults(func=cmd_import)

    search = sub.add_parser("search", help="Search indexed knowledge")
    _add_common_server_args(search)
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.set_defaults(func=cmd_search)

    ask = sub.add_parser("ask", help="Ask the Agentic RAG service")
    _add_common_server_args(ask)
    ask.add_argument("query")
    ask.set_defaults(func=cmd_ask)

    logs = sub.add_parser("logs", help="Show/follow local or Docker logs")
    logs.add_argument("--lines", type=int, default=100)
    logs.add_argument("--follow", "-f", action="store_true")
    logs.set_defaults(func=cmd_logs)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
