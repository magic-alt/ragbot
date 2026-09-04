#!/usr/bin/env python3
"""Compatibility entry point for the Ragbot bootstrap controller.

The implementation lives in ``ragbot_impl.py``. This thin wrapper normalizes
unquoted ``ingest`` paths containing spaces and adds smart local-directory
routing before delegating to the original controller.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from cli.job_wait import wait_for_job

_IMPL_PATH = Path(__file__).with_name("ragbot_impl.py")
_PDF_INGEST_PATH = Path(__file__).with_name("ingest_pdfs.py")
_SPEC = importlib.util.spec_from_file_location("ragbot_bootstrap_impl", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - installation corruption
    raise ImportError(f"Could not load Ragbot bootstrap implementation: {_IMPL_PATH}")
_impl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_impl)

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_impl, _name))

_SYNCED_GLOBALS = (
    "ROOT",
    "VENV",
    "DATA_DIR",
    "LOG_DIR",
    "TMP_DIR",
    "LOCAL_LOG",
    "LOCAL_PID",
    "STATE_FILE",
    "ENV_FILE",
    "ENV_EXAMPLE",
    "DEFAULT_SERVER",
)

_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".csv", ".log"}
_EXCLUDED_DIRS = {
    ".git", "node_modules", "__pycache__", ".mypy_cache", ".venv", "venv",
    "dist", "build", ".tox", ".eggs",
}


def _sync_impl_state() -> None:
    for name in _SYNCED_GLOBALS:
        if name in globals():
            setattr(_impl, name, globals()[name])


def _wait_for_job(
    server: str,
    job_id: str,
    *,
    headers: Dict[str, str],
    timeout: float,
    poll_interval: float,
    quiet: bool = False,
) -> Dict[str, Any]:
    return wait_for_job(
        _impl._api_request,
        server,
        job_id,
        headers=headers,
        timeout=timeout,
        poll_interval=poll_interval,
        quiet=quiet,
    )


_impl._wait_for_job = _wait_for_job


def _base_env():
    _sync_impl_state()
    return _impl._base_env()


def _local_env():
    _sync_impl_state()
    return _impl._local_env()


def _copy_default_env():
    _sync_impl_state()
    return _impl._copy_default_env()


def _ensure_dirs():
    _sync_impl_state()
    return _impl._ensure_dirs()


def _docker_location(location: str) -> str:
    _sync_impl_state()
    return _impl._docker_location(location)


def _local_location(location: str) -> str:
    _sync_impl_state()
    return _impl._local_location(location)


def _normalize_ingest_argv(argv: Sequence[str]) -> List[str]:
    tokens = list(argv)
    try:
        ingest_index = tokens.index("ingest")
    except ValueError:
        return tokens
    start = ingest_index + 1
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
            if index + 1 < len(tokens):
                return tokens[index + 1]
            return None
        prefix = name + "="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _option_values(tokens: Sequence[str], name: str) -> List[str]:
    values: List[str] = []
    for index, token in enumerate(tokens):
        if token == name and index + 1 < len(tokens):
            values.append(tokens[index + 1])
        else:
            prefix = name + "="
            if token.startswith(prefix):
                values.append(token[len(prefix) :])
    return values


def _resolved_directory(location: str) -> Optional[Path]:
    if _impl._is_remote_location(location):
        return None
    path = Path(location)
    if not path.is_absolute():
        path = Path(ROOT) / path
    try:
        resolved = path.resolve()
    except OSError:
        return None
    return resolved if resolved.is_dir() else None


def _directory_inventory(directory: Path) -> tuple[int, int]:
    pdfs = 0
    text_files = 0
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        try:
            relative_parts = path.relative_to(directory).parts[:-1]
        except ValueError:  # pragma: no cover
            relative_parts = path.parts[:-1]
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


def _smart_directory_ingest(tokens: Sequence[str]) -> Optional[int]:
    if "ingest" not in tokens or _has_option(tokens, "--type"):
        return None
    ingest_index = list(tokens).index("ingest")
    if ingest_index + 1 >= len(tokens):
        return None

    directory = _resolved_directory(tokens[ingest_index + 1])
    if directory is None:
        return None
    pdf_count, text_count = _directory_inventory(directory)
    if pdf_count == 0:
        return None

    print(
        f"Directory corpus detected: {directory} "
        f"({pdf_count} PDF(s), {text_count} text/local_fs file(s))"
    )

    if text_count:
        print("Ingesting text/local_fs files first, then recursively ingesting PDFs ...")
        result = _impl.main(list(tokens))
        code = int(result or 0)
        if code != 0:
            return code
    else:
        print(
            "PDF-only directory detected; skipping local_fs so the command does not "
            "report a misleading completed job with 0 documents."
        )

    if not _PDF_INGEST_PATH.exists():
        raise RuntimeError(f"PDF corpus helper is missing: {_PDF_INGEST_PATH}")
    command = _pdf_directory_command(tokens, directory)
    print("+", " ".join(str(part) for part in command))
    completed = subprocess.run(command, cwd=ROOT)
    return int(completed.returncode)


def main(argv: Optional[List[str]] = None) -> int:
    raw = sys.argv[1:] if argv is None else argv
    _sync_impl_state()
    normalized = _normalize_ingest_argv(raw)
    smart_result = _smart_directory_ingest(normalized)
    if smart_result is not None:
        return smart_result
    return _impl.main(normalized)


if __name__ == "__main__":
    raise SystemExit(main())
