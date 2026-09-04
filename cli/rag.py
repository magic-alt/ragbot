"""Compatibility entry point for the Ragbot product CLI.

The implementation lives in :mod:`cli.rag_impl`. This wrapper accepts unquoted
``rag ingest`` locations containing spaces by joining the positional tokens
between ``ingest`` and the first ``--option`` before delegating to the original
CLI implementation.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence

from . import rag_impl as _impl
from .job_wait import wait_for_job

# Preserve the existing public and private helper surface used by tests and
# downstream callers importing from cli.rag.
for _name in dir(_impl):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_impl, _name))


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


# cmd_ingest/cmd_import are function objects defined in rag_impl and resolve
# globals in that module. Replace its waiter as well as the wrapper export.
_impl._wait_for_job = _wait_for_job


def _normalize_ingest_argv(argv: Sequence[str]) -> List[str]:
    """Join split ingest path/URL tokens while preserving CLI options."""
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


def main(argv: Optional[List[str]] = None) -> None:
    raw = sys.argv[1:] if argv is None else argv
    return _impl.main(_normalize_ingest_argv(raw))


if __name__ == "__main__":
    main()
