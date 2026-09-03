#!/usr/bin/env python3
"""Compatibility entry point for the Ragbot bootstrap controller.

The implementation lives in ``ragbot_impl.py``. This thin wrapper normalizes
unquoted ``ingest`` paths containing spaces before delegating to the original
controller, so both of these forms work on PowerShell/cmd/bash::

    python scripts/ragbot.py ingest "data/My Manual.pdf" --type pdf
    python scripts/ragbot.py ingest data/My Manual.pdf --type pdf

Quoting paths is still recommended for shell portability, but it is no longer
required for ordinary paths whose space-separated tokens appear before the
first ingest option.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Sequence

_IMPL_PATH = Path(__file__).with_name("ragbot_impl.py")
_SPEC = importlib.util.spec_from_file_location("ragbot_bootstrap_impl", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - installation corruption
    raise ImportError(f"Could not load Ragbot bootstrap implementation: {_IMPL_PATH}")
_impl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_impl)

# Preserve the historical module surface for tests and callers that import
# helpers directly from scripts/ragbot.py.
for _name in dir(_impl):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_impl, _name))


def _normalize_ingest_argv(argv: Sequence[str]) -> List[str]:
    """Join split ingest-location tokens while leaving all options untouched."""
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


def main(argv: Optional[List[str]] = None) -> int:
    raw = sys.argv[1:] if argv is None else argv
    return _impl.main(_normalize_ingest_argv(raw))


if __name__ == "__main__":
    raise SystemExit(main())
