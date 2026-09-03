#!/usr/bin/env python3
"""Stable entrypoint for the Ollama PDF RAG smoke-test implementation."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

SCRIPT_DIR = Path(__file__).resolve().parent
IMPL = SCRIPT_DIR / "ollama_pdf_rag_test_impl.py"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "qwen3-embedding:0.6b"


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


_impl = _load_impl()
for _name, _value in vars(_impl).items():
    if _name not in {"__name__", "__file__", "__spec__", "__loader__", "__package__"}:
        globals().setdefault(_name, _value)


if __name__ == "__main__":
    raise SystemExit(_impl.main())
