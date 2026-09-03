from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ollama_pdf_rag_test.py"


def _load():
    spec = importlib.util.spec_from_file_location("ollama_pdf_rag_test_entrypoint_test", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_ollama_smoke_default_does_not_inherit_dotenv_openai_embedding(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_EMBEDDING_MODEL", raising=False)
    with patch("subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = "NAME ID SIZE MODIFIED\nqwen3-embedding:8b abc 4.7GB now\n"
        mod = _load()
    assert mod._resolve_embedding_default() == "qwen3-embedding:8b"


def test_ollama_embedding_specific_env_has_priority(monkeypatch):
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:4b")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    mod = _load()
    assert mod._resolve_embedding_default() == "qwen3-embedding:4b"
