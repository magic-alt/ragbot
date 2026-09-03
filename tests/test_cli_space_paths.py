from __future__ import annotations

import importlib.util
from pathlib import Path

from cli import rag as cli_rag


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ragbot.py"
SPEC = importlib.util.spec_from_file_location("ragbot_bootstrap_wrapper", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def test_product_cli_joins_unquoted_ingest_path_tokens() -> None:
    argv = [
        "--server",
        "http://127.0.0.1:8000",
        "--tenant",
        "engineering",
        "ingest",
        r".\data\DeepSeek",
        "in",
        "Action",
        "LLM",
        "Deployment.pdf.pdf",
        "--type",
        "pdf",
        "--wait",
    ]

    normalized = cli_rag._normalize_ingest_argv(argv)

    assert normalized == [
        "--server",
        "http://127.0.0.1:8000",
        "--tenant",
        "engineering",
        "ingest",
        r".\data\DeepSeek in Action LLM Deployment.pdf.pdf",
        "--type",
        "pdf",
        "--wait",
    ]


def test_bootstrap_cli_joins_unquoted_ingest_path_tokens() -> None:
    argv = [
        "ingest",
        r".\data\DeepSeek",
        "in",
        "Action",
        "LLM",
        "Deployment",
        "Fine-Tuning.pdf.pdf",
        "--tenant",
        "engineering",
        "--type",
        "pdf",
    ]

    normalized = bootstrap._normalize_ingest_argv(argv)

    assert normalized == [
        "ingest",
        r".\data\DeepSeek in Action LLM Deployment Fine-Tuning.pdf.pdf",
        "--tenant",
        "engineering",
        "--type",
        "pdf",
    ]


def test_quoted_or_single_token_location_is_unchanged() -> None:
    argv = ["ingest", r".\data\My Manual.pdf", "--type", "pdf"]
    assert cli_rag._normalize_ingest_argv(argv) == argv
    assert bootstrap._normalize_ingest_argv(argv) == argv


def test_non_ingest_commands_are_unchanged() -> None:
    argv = ["search", "motor control", "--top-k", "5"]
    assert cli_rag._normalize_ingest_argv(argv) == argv
    assert bootstrap._normalize_ingest_argv(argv) == argv
