from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ragbot.py"
SPEC = importlib.util.spec_from_file_location("ragbot_smart_directory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def test_directory_inventory_counts_pdf_and_text_and_skips_excluded_dirs(tmp_path: Path) -> None:
    corpus = tmp_path / "data"
    (corpus / "nested").mkdir(parents=True)
    (corpus / ".git").mkdir()
    (corpus / "book.pdf").write_bytes(b"pdf")
    (corpus / "nested" / "manual.PDF").write_bytes(b"pdf")
    (corpus / "notes.md").write_text("notes", encoding="utf-8")
    (corpus / "ignored.bin").write_bytes(b"bin")
    (corpus / ".git" / "hidden.pdf").write_bytes(b"pdf")

    assert bootstrap._directory_inventory(corpus) == (2, 1)


def test_pdf_directory_command_preserves_common_ingest_options(tmp_path: Path) -> None:
    corpus = tmp_path / "data"
    corpus.mkdir()
    tokens = [
        "ingest",
        str(corpus),
        "--tenant",
        "engineering",
        "--user",
        "kaermax",
        "--server",
        "http://127.0.0.1:8000",
        "--tag",
        "pdf",
        "--tag",
        "manuals",
        "--chunk-size",
        "900",
        "--chunk-overlap",
        "120",
        "--no-wait",
    ]

    command = bootstrap._pdf_directory_command(tokens, corpus)

    assert command[0] == bootstrap.sys.executable
    assert command[1].endswith("ingest_pdfs.py")
    assert command[2] == str(corpus)
    assert command.count("--tag") == 2
    assert "engineering" in command
    assert "kaermax" in command
    assert "900" in command
    assert "120" in command
    assert "--no-wait" in command


def test_pdf_only_directory_bypasses_zero_document_local_fs_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "data"
    corpus.mkdir()
    (corpus / "DeepSeek in Action.pdf").write_bytes(b"pdf")
    helper = tmp_path / "scripts" / "ingest_pdfs.py"
    helper.parent.mkdir()
    helper.write_text("# test helper", encoding="utf-8")

    impl_calls: list[list[str]] = []
    commands: list[list[str]] = []

    monkeypatch.setattr(bootstrap, "_PDF_INGEST_PATH", helper)
    monkeypatch.setattr(
        bootstrap._impl,
        "main",
        lambda argv: impl_calls.append(list(argv)) or 0,
    )

    def fake_run(command, **kwargs):
        commands.append([str(item) for item in command])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    result = bootstrap._smart_directory_ingest(
        ["ingest", str(corpus), "--tenant", "engineering", "--tag", "pdf"]
    )

    assert result == 0
    assert impl_calls == []
    assert len(commands) == 1
    assert str(corpus) in commands[0]
    assert "engineering" in commands[0]
    assert "pdf" in commands[0]


def test_mixed_directory_ingests_text_then_pdfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "data"
    corpus.mkdir()
    (corpus / "notes.md").write_text("hello", encoding="utf-8")
    (corpus / "manual.pdf").write_bytes(b"pdf")
    helper = tmp_path / "scripts" / "ingest_pdfs.py"
    helper.parent.mkdir()
    helper.write_text("# test helper", encoding="utf-8")

    impl_calls: list[list[str]] = []
    commands: list[list[str]] = []
    monkeypatch.setattr(bootstrap, "_PDF_INGEST_PATH", helper)
    monkeypatch.setattr(
        bootstrap._impl,
        "main",
        lambda argv: impl_calls.append(list(argv)) or 0,
    )

    def fake_run(command, **kwargs):
        commands.append([str(item) for item in command])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    tokens = ["ingest", str(corpus), "--tenant", "engineering"]
    result = bootstrap._smart_directory_ingest(tokens)

    assert result == 0
    assert impl_calls == [tokens]
    assert len(commands) == 1


def test_explicit_source_type_disables_smart_directory_routing(tmp_path: Path) -> None:
    corpus = tmp_path / "data"
    corpus.mkdir()
    (corpus / "manual.pdf").write_bytes(b"pdf")

    assert (
        bootstrap._smart_directory_ingest(
            ["ingest", str(corpus), "--type", "local_fs", "--tenant", "engineering"]
        )
        is None
    )
