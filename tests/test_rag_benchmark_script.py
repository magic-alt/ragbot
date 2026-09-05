from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rag_benchmark.py"
SPEC = importlib.util.spec_from_file_location("rag_benchmark_entrypoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_entrypoint_import_is_stdlib_only_before_bootstrap(monkeypatch: pytest.MonkeyPatch):
    """A bare system Python must load the entrypoint before requests/frameworks exist."""
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "requests" or name.startswith("benchmarks") or name.startswith("services"):
            raise AssertionError(f"project dependency imported too early: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec = importlib.util.spec_from_file_location("rag_benchmark_stdlib_bootstrap_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module._ensure_runtime)


def test_runtime_uses_current_python_when_no_repo_venv_and_dependencies_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        benchmark,
        "_venv_python",
        lambda: tmp_path / "missing-venv" / "bin" / "python",
    )
    monkeypatch.setattr(
        benchmark,
        "_python_has_runtime",
        lambda executable, *, require_frameworks: True,
    )

    def unexpected(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr(benchmark.subprocess, "run", unexpected)
    assert benchmark._ensure_runtime(["--level", "3"], require_frameworks=True, bootstrap=True) is None


def test_runtime_prefers_ready_repo_venv_even_when_current_python_is_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    current = tmp_path / "system-python"
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.touch()
    monkeypatch.setattr(benchmark.sys, "executable", str(current))
    monkeypatch.setattr(benchmark, "_venv_python", lambda: venv_python)
    monkeypatch.setattr(
        benchmark,
        "_python_has_runtime",
        lambda executable, *, require_frameworks: True,
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=23)

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    result = benchmark._ensure_runtime(["--level", "3"], require_frameworks=True, bootstrap=True)
    assert result == 23
    assert calls == [[str(venv_python), str(SCRIPT), "--level", "3"]]


def test_runtime_reexecutes_with_ready_repo_venv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    current = tmp_path / "system-python"
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.touch()
    monkeypatch.setattr(benchmark.sys, "executable", str(current))
    monkeypatch.setattr(benchmark, "_venv_python", lambda: venv_python)
    monkeypatch.setattr(
        benchmark,
        "_python_has_runtime",
        lambda executable, *, require_frameworks: Path(executable) == venv_python,
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    result = benchmark._ensure_runtime(["--level", "3"], require_frameworks=True, bootstrap=True)
    assert result == 17
    assert calls == [[str(venv_python), str(SCRIPT), "--level", "3"]]


def test_runtime_bootstraps_expected_level3_extras(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    current = tmp_path / "system-python"
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.touch()
    monkeypatch.setattr(benchmark.sys, "executable", str(current))
    monkeypatch.setattr(benchmark, "_venv_python", lambda: venv_python)
    probes = iter([False, True])
    monkeypatch.setattr(
        benchmark,
        "_python_has_runtime",
        lambda executable, *, require_frameworks: next(probes),
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:4] == [str(venv_python), "-m", "pip", "install"]:
            return SimpleNamespace(returncode=0)
        if command and command[0] == str(venv_python):
            return SimpleNamespace(returncode=0)
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    result = benchmark._ensure_runtime(["--level", "3"], require_frameworks=True, bootstrap=True)
    assert result == 0
    assert [str(venv_python), "-m", "pip", "install", "-e", ".[worker,benchmark-frameworks]"] in calls


def test_level2_runtime_bootstraps_base_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    current = tmp_path / "system-python"
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.touch()
    monkeypatch.setattr(benchmark.sys, "executable", str(current))
    monkeypatch.setattr(benchmark, "_venv_python", lambda: venv_python)
    probes = iter([False, True])
    monkeypatch.setattr(
        benchmark,
        "_python_has_runtime",
        lambda executable, *, require_frameworks: next(probes),
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:4] == [str(venv_python), "-m", "pip", "install"]:
            return SimpleNamespace(returncode=0)
        if command and command[0] == str(venv_python):
            return SimpleNamespace(returncode=0)
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    result = benchmark._ensure_runtime(["--level", "2"], require_frameworks=False, bootstrap=True)
    assert result == 0
    assert [str(venv_python), "-m", "pip", "install", "-e", ".[worker]"] in calls


def test_runtime_can_disable_bootstrap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(benchmark.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(benchmark, "_venv_python", lambda: tmp_path / ".venv" / "bin" / "python")
    monkeypatch.setattr(
        benchmark,
        "_python_has_runtime",
        lambda executable, *, require_frameworks: False,
    )
    with pytest.raises(RuntimeError, match="native framework dependencies are missing"):
        benchmark._ensure_runtime(["--level", "3"], require_frameworks=True, bootstrap=False)


def test_corpus_source_accepts_single_pdf_and_strips_trailing_space(tmp_path: Path):
    pdf = tmp_path / "Building Agent-Powered Applications.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    resolved = benchmark._resolve_corpus_source(f"{pdf} ")
    assert resolved == pdf.resolve()
    assert resolved.is_file()


def test_corpus_source_accepts_directory(tmp_path: Path):
    assert benchmark._resolve_corpus_source(str(tmp_path)) == tmp_path.resolve()


def test_corpus_source_rejects_unsupported_file(tmp_path: Path):
    unsupported = tmp_path / "notes.csv"
    unsupported.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported corpus file"):
        benchmark._resolve_corpus_source(str(unsupported))


def test_parser_keeps_corpus_dir_as_backward_compatible_alias():
    parser = benchmark.build_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "golden.json",
            "--corpus-dir",
            "book.pdf",
        ]
    )
    assert args.corpus_dir == "book.pdf"
