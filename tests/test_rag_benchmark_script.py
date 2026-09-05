from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rag_benchmark.py"
SPEC = importlib.util.spec_from_file_location("rag_benchmark_entrypoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_level3_runtime_uses_current_python_when_frameworks_exist(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(benchmark, "_python_has_frameworks", lambda executable: True)

    def unexpected(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr(benchmark.subprocess, "run", unexpected)
    assert benchmark._ensure_level3_runtime(["--level", "3"], bootstrap=True) is None


def test_level3_runtime_reexecutes_with_ready_repo_venv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    current = tmp_path / "system-python"
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.touch()
    monkeypatch.setattr(benchmark.sys, "executable", str(current))
    monkeypatch.setattr(benchmark, "_venv_python", lambda: venv_python)
    monkeypatch.setattr(
        benchmark,
        "_python_has_frameworks",
        lambda executable: Path(executable) == venv_python,
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    result = benchmark._ensure_level3_runtime(["--level", "3"], bootstrap=True)
    assert result == 17
    assert calls == [[str(venv_python), str(SCRIPT), "--level", "3"]]


def test_level3_runtime_bootstraps_expected_extras(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    current = tmp_path / "system-python"
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.touch()
    monkeypatch.setattr(benchmark.sys, "executable", str(current))
    monkeypatch.setattr(benchmark, "_venv_python", lambda: venv_python)
    probes = iter([False, False, True])
    monkeypatch.setattr(benchmark, "_python_has_frameworks", lambda executable: next(probes))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:4] == [str(venv_python), "-m", "pip", "install"]:
            return SimpleNamespace(returncode=0)
        if command and command[0] == str(venv_python):
            return SimpleNamespace(returncode=0)
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    result = benchmark._ensure_level3_runtime(["--level", "3"], bootstrap=True)
    assert result == 0
    assert [str(venv_python), "-m", "pip", "install", "-e", ".[worker,benchmark-frameworks]"] in calls


def test_level3_runtime_can_disable_bootstrap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(benchmark.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(benchmark, "_venv_python", lambda: tmp_path / ".venv" / "bin" / "python")
    monkeypatch.setattr(benchmark, "_python_has_frameworks", lambda executable: False)
    with pytest.raises(RuntimeError, match="native framework dependencies are missing"):
        benchmark._ensure_level3_runtime(["--level", "3"], bootstrap=False)
