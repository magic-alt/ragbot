from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_ragbot.py"
SPEC = importlib.util.spec_from_file_location("verify_ragbot_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
verify_ragbot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_ragbot
SPEC.loader.exec_module(verify_ragbot)


def test_pytest_auto_skips_without_mutating_environment(monkeypatch):
    monkeypatch.setattr(verify_ragbot, "_existing_pytest_python", lambda: None)

    executable, result = verify_ragbot._resolve_pytest_python("auto")

    assert executable is None
    assert result["status"] == "skipped"
    assert "--pytest on" in result["reason"]


def test_pytest_on_bootstraps_repo_virtualenv(monkeypatch):
    expected = Path("/tmp/ragbot-test-venv/bin/python")
    monkeypatch.setattr(verify_ragbot, "_existing_pytest_python", lambda: None)
    monkeypatch.setattr(
        verify_ragbot,
        "_bootstrap_functional_test_env",
        lambda: (expected, None),
    )

    executable, result = verify_ragbot._resolve_pytest_python("on")

    assert executable == expected
    assert result == {
        "status": "ready",
        "python": str(expected),
        "bootstrapped": True,
    }


def test_bootstrap_installs_same_functional_extras_as_ci(tmp_path, monkeypatch):
    venv = tmp_path / ".venv"
    venv_python = venv / "bin" / "python"
    commands = []

    monkeypatch.setattr(verify_ragbot, "VENV", venv)
    monkeypatch.setattr(verify_ragbot, "_venv_python", lambda: venv_python)
    monkeypatch.setattr(verify_ragbot, "_python_has_pytest", lambda _python: True)

    def fake_run(command, **kwargs):
        commands.append([str(item) for item in command])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(verify_ragbot.subprocess, "run", fake_run)

    executable, error = verify_ragbot._bootstrap_functional_test_env()

    assert error is None
    assert executable == venv_python
    assert commands[0] == [sys.executable, "-m", "venv", str(venv)]
    assert commands[1][:5] == [
        str(venv_python),
        "-m",
        "pip",
        "install",
        "-e",
    ]
    assert commands[1][5] == f".[{verify_ragbot.FUNCTIONAL_EXTRAS}]"


def test_run_pytest_uses_resolved_repository_python(monkeypatch):
    executable = Path("/tmp/ragbot/.venv/bin/python")
    calls = []
    monkeypatch.setattr(
        verify_ragbot,
        "_resolve_pytest_python",
        lambda _mode: (
            executable,
            {"status": "ready", "python": str(executable), "bootstrapped": False},
        ),
    )

    def fake_run(command, **kwargs):
        calls.append([str(item) for item in command])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(verify_ragbot.subprocess, "run", fake_run)

    result = verify_ragbot._run_pytest("on")

    assert calls == [[str(executable), "-m", "pytest", "-q"]]
    assert result["status"] == "passed"
    assert result["python"] == str(executable)
