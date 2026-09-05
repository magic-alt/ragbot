from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ragbot.py"
SPEC = importlib.util.spec_from_file_location("ragbot_bootstrap", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def test_parse_dotenv_handles_comments_and_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nFOO=bar\nDOUBLE=\"two words\"\nSINGLE='value'\nEMPTY=\n",
        encoding="utf-8",
    )

    assert bootstrap._parse_dotenv(env_file) == {
        "FOO": "bar",
        "DOUBLE": "two words",
        "SINGLE": "value",
        "EMPTY": "",
    }


def test_local_env_forces_inline_and_removes_durable_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "POSTGRES_DSN=postgresql://example\nQDRANT_URL=http://example:6333\nRAGBOT_ENV=production\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(bootstrap, "ENV_FILE", env_file)
    monkeypatch.setattr(bootstrap, "DATA_DIR", data)

    env = bootstrap._local_env()

    assert env["RAGBOT_ENV"] == "development"
    assert env["RAGBOT_INGESTION_MODE"] == "inline"
    assert "POSTGRES_DSN" not in env
    assert "QDRANT_URL" not in env
    assert env["RAGBOT_DATA_DIR"] == str(data.resolve())
    assert env["RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS"] == str(data.resolve())


def test_docker_start_forces_controller_repo_data_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text("RAGBOT_DATA_DIR=/tmp/stale-corpus\n", encoding="utf-8")
    captured: list[dict[str, str]] = []

    monkeypatch.setattr(bootstrap, "DATA_DIR", data)
    monkeypatch.setattr(bootstrap, "ENV_FILE", env_file)
    monkeypatch.setattr(bootstrap, "_docker_available", lambda: True)
    monkeypatch.setattr(bootstrap, "_copy_default_env", lambda: None)
    monkeypatch.setattr(bootstrap, "_ensure_dirs", lambda: None)
    monkeypatch.setattr(bootstrap, "_ensure_venv", lambda **_kwargs: tmp_path / "python")
    monkeypatch.setattr(bootstrap, "_wait_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bootstrap, "_write_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bootstrap, "_stop_local_before_docker", lambda: None)
    monkeypatch.setattr(bootstrap, "_assert_docker_runtime", lambda *_args, **_kwargs: {"boot_id": "test"})

    def fake_run(command, *, env=None, **_kwargs):
        assert command[:3] == ["docker", "compose", "up"]
        captured.append(dict(env or {}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bootstrap, "_run", fake_run)
    bootstrap._start_docker(
        SimpleNamespace(force_install=False, port=8000, host="127.0.0.1", timeout=1.0)
    )

    assert captured
    assert captured[0]["RAGBOT_DATA_DIR"] == str(data.resolve())


def test_docker_switch_stops_verified_local_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "_read_pid", lambda: 1234)
    monkeypatch.setattr(bootstrap, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(bootstrap.os, "name", "posix")
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=0, stdout="python -m uvicorn services.api.app.api:app --port 8000",
    ))
    stopped = []
    monkeypatch.setattr(bootstrap, "_stop_local", lambda: stopped.append(True))
    bootstrap._stop_local_before_docker()
    assert stopped == [True]


def test_docker_switch_does_not_kill_reused_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "_read_pid", lambda: 1234)
    monkeypatch.setattr(bootstrap, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(bootstrap.os, "name", "posix")
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=0, stdout="python unrelated_application.py",
    ))
    monkeypatch.setattr(bootstrap, "_stop_local", lambda: pytest.fail("must not stop unrelated process"))
    with pytest.raises(bootstrap.UserError, match="identity could not be verified"):
        bootstrap._stop_local_before_docker()


@pytest.mark.parametrize("host_boot_id", ["container-boot", "old-local-boot", None])
def test_docker_routing_checks_process_identity(
    monkeypatch: pytest.MonkeyPatch, host_boot_id: str | None,
) -> None:
    identity = {"service": "ragbot-api", "boot_id": "container-boot",
                "capabilities": ["server-managed-pdf-upload"]}
    monkeypatch.setattr(bootstrap, "_run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps(identity),
    ))
    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(
        json.dumps({**identity, "boot_id": host_boot_id}).encode(),
    ))
    if host_boot_id == "container-boot":
        assert bootstrap._assert_docker_runtime("http://127.0.0.1:8000", env={}) == identity
    else:
        with pytest.raises(bootstrap.UserError, match="local API or another stack"):
            bootstrap._assert_docker_runtime("http://127.0.0.1:8000", env={})


def test_docker_routing_rejects_legacy_host_without_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "_run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps({"boot_id": "container-boot"}),
    ))
    def missing(*_args, **_kwargs):
        raise OSError("HTTP 404")
    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", missing)
    with pytest.raises(bootstrap.UserError, match="routing verification failed"):
        bootstrap._assert_docker_runtime("http://127.0.0.1:8000", env={})


@pytest.mark.parametrize("routing_matches", [True, False])
def test_docker_start_records_success_only_after_routing_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, routing_matches: bool,
) -> None:
    events = []
    monkeypatch.setattr(bootstrap, "_docker_available", lambda: True)
    monkeypatch.setattr(bootstrap, "_copy_default_env", lambda: None)
    monkeypatch.setattr(bootstrap, "_ensure_dirs", lambda: None)
    monkeypatch.setattr(bootstrap, "_ensure_venv", lambda **_kwargs: tmp_path / "python")
    monkeypatch.setattr(bootstrap, "_base_env", lambda: {})
    monkeypatch.setattr(bootstrap, "_stop_local_before_docker", lambda: events.append("stop_local"))
    monkeypatch.setattr(bootstrap, "_run", lambda *_args, **_kwargs: events.append("compose_up"))
    def ready(*_args, **kwargs):
        assert kwargs["announce"] is False
        events.append("ready")
    def verify(*_args, **_kwargs):
        events.append("verify")
        if not routing_matches:
            raise bootstrap.UserError("routing mismatch")
        return {"boot_id": "container-boot"}
    def save(mode, server, **kwargs):
        assert kwargs["boot_id"] == "container-boot"
        events.append("save")
    monkeypatch.setattr(bootstrap, "_wait_ready", ready)
    monkeypatch.setattr(bootstrap, "_assert_docker_runtime", verify)
    monkeypatch.setattr(bootstrap, "_write_state", save)
    args = SimpleNamespace(force_install=False, port=8000, host="127.0.0.1", timeout=1.0)
    if routing_matches:
        bootstrap._start_docker(args)
        assert events == ["stop_local", "compose_up", "ready", "verify", "save"]
        assert "Ragbot is READY" in capsys.readouterr().out
    else:
        with pytest.raises(bootstrap.UserError, match="routing mismatch"):
            bootstrap._start_docker(args)
        assert events == ["stop_local", "compose_up", "ready", "verify"]
        assert "Ragbot is READY" not in capsys.readouterr().out


def test_runtime_mode_recovers_from_stale_docker_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bootstrap, "_read_state", lambda: {"mode": "docker"})
    monkeypatch.setattr(bootstrap, "_read_pid", lambda: 1234)
    monkeypatch.setattr(bootstrap, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(bootstrap, "_docker_stack_running", lambda: False)

    assert bootstrap._runtime_mode() == "local"


def test_docker_location_maps_repo_data_to_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    manuals = data / "manuals"
    manuals.mkdir(parents=True)
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    monkeypatch.setattr(bootstrap, "DATA_DIR", data)

    assert bootstrap._docker_location("data/manuals") == "/data/manuals"
    assert bootstrap._docker_location("data") == "/data"


def test_docker_location_rejects_paths_outside_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    monkeypatch.setattr(bootstrap, "DATA_DIR", data)

    with pytest.raises(bootstrap.UserError):
        bootstrap._docker_location(str(outside))


def test_remote_location_is_not_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    monkeypatch.setattr(bootstrap, "DATA_DIR", tmp_path / "data")

    url = "https://example.com/manual.pdf"
    assert bootstrap._docker_location(url) == url
