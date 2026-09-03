from __future__ import annotations

import importlib.util
from pathlib import Path

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
    assert env["RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS"] == str(data.resolve())


def test_docker_location_maps_repo_data_to_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    manuals = data / "manuals"
    manuals.mkdir(parents=True)
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    monkeypatch.setattr(bootstrap, "DATA_DIR", data)

    assert bootstrap._docker_location("data/manuals") == "/data/manuals"


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
