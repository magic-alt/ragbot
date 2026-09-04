from __future__ import annotations

from pathlib import Path

import pytest

from services.worker.connectors.managed_data import (
    canonical_managed_data_uri,
    managed_data_uri,
    resolve_local_source_reference,
    resolve_managed_data_uri,
)
from services.worker.connectors.security import validate_local_source_path


def test_managed_data_uri_resolves_against_executor_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    pdf = data / "manuals" / "Deep Seek 入门.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"pdf")
    monkeypatch.setenv("RAGBOT_DATA_DIR", str(data))
    monkeypatch.setenv("RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS", str(data))

    uri = managed_data_uri("manuals/Deep Seek 入门.pdf")

    assert uri == "ragbot-data:///manuals/Deep%20Seek%20%E5%85%A5%E9%97%A8.pdf"
    assert canonical_managed_data_uri(uri) == uri
    assert resolve_managed_data_uri(uri) == str(pdf.resolve())
    assert validate_local_source_path(uri) == str(pdf.resolve())


def test_legacy_docker_data_path_maps_to_host_managed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    pdf = data / "guide.pdf"
    data.mkdir()
    pdf.write_bytes(b"pdf")

    # This mirrors the common local .env combination. /data is the historical
    # Docker alias, while RAGBOT_DATA_DIR identifies the executor's real root.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAGBOT_DATA_DIR", "./data")
    monkeypatch.setenv("RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS", "/data")
    monkeypatch.setenv("RAGBOT_ENV", "development")

    assert resolve_local_source_reference("/data/guide.pdf") == str(pdf.resolve())
    assert validate_local_source_path("/data/guide.pdf") == str(pdf.resolve())


def test_managed_data_uri_keeps_allowlist_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    other = tmp_path / "other"
    data.mkdir()
    other.mkdir()
    (data / "guide.pdf").write_bytes(b"pdf")
    monkeypatch.setenv("RAGBOT_DATA_DIR", str(data))
    monkeypatch.setenv("RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS", str(other))

    with pytest.raises(ValueError, match="outside RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS"):
        validate_local_source_path("ragbot-data:///guide.pdf")


def test_managed_data_uri_rejects_parent_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAGBOT_DATA_DIR", str(tmp_path / "data"))

    with pytest.raises(ValueError, match="below the Ragbot data root"):
        resolve_managed_data_uri("ragbot-data:///../secret.pdf")
    with pytest.raises(ValueError, match="below the Ragbot data root"):
        resolve_managed_data_uri("ragbot-data:///%2E%2E/secret.pdf")


def test_managed_data_root_uri_resolves_to_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("RAGBOT_DATA_DIR", str(data))

    assert managed_data_uri("") == "ragbot-data:///"
    assert resolve_managed_data_uri("ragbot-data:///") == str(data.resolve())
