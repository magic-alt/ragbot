from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ingest_pdfs.py"
SPEC = importlib.util.spec_from_file_location("ragbot_ingest_pdfs", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_discover_pdfs_is_recursive_and_case_insensitive(tmp_path: Path) -> None:
    root = tmp_path / "data"
    nested = root / "manuals" / "nested"
    nested.mkdir(parents=True)
    (root / "a.pdf").write_bytes(b"pdf")
    (nested / "b.PDF").write_bytes(b"pdf")
    (nested / "ignore.md").write_text("text", encoding="utf-8")

    found = MODULE._discover_pdfs(root, recursive=True)

    assert [path.name for path in found] == ["a.pdf", "b.PDF"]


def test_discover_pdfs_can_be_non_recursive(tmp_path: Path) -> None:
    root = tmp_path / "data"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "top.pdf").write_bytes(b"pdf")
    (nested / "deep.pdf").write_bytes(b"pdf")

    found = MODULE._discover_pdfs(root, recursive=False)

    assert [path.name for path in found] == ["top.pdf"]


def test_runtime_location_maps_docker_data_mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    pdf = data / "manuals" / "guide.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"pdf")
    monkeypatch.setattr(MODULE, "DATA_DIR", data)

    assert MODULE._runtime_location(pdf, "docker") == "/data/manuals/guide.pdf"
    assert MODULE._runtime_location(pdf, "local") == str(pdf.resolve())


def test_source_location_is_executor_independent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    pdf = data / "manuals" / "Deep Seek 入门.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"pdf")
    monkeypatch.setattr(MODULE, "DATA_DIR", data)

    assert MODULE._source_location(pdf) == (
        "ragbot-data:///manuals/Deep%20Seek%20%E5%85%A5%E9%97%A8.pdf"
    )


def test_resolve_runtime_mode_recovers_stale_saved_docker_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "_docker_stack_running", lambda: False)
    monkeypatch.setattr(MODULE, "_local_runtime_running", lambda: True)

    assert MODULE._resolve_runtime_mode({"mode": "docker"}) == "local"


def test_resolve_runtime_mode_prefers_live_docker_when_saved_local_is_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "_docker_stack_running", lambda: True)
    monkeypatch.setattr(MODULE, "_local_runtime_running", lambda: False)

    assert MODULE._resolve_runtime_mode({"mode": "local"}) == "docker"


def test_docker_postgres_host_port_reads_compose_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="127.0.0.1:49173\n",
            stderr="",
        ),
    )

    assert MODULE._docker_postgres_host_port() == 49173


def test_competing_host_python_client_blocks_docker_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "_docker_postgres_host_port", lambda: 49173)
    monkeypatch.setattr(
        MODULE,
        "_host_postgres_python_clients",
        lambda _port: ["Python 4242 kaermax 12u IPv4 TCP 127.0.0.1:60000->127.0.0.1:49173"],
    )

    with pytest.raises(MODULE.UserError, match="Competing host Python process"):
        MODULE._assert_no_competing_host_worker()


def test_docker_source_contract_uses_production_validator_with_portable_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "Users" / "kaermax" / "ragbot" / "data"
    pdf = data / "manuals" / "guide.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"pdf")
    monkeypatch.setattr(MODULE, "DATA_DIR", data)
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    captured = []

    def fake_run(command, **kwargs):
        captured.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"checks": [{"allowed": true, "is_file": true}]}', stderr="")

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    MODULE._docker_source_contract(pdf)

    command, kwargs = captured[0]
    assert command[:5] == ["docker", "compose", "exec", "-T", "worker"]
    assert command[-1] == "ragbot-data:///manuals/guide.pdf"
    assert "validate_local_source_path" in command[-2]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["capture_output"] is True


def test_docker_source_contract_probes_every_pdf_in_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    first = data / "a.pdf"
    second = data / "nested" / "b.pdf"
    second.parent.mkdir(parents=True)
    first.write_bytes(b"pdf")
    second.write_bytes(b"pdf")
    monkeypatch.setattr(MODULE, "DATA_DIR", data)
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    captured = []

    def fake_run(command, **kwargs):
        captured.append(command)
        return SimpleNamespace(returncode=0, stdout='{"checks": []}', stderr="")

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    MODULE._docker_source_contract([first, second])

    assert captured[0][-2:] == ["ragbot-data:///a.pdf", "ragbot-data:///nested/b.pdf"]


def test_docker_source_contract_fails_early_with_recreate_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    pdf = data / "guide.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"pdf")
    monkeypatch.setattr(MODULE, "DATA_DIR", data)
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)

    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=3,
            stdout='{"checks": [{"allowed": false, "allowed_roots": ["/old-data"], "is_file": false}]}',
            stderr="",
        ),
    )

    with pytest.raises(MODULE.UserError, match="restart --mode docker"):
        MODULE._docker_source_contract(pdf)


def test_source_spec_builds_portable_pdf_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    pdf = data / "docs" / "guide.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"pdf")
    monkeypatch.setattr(MODULE, "DATA_DIR", data)

    spec = MODULE._source_spec(
        pdf,
        mode="docker",
        tags=["manuals", "pdf"],
        chunk_size=900,
        chunk_overlap=120,
    )

    assert spec == {
        "location": "ragbot-data:///docs/guide.pdf",
        "source_type": "pdf",
        "name": "docs/guide.pdf",
        "tags": ["manuals", "pdf"],
        "config": {"chunk_size": 900, "chunk_overlap": 120},
    }


def test_source_spec_is_identical_in_local_and_docker_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    pdf = data / "guide.pdf"
    data.mkdir()
    pdf.write_bytes(b"pdf")
    monkeypatch.setattr(MODULE, "DATA_DIR", data)

    local = MODULE._source_spec(
        pdf,
        mode="local",
        tags=[],
        chunk_size=None,
        chunk_overlap=None,
    )
    docker = MODULE._source_spec(
        pdf,
        mode="docker",
        tags=[],
        chunk_size=None,
        chunk_overlap=None,
    )

    assert local["location"] == docker["location"] == "ragbot-data:///guide.pdf"


def test_batches_preserve_every_pdf() -> None:
    values = [Path(f"{index}.pdf") for index in range(205)]

    batches = list(MODULE._batches(values, 100))

    assert [len(batch) for batch in batches] == [100, 100, 5]
    assert [item for batch in batches for item in batch] == values
