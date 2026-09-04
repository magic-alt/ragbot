from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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


def test_resolve_runtime_mode_recovers_stale_saved_docker_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "_docker_stack_running", lambda: False)
    monkeypatch.setattr(MODULE, "_local_runtime_running", lambda: True)

    assert MODULE._resolve_runtime_mode({"mode": "docker"}) == "local"


def test_resolve_runtime_mode_prefers_live_docker_when_saved_local_is_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "_docker_stack_running", lambda: True)
    monkeypatch.setattr(MODULE, "_local_runtime_running", lambda: False)

    assert MODULE._resolve_runtime_mode({"mode": "local"}) == "docker"


def test_upload_transport_contract_accepts_current_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MODULE,
        "_request_json",
        lambda *_args, **_kwargs: {
            "openapi": "3.1.0",
            "paths": {"/ingest/upload/pdf": {"post": {}}},
        },
    )

    MODULE._assert_upload_transport_available(
        "http://127.0.0.1:8000",
        api_key=None,
        mode="docker",
    )


def test_upload_transport_contract_rejects_stale_ready_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MODULE,
        "_request_json",
        lambda *_args, **_kwargs: {
            "openapi": "3.1.0",
            "paths": {"/ingest/quick": {"post": {}}},
        },
    )

    with pytest.raises(MODULE.UserError, match="different revisions") as exc_info:
        MODULE._assert_upload_transport_available(
            "http://127.0.0.1:8000",
            api_key=None,
            mode="docker",
        )

    assert "restart --mode docker" in str(exc_info.value)


def test_upload_transport_contract_surfaces_probe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args, **_kwargs):
        raise MODULE.UserError("HTTP 503")

    monkeypatch.setattr(MODULE, "_request_json", fail)

    with pytest.raises(MODULE.UserError, match="could not be verified") as exc_info:
        MODULE._assert_upload_transport_available(
            "http://127.0.0.1:8000",
            api_key=None,
            mode="local",
        )

    assert "restart --mode local" in str(exc_info.value)


class _FakeResponse:
    status = 202

    def read(self):
        return json.dumps(
            {
                "status": "accepted",
                "job_id": "job-1",
                "location": "ragbot-upload:///0123456789abcdef0123456789abcdef",
            }
        ).encode()


class _FakeConnection:
    instances = []

    def __init__(self, host, port=None, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.method = None
        self.endpoint = None
        self.headers = {}
        self.sent = bytearray()
        self.__class__.instances.append(self)

    def putrequest(self, method, endpoint):
        self.method = method
        self.endpoint = endpoint

    def putheader(self, key, value):
        self.headers[key] = value

    def endheaders(self):
        pass

    def send(self, chunk):
        self.sent.extend(chunk)

    def getresponse(self):
        return _FakeResponse()

    def close(self):
        pass


def test_upload_pdf_streams_bytes_without_persisting_client_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "Deep Seek 入门.pdf"
    pdf.write_bytes(b"%PDF-test-body")
    _FakeConnection.instances.clear()
    monkeypatch.setattr(MODULE.http.client, "HTTPConnection", _FakeConnection)

    result = MODULE._upload_pdf(
        "http://127.0.0.1:8000",
        pdf,
        tenant="engineering",
        api_key="secret",
        tags=["manuals"],
        chunk_size=900,
        chunk_overlap=120,
        timeout=30,
    )

    conn = _FakeConnection.instances[0]
    assert conn.method == "POST"
    assert conn.endpoint.startswith("/ingest/upload/pdf?")
    assert "tenant_id=engineering" in conn.endpoint
    assert "filename=Deep+Seek+%E5%85%A5%E9%97%A8.pdf" in conn.endpoint
    assert "tag=manuals" in conn.endpoint
    assert conn.headers["Content-Type"] == "application/pdf"
    assert conn.headers["X-API-Key"] == "secret"
    assert bytes(conn.sent) == b"%PDF-test-body"
    assert str(pdf.resolve()) not in conn.endpoint
    assert result["location"].startswith("ragbot-upload:///")


def test_wait_job_surfaces_durable_source_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MODULE,
        "_request_json",
        lambda *_args, **_kwargs: {
            "status": "failed",
            "error": "parser failed",
            "source_config": {"path": "ragbot-upload:///0123456789abcdef0123456789abcdef"},
        },
    )

    with pytest.raises(MODULE.UserError, match="ragbot-upload"):
        MODULE._wait_job(
            "http://127.0.0.1:8000",
            "job-1",
            api_key=None,
            timeout=1,
            poll_interval=0.1,
        )


def test_main_checks_contract_before_uploading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "guide.pdf").write_bytes(b"%PDF-test")
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "DATA_DIR", data)
    monkeypatch.setattr(MODULE, "STATE_FILE", tmp_path / "state.json")
    MODULE.STATE_FILE.write_text(json.dumps({"mode": "docker", "server": "http://127.0.0.1:8000"}))
    monkeypatch.setattr(MODULE, "_resolve_runtime_mode", lambda _state: "docker")
    order = []
    monkeypatch.setattr(
        MODULE,
        "_assert_upload_transport_available",
        lambda *args, **kwargs: order.append("contract"),
    )
    monkeypatch.setattr(
        MODULE,
        "_upload_pdf",
        lambda *args, **kwargs: order.append("upload") or {
            "status": "accepted",
            "job_id": "job-1",
            "location": "ragbot-upload:///0123456789abcdef0123456789abcdef",
        },
    )

    assert MODULE.main([str(data), "--tenant", "engineering", "--no-wait"]) == 0
    assert order == ["contract", "upload"]


def test_main_uses_same_upload_transport_for_local_and_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "guide.pdf").write_bytes(b"%PDF-test")
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "DATA_DIR", data)
    monkeypatch.setattr(MODULE, "STATE_FILE", tmp_path / "state.json")
    MODULE.STATE_FILE.write_text(json.dumps({"mode": "docker", "server": "http://127.0.0.1:8000"}))
    monkeypatch.setattr(MODULE, "_resolve_runtime_mode", lambda _state: "docker")
    monkeypatch.setattr(MODULE, "_assert_upload_transport_available", lambda *args, **kwargs: None)
    calls = []
    monkeypatch.setattr(
        MODULE,
        "_upload_pdf",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {
            "status": "accepted",
            "job_id": "job-1",
            "location": "ragbot-upload:///0123456789abcdef0123456789abcdef",
        },
    )

    assert MODULE.main([str(data), "--tenant", "engineering", "--no-wait"]) == 0
    assert len(calls) == 1
    assert calls[0][0][1] == data / "guide.pdf"
