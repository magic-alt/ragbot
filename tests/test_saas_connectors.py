from __future__ import annotations

from services.api.app.storage.models import Chunk
from services.api.app.routes.quick_import import build_source_config, canonical_location, infer_source_type
from services.api.app.routes.sources import _validate_source_config
from services.worker.connectors.credentials import resolve_secret, validate_secret_ref
from services.worker.connectors.incremental import stable_document_id
from services.worker.jobs.ingest_confluence import ingest_confluence
from services.worker.jobs.ingest_google_drive import ingest_google_drive
from services.worker.jobs.ingest_notion import ingest_notion


class _Response:
    def __init__(self, payload=None, body: bytes = b""):
        self._payload = payload
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=65536):
        if self._body:
            yield self._body


class _Headers(dict):
    pass


def _previous(base: str, external_id: str, remote_version: str, text: str = "unchanged") -> Chunk:
    return Chunk(
        chunk_id=f"old-{external_id}",
        doc_id=stable_document_id(base, external_id),
        tenant_id="tenant-a",
        chunk_index=0,
        text=text,
        checksum=f"checksum-{external_id}",
        metadata={
            "source_type": "test",
            "external_id": external_id,
            "remote_version": remote_version,
            "version": "1.0",
            "tags": [],
            "acl_hash": "public",
        },
    )


def test_secret_refs_are_environment_only(monkeypatch):
    assert validate_secret_ref("env:RAGBOT_NOTION_TOKEN") == "env:RAGBOT_NOTION_TOKEN"
    monkeypatch.setenv("RAGBOT_NOTION_TOKEN", "secret-value")
    assert resolve_secret("env:RAGBOT_NOTION_TOKEN") == "secret-value"
    for invalid in ("secret-value", "file:/tmp/token", "vault:secret/path"):
        try:
            validate_secret_ref(invalid)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"secret reference unexpectedly accepted: {invalid}")


def test_quick_import_recognizes_cloud_locations():
    assert infer_source_type("gdrive://folder-123") == "gdrive"
    assert infer_source_type("https://drive.google.com/drive/folders/folder-123") == "gdrive"
    assert infer_source_type("notion://0123456789abcdef0123456789abcdef") == "notion"
    assert infer_source_type("https://www.notion.so/Runbook-0123456789abcdef0123456789abcdef") == "notion"
    assert infer_source_type("confluence://acme.atlassian.net/ENG") == "confluence"
    assert infer_source_type("https://acme.atlassian.net/wiki/spaces/ENG/overview") == "confluence"

    drive = build_source_config("gdrive", "gdrive://folder-123", {"credential_ref": "env:DRIVE_TOKEN"})
    assert drive["folder_id"] == "folder-123"
    notion = build_source_config(
        "notion",
        "https://www.notion.so/Runbook-0123456789abcdef0123456789abcdef",
        {"credential_ref": "env:NOTION_TOKEN"},
    )
    assert notion["page_id"].replace("-", "") == "0123456789abcdef0123456789abcdef"
    confluence = build_source_config(
        "confluence",
        "https://acme.atlassian.net/wiki/spaces/ENG/overview",
        {"credential_ref": "env:CONF_TOKEN", "email": "bot@example.com"},
    )
    assert confluence["base_url"] == "https://acme.atlassian.net/wiki"
    assert confluence["space_key"] == "ENG"
    assert canonical_location("https://drive.google.com/drive/folders/folder-123") == "gdrive://folder-123"


def test_cloud_source_validation_rejects_inline_credentials():
    _validate_source_config("gdrive", {"folder_id": "f", "credential_ref": "env:DRIVE_TOKEN"})
    _validate_source_config("notion", {"page_id": "p", "credential_ref": "env:NOTION_TOKEN"})
    _validate_source_config(
        "confluence",
        {
            "base_url": "https://acme.atlassian.net/wiki",
            "space_key": "ENG",
            "credential_ref": "env:CONF_TOKEN",
            "email": "bot@example.com",
        },
    )
    try:
        _validate_source_config("notion", {"page_id": "p", "credential_ref": "env:T", "access_token": "raw"})
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 422
    else:  # pragma: no cover
        raise AssertionError("inline credential unexpectedly accepted")


class _DriveSession:
    def __init__(self):
        self.headers = _Headers()
        self.calls = []

    def get(self, url, params=None, **kwargs):
        self.calls.append((url, params))
        if url.endswith("/files"):
            return _Response(
                {
                    "files": [
                        {
                            "id": "same",
                            "name": "same.txt",
                            "mimeType": "text/plain",
                            "modifiedTime": "2026-09-01T00:00:00Z",
                            "version": "1",
                            "md5Checksum": "aaa",
                            "size": "20",
                        },
                        {
                            "id": "changed",
                            "name": "changed.txt",
                            "mimeType": "text/plain",
                            "modifiedTime": "2026-09-02T00:00:00Z",
                            "version": "2",
                            "md5Checksum": "bbb",
                            "size": "40",
                        },
                    ]
                }
            )
        if url.endswith("/files/changed"):
            return _Response(body=b"Changed Drive engineering runbook content.")
        raise AssertionError(f"unexpected Drive request: {url}")


def test_google_drive_reuses_unchanged_file_without_downloading(monkeypatch):
    monkeypatch.setenv("DRIVE_TOKEN", "token")
    base = "doc-drive"
    previous = [_previous(base, "same", "2026-09-01T00:00:00Z:1:aaa")]
    session = _DriveSession()
    chunks = list(
        ingest_google_drive(
            folder_id="folder",
            credential_ref="env:DRIVE_TOKEN",
            doc_id=base,
            tenant_id="tenant-a",
            recursive=False,
            previous_chunks=previous,
            session=session,
        )
    )
    assert any(chunk.chunk_id == "old-same" for chunk in chunks)
    assert any(chunk.metadata.get("external_id") == "changed" for chunk in chunks)
    assert not any(url.endswith("/files/same") for url, _ in session.calls)
    assert any(url.endswith("/files/changed") for url, _ in session.calls)


class _NotionSession:
    def __init__(self):
        self.headers = _Headers()
        self.calls = []

    def get(self, url, params=None, **kwargs):
        self.calls.append(url)
        if "/pages/" in url:
            return _Response(
                {
                    "id": "01234567-89ab-cdef-0123-456789abcdef",
                    "last_edited_time": "2026-09-02T01:00:00Z",
                    "url": "https://www.notion.so/page",
                    "properties": {
                        "title": {
                            "type": "title",
                            "title": [{"plain_text": "Runbook"}],
                        }
                    },
                }
            )
        if "/blocks/" in url:
            return _Response(
                {
                    "results": [
                        {
                            "id": "block-1",
                            "type": "paragraph",
                            "paragraph": {"rich_text": [{"plain_text": "Servo deployment checklist"}]},
                            "has_children": False,
                        }
                    ],
                    "has_more": False,
                }
            )
        raise AssertionError(url)


def test_notion_nonrecursive_unchanged_page_avoids_block_download(monkeypatch):
    monkeypatch.setenv("NOTION_TOKEN", "token")
    page_id = "01234567-89ab-cdef-0123-456789abcdef"
    previous = [_previous("doc-notion", page_id, "2026-09-02T01:00:00Z")]
    session = _NotionSession()
    chunks = list(
        ingest_notion(
            page_id=page_id,
            credential_ref="env:NOTION_TOKEN",
            doc_id="doc-notion",
            tenant_id="tenant-a",
            recursive=False,
            previous_chunks=previous,
            session=session,
        )
    )
    assert [chunk.chunk_id for chunk in chunks] == [f"old-{page_id}"]
    assert not any("/blocks/" in url for url in session.calls)


class _ConfluenceSession:
    def __init__(self):
        self.headers = _Headers()
        self.auth = None
        self.calls = []

    def get(self, url, params=None, **kwargs):
        self.calls.append(url)
        if url.endswith("/rest/api/content"):
            return _Response(
                {
                    "results": [
                        {
                            "id": "10",
                            "title": "Stable",
                            "version": {"number": 3},
                            "history": {"lastUpdated": {"when": "2026-09-01T00:00:00Z"}},
                            "ancestors": [],
                        },
                        {
                            "id": "20",
                            "title": "Changed",
                            "version": {"number": 4},
                            "history": {"lastUpdated": {"when": "2026-09-02T00:00:00Z"}},
                            "ancestors": [],
                        },
                    ],
                    "size": 2,
                    "limit": 100,
                    "_links": {},
                }
            )
        if url.endswith("/rest/api/content/20"):
            return _Response(
                {
                    "id": "20",
                    "title": "Changed",
                    "body": {"storage": {"value": "<h1>EtherCAT</h1><p>Updated commissioning procedure.</p>"}},
                    "version": {"number": 4},
                    "history": {"lastUpdated": {"when": "2026-09-02T00:00:00Z"}},
                    "_links": {"webui": "/spaces/ENG/pages/20"},
                }
            )
        raise AssertionError(url)


def test_confluence_reuses_stable_page_and_fetches_only_changed(monkeypatch):
    monkeypatch.setenv("CONF_TOKEN", "token")
    monkeypatch.setenv("RAGBOT_ALLOW_PRIVATE_SOURCE_NETWORKS", "true")
    previous = [_previous("doc-conf", "10", "3:2026-09-01T00:00:00Z")]
    session = _ConfluenceSession()
    chunks = list(
        ingest_confluence(
            base_url="https://acme.atlassian.net/wiki",
            space_key="ENG",
            credential_ref="env:CONF_TOKEN",
            email="bot@example.com",
            doc_id="doc-conf",
            tenant_id="tenant-a",
            previous_chunks=previous,
            session=session,
        )
    )
    assert any(chunk.chunk_id == "old-10" for chunk in chunks)
    assert any(chunk.metadata.get("external_id") == "20" for chunk in chunks)
    assert not any(url.endswith("/rest/api/content/10") for url in session.calls)
    assert any(url.endswith("/rest/api/content/20") for url in session.calls)
    assert session.auth == ("bot@example.com", "token")
