"""Incremental Google Drive folder ingestion.

Each sync enumerates remote file metadata but downloads content only when the
Drive version changed. Unchanged files reuse the prior Ragbot chunks so the
replacement-oriented pipeline still receives a complete source snapshot.
"""
from __future__ import annotations

import io
import logging
import uuid
from pathlib import PurePosixPath
from typing import Iterable, Optional

import requests

from services.api.app.storage.models import Chunk
from services.worker.connectors.credentials import resolve_json_secret, resolve_secret
from services.worker.connectors.incremental import previous_by_external_id, reusable_chunks, stable_document_id
from services.worker.dedup.hashing import content_hash
from services.worker.jobs.ingest_text import _extract_section, _split_text
from services.worker.reliability import provider_request

logger = logging.getLogger(__name__)
_DRIVE_API = "https://www.googleapis.com/drive/v3"
_FOLDER_MIME = "application/vnd.google-apps.folder"
_GOOGLE_EXPORTS = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "application/pdf",
    "application/vnd.google-apps.drawing": "application/pdf",
}
_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".java", ".c", ".cc", ".cpp", ".h", ".hpp", ".go", ".rs", ".yaml", ".yml", ".json", ".toml", ".ini", ".csv"}


def ingest_google_drive(
    *,
    folder_id: str,
    credential_ref: str,
    doc_id: str,
    tenant_id: str,
    credential_type: str = "access_token",
    recursive: bool = True,
    max_file_bytes: int = 20 * 1024 * 1024,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    version: str = "1.0",
    tags: Optional[list] = None,
    acl_hash: Optional[str] = None,
    previous_chunks: Optional[Iterable[Chunk]] = None,
    session: Optional[requests.Session] = None,
) -> Iterable[Chunk]:
    if not folder_id.strip():
        raise ValueError("Google Drive folder_id must not be empty")
    if max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be > 0")
    token = _google_access_token(credential_ref, credential_type)
    client = session or requests.Session()
    client.headers.update({"Authorization": f"Bearer {token}"})
    previous = previous_by_external_id(previous_chunks or [])

    for item in _list_folder_files(client, folder_id.strip(), recursive=recursive):
        file_id = str(item["id"])
        name = str(item.get("name") or file_id)
        mime = str(item.get("mimeType") or "")
        remote_version = _remote_version(item)
        reused = reusable_chunks(previous, external_id=file_id, remote_version=remote_version)
        if reused is not None:
            yield from reused
            continue
        size = int(item.get("size") or 0)
        if size and size > max_file_bytes:
            logger.warning("Skipping oversized Drive file: %s (%d bytes)", name, size)
            continue
        fetched = _download_text(client, file_id, name, mime, max_file_bytes)
        if not fetched or not fetched.strip():
            continue
        file_doc_id = stable_document_id(doc_id, file_id)
        suffix = PurePosixPath(name).suffix.lower()
        uri = str(item.get("webViewLink") or f"https://drive.google.com/open?id={file_id}")
        for index, text in enumerate(_split_text(fetched, chunk_size, chunk_overlap)):
            yield Chunk(
                chunk_id=uuid.uuid4().hex,
                doc_id=file_doc_id,
                tenant_id=tenant_id,
                chunk_index=index,
                text=text,
                url=uri,
                section=_extract_section(text) if suffix in {".md", ".markdown"} else None,
                checksum=content_hash(text),
                metadata={
                    "source_type": "gdrive",
                    "external_id": file_id,
                    "remote_version": remote_version,
                    "mime_type": mime,
                    "filename": name,
                    "document_title": name,
                    "document_uri": uri,
                    "version": version,
                    "tags": tags or [],
                    "acl_hash": acl_hash or "public",
                },
            )


def _list_folder_files(client: requests.Session, root_folder_id: str, *, recursive: bool) -> Iterable[dict]:
    queue = [root_folder_id]
    visited: set[str] = set()
    while queue:
        folder_id = queue.pop(0)
        if folder_id in visited:
            continue
        visited.add(folder_id)
        page_token = None
        while True:
            params = {
                "q": f"'{folder_id}' in parents and trashed = false",
                "pageSize": 1000,
                "fields": "nextPageToken,files(id,name,mimeType,modifiedTime,md5Checksum,version,size,webViewLink)",
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            }
            if page_token:
                params["pageToken"] = page_token
            response = provider_request(
                client, "get", f"{_DRIVE_API}/files", params=params, timeout=30
            )
            response.raise_for_status()
            payload = response.json()
            for item in payload.get("files", []):
                if item.get("mimeType") == _FOLDER_MIME:
                    if recursive:
                        queue.append(str(item["id"]))
                    continue
                yield item
            page_token = payload.get("nextPageToken")
            if not page_token:
                break


def _download_text(client: requests.Session, file_id: str, name: str, mime: str, max_bytes: int) -> str:
    export_mime = _GOOGLE_EXPORTS.get(mime)
    if export_mime:
        response = provider_request(
            client,
            "get",
            f"{_DRIVE_API}/files/{file_id}/export",
            params={"mimeType": export_mime},
            stream=True,
            timeout=60,
        )
        response.raise_for_status()
        body = _read_limited(response, max_bytes)
        if export_mime == "application/pdf":
            return _pdf_text(body)
        return body.decode("utf-8", errors="replace")

    suffix = PurePosixPath(name).suffix.lower()
    if mime != "application/pdf" and not mime.startswith("text/") and suffix not in _TEXT_SUFFIXES:
        logger.info("Skipping unsupported Drive file: %s mime=%s", name, mime)
        return ""
    response = provider_request(
        client,
        "get",
        f"{_DRIVE_API}/files/{file_id}",
        params={"alt": "media", "supportsAllDrives": "true"},
        stream=True,
        timeout=60,
    )
    response.raise_for_status()
    body = _read_limited(response, max_bytes)
    if mime == "application/pdf" or suffix == ".pdf":
        return _pdf_text(body)
    return body.decode("utf-8", errors="replace")


def _read_limited(response, limit: int) -> bytes:
    chunks = []
    total = 0
    for block in response.iter_content(chunk_size=64 * 1024):
        if not block:
            continue
        total += len(block)
        if total > limit:
            raise ValueError(f"Remote Drive file exceeds max_file_bytes={limit}")
        chunks.append(block)
    return b"".join(chunks)


def _pdf_text(body: bytes) -> str:
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:
        raise RuntimeError("Google Drive PDF ingestion requires PyPDF2") from exc
    reader = PdfReader(io.BytesIO(body))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _remote_version(item: dict) -> str:
    return ":".join(
        str(item.get(key) or "") for key in ("modifiedTime", "version", "md5Checksum")
    )


def _google_access_token(credential_ref: str, credential_type: str) -> str:
    kind = credential_type.strip().lower()
    if kind == "access_token":
        return resolve_secret(credential_ref)
    info = resolve_json_secret(credential_ref)
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials as UserCredentials
        from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    except ImportError as exc:
        raise RuntimeError("Google JSON credentials require google-auth; install ragbot[saas]") from exc
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    if info.get("type") == "service_account":
        credentials = ServiceAccountCredentials.from_service_account_info(info, scopes=scopes)
    else:
        credentials = UserCredentials.from_authorized_user_info(info, scopes=scopes)
    if not credentials.valid:
        credentials.refresh(Request())
    if not credentials.token:
        raise RuntimeError("Google credential did not produce an access token")
    return str(credentials.token)
