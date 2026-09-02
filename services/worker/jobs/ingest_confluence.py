"""Incremental Confluence space ingestion for Cloud or Data Center REST APIs."""
from __future__ import annotations

import logging
import os
import uuid
from typing import Iterable, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from services.api.app.storage.models import Chunk
from services.worker.connectors.credentials import resolve_secret
from services.worker.connectors.incremental import previous_by_external_id, reusable_chunks, stable_document_id
from services.worker.connectors.security import csv_values, validate_remote_url
from services.worker.dedup.hashing import content_hash
from services.worker.jobs.ingest_text import _split_text
from services.worker.reliability import provider_request

logger = logging.getLogger(__name__)


def ingest_confluence(
    *,
    base_url: str,
    space_key: str,
    credential_ref: str,
    doc_id: str,
    tenant_id: str,
    auth_type: str = "basic",
    email: Optional[str] = None,
    root_page_id: Optional[str] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    version: str = "1.0",
    tags: Optional[list] = None,
    acl_hash: Optional[str] = None,
    previous_chunks: Optional[Iterable[Chunk]] = None,
    session: Optional[requests.Session] = None,
) -> Iterable[Chunk]:
    base = _validate_base_url(base_url)
    if not space_key.strip():
        raise ValueError("Confluence space_key must not be empty")
    client = session or requests.Session()
    secret = resolve_secret(credential_ref)
    mode = auth_type.strip().lower()
    if mode == "basic":
        if not email or not email.strip():
            raise ValueError("Confluence basic auth requires config.email")
        client.auth = (email.strip(), secret)
    elif mode == "bearer":
        client.headers.update({"Authorization": f"Bearer {secret}"})
    else:
        raise ValueError("Confluence auth_type must be basic or bearer")
    client.headers.update({"Accept": "application/json"})

    previous = previous_by_external_id(previous_chunks or [])
    for item in _list_pages(client, base, space_key.strip(), root_page_id=root_page_id):
        page_id = str(item["id"])
        remote_version = _remote_version(item)
        reused = reusable_chunks(previous, external_id=page_id, remote_version=remote_version)
        if reused is not None:
            yield from reused
            continue
        full = _get_page(client, base, page_id)
        storage = ((full.get("body") or {}).get("storage") or {}).get("value") or ""
        text = _html_to_text(str(storage))
        if not text.strip():
            continue
        title = str(full.get("title") or item.get("title") or page_id)
        webui = str((full.get("_links") or {}).get("webui") or (item.get("_links") or {}).get("webui") or "")
        uri = urljoin(base.rstrip("/") + "/", webui.lstrip("/")) if webui else f"{base}/pages/viewpage.action?pageId={page_id}"
        confluence_doc_id = stable_document_id(doc_id, page_id)
        for index, segment in enumerate(_split_text(text, chunk_size, chunk_overlap)):
            yield Chunk(
                chunk_id=uuid.uuid4().hex,
                doc_id=confluence_doc_id,
                tenant_id=tenant_id,
                chunk_index=index,
                text=segment,
                url=uri,
                checksum=content_hash(segment),
                metadata={
                    "source_type": "confluence",
                    "external_id": page_id,
                    "remote_version": _remote_version(full) or remote_version,
                    "space_key": space_key,
                    "document_title": title,
                    "document_uri": uri,
                    "version": version,
                    "tags": tags or [],
                    "acl_hash": acl_hash or "public",
                },
            )


def _list_pages(client: requests.Session, base: str, space_key: str, *, root_page_id: Optional[str]) -> Iterable[dict]:
    start = 0
    while True:
        params = {
            "spaceKey": space_key,
            "type": "page",
            "status": "current",
            "limit": 100,
            "start": start,
            "expand": "version,history.lastUpdated,ancestors",
        }
        response = provider_request(
            client, "get", f"{base}/rest/api/content", params=params, timeout=30
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", []) if isinstance(payload, dict) else []
        for item in results:
            if root_page_id:
                ancestors = {str(parent.get("id")) for parent in item.get("ancestors", []) if isinstance(parent, dict)}
                if str(item.get("id")) != str(root_page_id) and str(root_page_id) not in ancestors:
                    continue
            yield item
        if not results:
            break
        size = int(payload.get("size") or len(results))
        limit = int(payload.get("limit") or 100)
        if size < limit and not (payload.get("_links") or {}).get("next"):
            break
        start += size


def _get_page(client: requests.Session, base: str, page_id: str) -> dict:
    response = provider_request(
        client,
        "get",
        f"{base}/rest/api/content/{page_id}",
        params={"expand": "body.storage,version,history.lastUpdated"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Confluence API returned a non-object page")
    return payload


def _html_to_text(value: str) -> str:
    soup = BeautifulSoup(value, "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    lines = [line.strip() for line in soup.get_text("\n").splitlines()]
    return "\n".join(line for line in lines if line)


def _remote_version(item: dict) -> str:
    number = str((item.get("version") or {}).get("number") or "")
    updated = str(((item.get("history") or {}).get("lastUpdated") or {}).get("when") or "")
    return f"{number}:{updated}"


def _validate_base_url(value: str) -> str:
    base = str(value or "").strip().rstrip("/")
    allowed = csv_values("RAGBOT_CONFLUENCE_ALLOWED_HOSTS")
    environment = os.getenv("RAGBOT_ENV", "development").strip().lower()
    if environment in {"production", "prod"} and not allowed:
        raise ValueError("Production Confluence ingestion requires RAGBOT_CONFLUENCE_ALLOWED_HOSTS")
    validate_remote_url(base, allowed_hosts=allowed or None)
    return base
