"""Incremental Notion page-tree ingestion using a secret-referenced API token."""
from __future__ import annotations

import logging
import uuid
from typing import Iterable, Optional

import requests

from services.api.app.storage.models import Chunk
from services.worker.chunking import chunking_metadata, split_text
from services.worker.connectors.credentials import resolve_secret
from services.worker.connectors.incremental import previous_by_external_id, reusable_chunks, stable_document_id
from services.worker.dedup.hashing import content_hash
from services.worker.reliability import provider_request

logger = logging.getLogger(__name__)
_NOTION_API = "https://api.notion.com/v1"


def ingest_notion(
    *,
    page_id: str,
    credential_ref: str,
    doc_id: str,
    tenant_id: str,
    recursive: bool = True,
    notion_version: str = "2022-06-28",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    version: str = "1.0",
    tags: Optional[list] = None,
    acl_hash: Optional[str] = None,
    previous_chunks: Optional[Iterable[Chunk]] = None,
    session: Optional[requests.Session] = None,
    chunking: Optional[dict] = None,
) -> Iterable[Chunk]:
    root = _clean_notion_id(page_id)
    if not root:
        raise ValueError("Notion page_id must not be empty")
    token = resolve_secret(credential_ref)
    client = session or requests.Session()
    client.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Notion-Version": notion_version,
            "Content-Type": "application/json",
        }
    )
    previous = previous_by_external_id(previous_chunks or [])
    required_chunking = chunking_metadata(
        chunking,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    queue: list[str] = [root]
    visited: set[str] = set()

    while queue:
        current = _clean_notion_id(queue.pop(0))
        if current in visited:
            continue
        visited.add(current)
        page = _get_json(client, f"{_NOTION_API}/pages/{current}")
        remote_version = str(page.get("last_edited_time") or page.get("created_time") or "")
        title = _page_title(page) or current
        reused = reusable_chunks(
            previous,
            external_id=current,
            remote_version=remote_version,
            required_metadata=required_chunking,
        )

        text = ""
        children: list[str] = []
        if recursive or reused is None:
            text, children = _fetch_block_tree(client, current)
        if recursive:
            queue.extend(child for child in children if child not in visited)

        if reused is not None:
            yield from reused
            continue
        if not text.strip():
            logger.info("Skipping empty Notion page: %s", current)
            continue

        notion_doc_id = stable_document_id(doc_id, current)
        uri = str(page.get("url") or f"https://www.notion.so/{current.replace('-', '')}")
        segments, chunker_metadata = split_text(
            text,
            chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for index, segment in enumerate(segments):
            yield Chunk(
                chunk_id=uuid.uuid4().hex,
                doc_id=notion_doc_id,
                tenant_id=tenant_id,
                chunk_index=index,
                text=segment,
                url=uri,
                checksum=content_hash(segment),
                metadata={
                    "source_type": "notion",
                    "external_id": current,
                    "remote_version": remote_version,
                    "document_title": title,
                    "document_uri": uri,
                    "version": version,
                    "tags": tags or [],
                    "acl_hash": acl_hash or "public",
                    **chunker_metadata,
                },
            )


def _fetch_block_tree(client: requests.Session, root_block_id: str) -> tuple[str, list[str]]:
    lines: list[str] = []
    child_pages: list[str] = []
    queue = [root_block_id]
    visited: set[str] = set()
    while queue:
        block_id = queue.pop(0)
        if block_id in visited:
            continue
        visited.add(block_id)
        cursor = None
        while True:
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            payload = _get_json(client, f"{_NOTION_API}/blocks/{block_id}/children", params=params)
            for block in payload.get("results", []):
                block_type = str(block.get("type") or "")
                if block_type == "child_page":
                    child_pages.append(str(block.get("id") or ""))
                    title = str((block.get("child_page") or {}).get("title") or "")
                    if title:
                        lines.append(title)
                    continue
                text = _block_text(block)
                if text:
                    lines.append(text)
                if block.get("has_children"):
                    nested = str(block.get("id") or "")
                    if nested:
                        queue.append(nested)
            if not payload.get("has_more"):
                break
            cursor = payload.get("next_cursor")
            if not cursor:
                break
    return "\n".join(lines), child_pages


def _block_text(block: dict) -> str:
    block_type = str(block.get("type") or "")
    value = block.get(block_type)
    if not isinstance(value, dict):
        return ""
    rich_text = value.get("rich_text")
    if isinstance(rich_text, list):
        text = "".join(str(item.get("plain_text") or "") for item in rich_text if isinstance(item, dict))
    else:
        text = ""
    if block_type.startswith("heading_") and text:
        level = block_type.rsplit("_", 1)[-1]
        prefix = "#" * int(level) if level.isdigit() else "#"
        return f"{prefix} {text}"
    if block_type in {"bulleted_list_item", "numbered_list_item", "to_do"} and text:
        return f"- {text}"
    if block_type == "code" and text:
        language = str(value.get("language") or "")
        return f"```{language}\n{text}\n```"
    return text


def _page_title(page: dict) -> str:
    properties = page.get("properties") or {}
    if not isinstance(properties, dict):
        return ""
    for prop in properties.values():
        if not isinstance(prop, dict) or prop.get("type") != "title":
            continue
        return "".join(
            str(item.get("plain_text") or "")
            for item in prop.get("title", [])
            if isinstance(item, dict)
        ).strip()
    return ""


def _get_json(client: requests.Session, url: str, params: Optional[dict] = None) -> dict:
    response = provider_request(client, "get", url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Notion API returned a non-object payload")
    return payload


def _clean_notion_id(value: str) -> str:
    raw = str(value or "").strip().replace("-", "")
    if len(raw) != 32 or any(ch not in "0123456789abcdefABCDEF" for ch in raw):
        return str(value or "").strip()
    return f"{raw[0:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:32]}".lower()
