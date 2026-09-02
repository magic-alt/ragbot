"""Helpers shared by metadata-first incremental cloud connectors."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Iterable

from services.api.app.storage.models import Chunk


def stable_document_id(base_doc_id: str, external_id: str) -> str:
    digest = hashlib.sha256(external_id.encode("utf-8")).hexdigest()[:20]
    return f"{base_doc_id}:{digest}"


def previous_by_external_id(chunks: Iterable[Chunk]) -> dict[str, list[Chunk]]:
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        external_id = str((chunk.metadata or {}).get("external_id") or "").strip()
        if external_id:
            grouped.setdefault(external_id, []).append(chunk)
    for items in grouped.values():
        items.sort(key=lambda chunk: chunk.chunk_index)
    return grouped


def reusable_chunks(
    previous: dict[str, list[Chunk]],
    *,
    external_id: str,
    remote_version: str,
) -> list[Chunk] | None:
    items = previous.get(external_id)
    if not items:
        return None
    versions = {str((chunk.metadata or {}).get("remote_version") or "") for chunk in items}
    if versions != {str(remote_version)}:
        return None
    return [replace(chunk, metadata=dict(chunk.metadata or {})) for chunk in items]
