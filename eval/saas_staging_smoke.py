"""Manual real-account smoke test for SaaS connectors.

This is intentionally separate from ordinary PR CI. It validates provider API
compatibility and one unchanged incremental replay using staging identities.
"""
from __future__ import annotations

import os


def main() -> int:
    requested = [item.strip().lower() for item in os.getenv("SAAS_SMOKE_CONNECTORS", "").split(",") if item.strip()]
    if not requested:
        raise RuntimeError("SAAS_SMOKE_CONNECTORS must name at least one connector")

    results = {}
    for connector in requested:
        if connector == "gdrive":
            results[connector] = _drive()
        elif connector == "notion":
            results[connector] = _notion()
        elif connector == "confluence":
            results[connector] = _confluence()
        else:
            raise ValueError(f"Unsupported SaaS smoke connector: {connector}")

    print("SaaS staging smoke results:")
    for name, result in results.items():
        print(f"  {name}: chunks={result['chunks']} replay_reused={result['replay_reused']}")
    if not all(result["chunks"] > 0 and result["replay_reused"] for result in results.values()):
        raise AssertionError("SaaS staging smoke did not meet non-empty/incremental-reuse gate")
    return 0


def _drive() -> dict:
    from services.worker.jobs.ingest_google_drive import ingest_google_drive

    folder_id = _required("STAGING_GDRIVE_FOLDER_ID")
    credential_ref = _required("STAGING_GDRIVE_CREDENTIAL_REF")
    credential_type = os.getenv("STAGING_GDRIVE_CREDENTIAL_TYPE", "google_json")
    first = list(
        ingest_google_drive(
            folder_id=folder_id,
            credential_ref=credential_ref,
            credential_type=credential_type,
            doc_id="staging-gdrive",
            tenant_id="staging",
            recursive=_flag("STAGING_GDRIVE_RECURSIVE", True),
        )
    )
    _require_chunks("gdrive", first)
    second = list(
        ingest_google_drive(
            folder_id=folder_id,
            credential_ref=credential_ref,
            credential_type=credential_type,
            doc_id="staging-gdrive",
            tenant_id="staging",
            recursive=_flag("STAGING_GDRIVE_RECURSIVE", True),
            previous_chunks=first,
        )
    )
    return {"chunks": len(first), "replay_reused": _same_chunk_ids(first, second)}


def _notion() -> dict:
    from services.worker.jobs.ingest_notion import ingest_notion

    page_id = _required("STAGING_NOTION_PAGE_ID")
    credential_ref = _required("STAGING_NOTION_CREDENTIAL_REF")
    recursive = _flag("STAGING_NOTION_RECURSIVE", False)
    first = list(
        ingest_notion(
            page_id=page_id,
            credential_ref=credential_ref,
            doc_id="staging-notion",
            tenant_id="staging",
            recursive=recursive,
        )
    )
    _require_chunks("notion", first)
    second = list(
        ingest_notion(
            page_id=page_id,
            credential_ref=credential_ref,
            doc_id="staging-notion",
            tenant_id="staging",
            recursive=recursive,
            previous_chunks=first,
        )
    )
    return {"chunks": len(first), "replay_reused": _same_chunk_ids(first, second)}


def _confluence() -> dict:
    from services.worker.jobs.ingest_confluence import ingest_confluence

    base_url = _required("STAGING_CONFLUENCE_BASE_URL")
    space_key = _required("STAGING_CONFLUENCE_SPACE_KEY")
    credential_ref = _required("STAGING_CONFLUENCE_CREDENTIAL_REF")
    auth_type = os.getenv("STAGING_CONFLUENCE_AUTH_TYPE", "basic")
    email = os.getenv("STAGING_CONFLUENCE_EMAIL")
    root_page_id = os.getenv("STAGING_CONFLUENCE_ROOT_PAGE_ID") or None
    first = list(
        ingest_confluence(
            base_url=base_url,
            space_key=space_key,
            credential_ref=credential_ref,
            auth_type=auth_type,
            email=email,
            root_page_id=root_page_id,
            doc_id="staging-confluence",
            tenant_id="staging",
        )
    )
    _require_chunks("confluence", first)
    second = list(
        ingest_confluence(
            base_url=base_url,
            space_key=space_key,
            credential_ref=credential_ref,
            auth_type=auth_type,
            email=email,
            root_page_id=root_page_id,
            doc_id="staging-confluence",
            tenant_id="staging",
            previous_chunks=first,
        )
    )
    return {"chunks": len(first), "replay_reused": _same_chunk_ids(first, second)}


def _same_chunk_ids(first, second) -> bool:
    return bool(first) and [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def _require_chunks(name: str, chunks) -> None:
    if not chunks:
        raise AssertionError(f"{name} staging source returned no chunks")


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required staging variable: {name}")
    return value


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
