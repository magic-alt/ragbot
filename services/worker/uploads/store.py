from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

from .uri import upload_object_id


@dataclass(frozen=True)
class StoredUpload:
    object_id: str
    sha256: str
    size_bytes: int
    storage_backend: str
    storage_key: str


@runtime_checkable
class UploadStore(Protocol):
    """Ragbot-owned port for server-managed uploaded objects."""

    def temporary_path(self, object_id: str) -> Path: ...

    def commit_pdf(self, temporary: Path, *, object_id: str, sha256: str, size_bytes: int) -> StoredUpload: ...

    def local_path(self, uri: str) -> Path: ...

    def delete_object(self, object_id: str, *, sha256: str | None = None) -> bool: ...


class FilesystemUploadStore:
    """Development/single-node adapter with content-addressed blob deduplication.

    Logical objects are independent from physical blobs. Each object gets a hard
    link below ``objects/`` pointing at a SHA-256 blob below ``blobs/``. Uploading
    the same bytes twice therefore creates distinct Source identities without
    duplicating the underlying payload on filesystems that support hard links.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.blob_root = self.root / "blobs"
        self.object_root = self.root / "objects"
        self.tmp_root = self.root / "tmp"
        for directory in (self.blob_root, self.object_root, self.tmp_root):
            directory.mkdir(parents=True, exist_ok=True)

    def temporary_path(self, object_id: str) -> Path:
        return self.tmp_root / f"{object_id}.part"

    def commit_pdf(self, temporary: Path, *, object_id: str, sha256: str, size_bytes: int) -> StoredUpload:
        blob = self.blob_root / f"{sha256}.pdf"
        object_path = self.object_root / f"{object_id}.pdf"
        if not blob.exists():
            os.replace(temporary, blob)
        else:
            temporary.unlink(missing_ok=True)
        object_path.unlink(missing_ok=True)
        try:
            os.link(blob, object_path)
        except OSError:
            shutil.copy2(blob, object_path)
        return StoredUpload(
            object_id=object_id,
            sha256=sha256,
            size_bytes=size_bytes,
            storage_backend="filesystem",
            storage_key=f"objects/{object_id}.pdf",
        )

    def local_path(self, uri: str) -> Path:
        object_id = upload_object_id(uri)
        path = (self.object_root / f"{object_id}.pdf").resolve()
        try:
            path.relative_to(self.object_root.resolve())
        except ValueError as exc:
            raise ValueError("Upload object escapes configured upload root") from exc
        if not path.is_file():
            raise ValueError(f"Upload object is not available: {object_id}")
        return path

    def delete_object(self, object_id: str, *, sha256: str | None = None) -> bool:
        object_path = self.object_root / f"{object_id}.pdf"
        existed = object_path.exists()
        object_path.unlink(missing_ok=True)
        if sha256:
            blob = self.blob_root / f"{sha256}.pdf"
            if blob.exists():
                try:
                    if blob.stat().st_nlink <= 1:
                        blob.unlink(missing_ok=True)
                except OSError:
                    pass
        return existed


def build_upload_store_from_env() -> UploadStore:
    backend = os.getenv("RAGBOT_UPLOAD_STORE", "filesystem").strip().lower()
    if backend != "filesystem":
        raise ValueError(
            f"Unsupported RAGBOT_UPLOAD_STORE={backend!r}; this release provides the filesystem adapter"
        )
    root = os.getenv("RAGBOT_UPLOAD_DIR", "").strip()
    if not root:
        raise ValueError("Server-managed uploads require RAGBOT_UPLOAD_DIR")
    return FilesystemUploadStore(root)
