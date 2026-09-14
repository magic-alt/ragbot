from __future__ import annotations

import io
import hashlib
from pathlib import Path

import pytest

from services.worker.uploads.s3_store import S3UploadStore
from services.worker.uploads.uri import upload_uri


class _NotFound(Exception):
    def __init__(self) -> None:
        self.response = {
            "Error": {"Code": "NoSuchKey"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }


class _Body(io.BytesIO):
    pass


class _FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.uploads: list[str] = []
        self.deletes: list[str] = []

    def head_object(self, *, Bucket: str, Key: str):
        try:
            payload = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise _NotFound() from exc
        return {"ContentLength": len(payload)}

    def upload_file(self, filename: str, bucket: str, key: str, **kwargs):
        self.objects[(bucket, key)] = Path(filename).read_bytes()
        self.uploads.append(key)

    def put_object(self, *, Bucket: str, Key: str, Body, **kwargs):
        if hasattr(Body, "read"):
            Body = Body.read()
        self.objects[(Bucket, Key)] = bytes(Body)
        return {}

    def get_object(self, *, Bucket: str, Key: str):
        try:
            payload = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise _NotFound() from exc
        return {"Body": _Body(payload), "ContentLength": len(payload)}

    def delete_object(self, *, Bucket: str, Key: str):
        self.objects.pop((Bucket, Key), None)
        self.deletes.append(Key)
        return {}


def _commit(store: S3UploadStore, root: Path, object_id: str, payload: bytes):
    temporary = root / f"{object_id}.part"
    temporary.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    return store.commit_pdf(
        temporary,
        object_id=object_id,
        sha256=digest,
        size_bytes=len(payload),
    )


def test_s3_store_deduplicates_blob_and_materializes_on_independent_node(tmp_path: Path) -> None:
    client = _FakeS3()
    payload = b"%PDF-1.7\nobject-store-ragbot\n"
    writer = S3UploadStore(
        bucket="uploads",
        prefix="ragbot",
        client=client,
        materialize_dir=tmp_path / "api-cache",
    )

    first = _commit(writer, tmp_path, "object-a", payload)
    second = _commit(writer, tmp_path, "object-b", payload)

    assert first.storage_backend == "s3"
    assert second.storage_backend == "s3"
    assert len([key for key in client.uploads if "/blobs/" in key]) == 1
    assert ("uploads", "ragbot/objects/object-a.json") in client.objects
    assert ("uploads", "ragbot/objects/object-b.json") in client.objects

    worker = S3UploadStore(
        bucket="uploads",
        prefix="ragbot",
        client=client,
        materialize_dir=tmp_path / "worker-cache",
    )
    materialized = worker.materialize_path(upload_uri("object-b"))
    assert materialized.read_bytes() == payload
    assert str(materialized).startswith(str((tmp_path / "worker-cache").resolve()))

    # Logical deletion must not delete the content-addressed blob: object-b still
    # references it and a worker on another node must remain able to materialize.
    assert writer.delete_object("object-a", sha256=first.sha256) is True
    assert ("uploads", "ragbot/objects/object-a.json") not in client.objects
    assert worker.materialize_path(upload_uri("object-b")).read_bytes() == payload
    assert any("/blobs/" in key for _bucket, key in client.objects)


def test_s3_store_rejects_commit_metadata_mismatch(tmp_path: Path) -> None:
    store = S3UploadStore(bucket="uploads", client=_FakeS3(), materialize_dir=tmp_path)
    temporary = tmp_path / "bad.part"
    temporary.write_bytes(b"%PDF-bad")
    with pytest.raises(ValueError, match="checksum/size"):
        store.commit_pdf(
            temporary,
            object_id="bad",
            sha256="0" * 64,
            size_bytes=temporary.stat().st_size,
        )


def test_s3_store_rejects_corrupt_download(tmp_path: Path) -> None:
    client = _FakeS3()
    payload = b"%PDF-1.7\ntrusted\n"
    store = S3UploadStore(bucket="uploads", prefix="x", client=client, materialize_dir=tmp_path)
    stored = _commit(store, tmp_path, "object-corrupt", payload)
    blob_key = f"x/blobs/{stored.sha256}.pdf"
    client.objects[("uploads", blob_key)] = b"corrupted bytes"
    (tmp_path / "object-corrupt.pdf").unlink(missing_ok=True)

    with pytest.raises(ValueError, match="checksum/size|expected size"):
        store.materialize_path(upload_uri("object-corrupt"))
