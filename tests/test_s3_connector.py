from __future__ import annotations

import sys

import pytest

from services.api.app.routes.quick_import import (
    build_source_config,
    canonical_location,
    infer_source_type,
)
from services.worker.jobs.ingest_s3 import _validate_custom_endpoint, ingest_s3


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self, amount: int = -1):
        return self._data if amount < 0 else self._data[:amount]


class _Paginator:
    def paginate(self, **kwargs):
        assert kwargs == {"Bucket": "manuals", "Prefix": "engineering/"}
        return [
            {
                "Contents": [
                    {"Key": "engineering/readme.md", "Size": 80, "ETag": '"etag-a"'},
                    {"Key": "engineering/servo.txt", "Size": 80, "ETag": '"etag-b"'},
                    {"Key": "engineering/image.png", "Size": 20, "ETag": '"ignored"'},
                    {"Key": "engineering/huge.txt", "Size": 1000000, "ETag": '"large"'},
                ]
            }
        ]


class _S3Client:
    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return _Paginator()

    def get_object(self, *, Bucket: str, Key: str):
        assert Bucket == "manuals"
        data = {
            "engineering/readme.md": b"# EtherCAT\nDistributed clocks synchronize servo loops. " * 2,
            "engineering/servo.txt": b"FOC current loops use dq transforms and current regulators. " * 2,
        }[Key]
        return {"Body": _Body(data)}


class _Boto3:
    def __init__(self):
        self.kwargs = None

    def client(self, service: str, **kwargs):
        assert service == "s3"
        self.kwargs = kwargs
        return _S3Client()


def test_s3_location_inference_and_canonical_identity():
    assert infer_source_type("s3://Manuals/Engineering/") == "s3"
    assert canonical_location("s3://Manuals/Engineering/") == "s3://manuals/Engineering"
    assert canonical_location("s3://manuals/Engineering") == "s3://manuals/Engineering"
    config = build_source_config(
        "s3",
        "s3://manuals/engineering/",
        {"endpoint_url": "http://minio:9000", "credential_env_prefix": "RAGBOT_MINIO"},
    )
    assert config == {
        "bucket": "manuals",
        "prefix": "engineering/",
        "endpoint_url": "http://minio:9000",
        "credential_env_prefix": "RAGBOT_MINIO",
    }


def test_s3_connector_lists_supported_objects_and_never_requires_config_secrets(monkeypatch):
    fake = _Boto3()
    monkeypatch.setitem(sys.modules, "boto3", fake)
    monkeypatch.setenv("MINIO_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("MINIO_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("RAGBOT_S3_ALLOWED_HOSTS", "minio")

    chunks = list(
        ingest_s3(
            bucket="manuals",
            prefix="engineering/",
            doc_id="doc-s3-source",
            tenant_id="tenant-a",
            endpoint_url="http://minio:9000",
            credential_env_prefix="MINIO",
            max_object_bytes=1024,
            chunk_size=60,
            chunk_overlap=10,
            tags=["manuals"],
            acl_hash="acl-hash",
        )
    )

    assert len(chunks) >= 4
    assert {chunk.metadata["object_key"] for chunk in chunks} == {
        "engineering/readme.md",
        "engineering/servo.txt",
    }
    assert len({chunk.doc_id for chunk in chunks}) == 2
    assert all(chunk.path.startswith("s3://manuals/") for chunk in chunks)
    assert all(chunk.metadata["source_type"] == "s3" for chunk in chunks)
    assert all(chunk.metadata["acl_hash"] == "acl-hash" for chunk in chunks)
    assert fake.kwargs["endpoint_url"] == "http://minio:9000"
    assert fake.kwargs["aws_access_key_id"] == "access"
    assert fake.kwargs["aws_secret_access_key"] == "secret"
    # Secret values are deployment environment data, never written into chunks.
    assert "access" not in str([chunk.metadata for chunk in chunks])
    assert "secret" not in str([chunk.metadata for chunk in chunks])


def test_production_custom_s3_endpoint_requires_explicit_allowlist(monkeypatch):
    monkeypatch.setenv("RAGBOT_ENV", "production")
    monkeypatch.delenv("RAGBOT_S3_ALLOWED_HOSTS", raising=False)
    with pytest.raises(ValueError, match="RAGBOT_S3_ALLOWED_HOSTS"):
        _validate_custom_endpoint("http://minio.internal:9000")

    monkeypatch.setenv("RAGBOT_S3_ALLOWED_HOSTS", "minio.internal")
    assert _validate_custom_endpoint("http://minio.internal:9000") == "http://minio.internal:9000"
