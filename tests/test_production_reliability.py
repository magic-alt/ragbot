from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
import requests
from fastapi import HTTPException

from services.api.app.auth.principal import require_operator
from services.api.app.storage.models import IngestionJob
from services.api.app.storage.repo import InMemoryRepo
from services.worker.main import _retry_or_dead_letter
from services.worker.reliability import (
    durable_retry_delay,
    provider_request,
)


class _Response:
    def __init__(self, status_code: int, *, headers=None, payload=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            response.url = "https://provider.example/resource"
            response._content = b"provider failure"
            raise requests.HTTPError(
                f"{self.status_code} provider failure",
                response=response,
            )


class _SequenceSession:
    def __init__(self, events):
        self.events = list(events)
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        event = self.events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event


def test_provider_request_honors_retry_after_then_succeeds():
    session = _SequenceSession(
        [
            _Response(429, headers={"Retry-After": "2"}),
            _Response(200, payload={"ok": True}),
        ]
    )
    slept = []
    response = provider_request(
        session,
        "get",
        "https://provider.example/resource",
        max_attempts=3,
        backoff_base_seconds=0.1,
        backoff_max_seconds=10,
        sleep=slept.append,
        random_fn=lambda: 0,
    )
    assert response.status_code == 200
    assert session.calls == 2
    assert slept == [2.0]


def test_provider_request_does_not_retry_permanent_401():
    session = _SequenceSession([_Response(401)])
    slept = []
    response = provider_request(
        session,
        "get",
        "https://provider.example/resource",
        max_attempts=4,
        backoff_base_seconds=0.1,
        backoff_max_seconds=1,
        sleep=slept.append,
    )
    assert response.status_code == 401
    assert session.calls == 1
    assert slept == []


def test_provider_request_retries_transport_error():
    session = _SequenceSession(
        [requests.ConnectionError("temporary network error"), _Response(200)]
    )
    slept = []
    response = provider_request(
        session,
        "get",
        "https://provider.example/resource",
        max_attempts=2,
        backoff_base_seconds=0.2,
        backoff_max_seconds=1,
        sleep=slept.append,
        random_fn=lambda: 0,
    )
    assert response.status_code == 200
    assert session.calls == 2
    assert slept == [0.1]


def test_durable_retry_delay_is_bounded_exponential():
    assert durable_retry_delay(1, base_seconds=5, max_seconds=60) == 5
    assert durable_retry_delay(2, base_seconds=5, max_seconds=60) == 10
    assert durable_retry_delay(5, base_seconds=5, max_seconds=60) == 60


def _job(*, attempts: int = 1) -> IngestionJob:
    return IngestionJob(
        job_id="job-1",
        tenant_id="tenant-1",
        source_id="source-1",
        source_type="notion",
        source_config={"page_id": "page", "credential_ref": "env:TOKEN"},
        status="failed",
        attempts=attempts,
        error="429 provider failure",
        completed_at=datetime.now(timezone.utc).isoformat(),
    )


def test_retryable_ingestion_failure_returns_to_pending_queue():
    repo = InMemoryRepo()
    job = _job(attempts=1)
    repo.add_job(job)
    before = datetime.now(timezone.utc)

    _retry_or_dead_letter(
        repo,
        job,
        error="429 provider failure",
        failure_class="http_429",
        retryable=True,
        max_attempts=3,
        retry_base_seconds=5,
        retry_max_seconds=60,
    )

    updated = repo.get_job(job.job_id)
    assert updated.status == "pending"
    assert updated.failure_class == "http_429"
    assert updated.completed_at is None
    assert updated.dead_lettered_at is None
    assert datetime.fromisoformat(updated.available_at) >= before
    failures = updated.stats["attempt_failures"]
    assert failures[-1]["attempt"] == 1
    assert failures[-1]["retry_delay_seconds"] == 5


def test_exhausted_retryable_failure_moves_to_dead_letter():
    repo = InMemoryRepo()
    job = _job(attempts=3)
    repo.add_job(job)

    _retry_or_dead_letter(
        repo,
        job,
        error="503 provider failure",
        failure_class="http_503",
        retryable=True,
        max_attempts=3,
        retry_base_seconds=5,
        retry_max_seconds=60,
    )

    updated = repo.get_job(job.job_id)
    assert updated.status == "dead_lettered"
    assert updated.failure_class == "http_503"
    assert updated.dead_lettered_at is not None
    assert updated.stats["dead_letter"]["attempts"] == 3


def test_nonretryable_failure_dead_letters_immediately():
    repo = InMemoryRepo()
    job = _job(attempts=1)
    repo.add_job(job)

    _retry_or_dead_letter(
        repo,
        job,
        error="401 invalid credential",
        failure_class="http_401",
        retryable=False,
        max_attempts=3,
        retry_base_seconds=5,
        retry_max_seconds=60,
    )

    updated = repo.get_job(job.job_id)
    assert updated.status == "dead_lettered"
    assert updated.failure_class == "http_401"


def _principal_env(role: str | None = None, *, admin: bool = False) -> dict[str, str]:
    roles = [role] if role else []
    return {
        "RAGBOT_API_KEY_PRINCIPALS": json.dumps(
            {
                "key": {
                    "tenant_ids": ["tenant-1"],
                    "user_id": "svc",
                    "groups": [],
                    "roles": roles,
                    "admin": admin,
                }
            }
        )
    }


@pytest.mark.parametrize("role", ["operator", "owner"])
def test_operator_roles_allow_mutation(role):
    with patch.dict(os.environ, _principal_env(role), clear=True):
        principal = require_operator("key")
        assert principal is not None
        assert role in principal.roles


def test_reader_role_cannot_mutate():
    with patch.dict(os.environ, _principal_env("reader"), clear=True):
        with pytest.raises(HTTPException) as exc:
            require_operator("key")
    assert exc.value.status_code == 403


def test_global_admin_bypasses_operator_role():
    with patch.dict(os.environ, _principal_env(None, admin=True), clear=True):
        principal = require_operator("key")
        assert principal is not None
        assert principal.admin is True
