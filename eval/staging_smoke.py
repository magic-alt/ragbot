"""End-to-end staging smoke for a production-style Ragbot deployment.

The caller is responsible for starting PostgreSQL, Qdrant, the API and the
independent ingestion worker with a real OpenAI-compatible LLM/embedding
credential. This module exercises all advertised ingestion source types plus
hybrid search, Agentic chat and an ACL negative case.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests

from services.api.app.auth.acl import build_policy
from services.api.app.storage.pg_repo import PostgresRepo

API_BASE = os.getenv("STAGING_API_BASE", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("STAGING_API_KEY", "staging-ci-key")
TENANT = os.getenv("STAGING_TENANT", "staging")
USER = os.getenv("STAGING_USER", "staging-user")
TIMEOUT_SECONDS = int(os.getenv("STAGING_JOB_TIMEOUT_SECONDS", "180"))


def main() -> int:
    _wait_ready()
    root = Path(os.getenv("STAGING_SOURCE_ROOT", "/tmp/ragbot-staging")).resolve()
    root.mkdir(parents=True, exist_ok=True)

    local_dir = root / "local"
    local_dir.mkdir(exist_ok=True)
    (local_dir / "knowledge.txt").write_text(
        "Ragbot staging sentinel: orbital-servo-731. "
        "EtherCAT总线控制伺服驱动器需要稳定的同步周期。",
        encoding="utf-8",
    )

    repo_dir = root / "tiny-repo"
    _prepare_git_repo(repo_dir)

    sources = [
        ("local_fs", "staging-local", {"path": str(local_dir)}),
        ("web", "staging-web", {"url": os.getenv("STAGING_WEB_URL", "https://example.com")}),
        (
            "pdf",
            "staging-pdf",
            {
                "path": os.getenv(
                    "STAGING_PDF_URL",
                    "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                )
            },
        ),
        ("repo", "staging-repo", {"path": str(repo_dir), "ref": "HEAD"}),
    ]

    job_ids: list[str] = []
    for source_type, name, config in sources:
        source = _request(
            "POST",
            "/sources",
            json={
                "tenant_id": TENANT,
                "source_type": source_type,
                "name": name,
                "config": config,
                "tags": ["staging", source_type],
            },
            expected=201,
        ).json()
        job = _request(
            "POST",
            "/ingest/jobs",
            json={"source_id": source["source_id"], "tenant_id": TENANT},
            expected=202,
        ).json()
        job_ids.append(job["job_id"])

    for job_id in job_ids:
        _wait_job(job_id)

    search = _request(
        "POST",
        "/search",
        json={
            "query": "orbital-servo-731 伺服驱动",
            "tenant_id": TENANT,
            "user_id": USER,
            "top_k": 10,
        },
    ).json()
    if not any("orbital-servo-731" in chunk.get("text", "") for chunk in search["chunks"]):
        raise AssertionError("Hybrid search did not retrieve the local staging sentinel")

    chat = _request(
        "POST",
        "/chat",
        json={
            "query": "根据知识库说明 orbital-servo-731 与什么主题相关？",
            "tenant_id": TENANT,
            "user_id": USER,
        },
    ).json()
    if not chat.get("answer"):
        raise AssertionError("Agentic /chat returned an empty answer")

    _assert_acl_negative(root)
    print("staging smoke passed: 4 source types + hybrid search + agent chat + ACL negative")
    return 0


def _assert_acl_negative(root: Path) -> None:
    dsn = os.environ["POSTGRES_DSN"]
    repo = PostgresRepo(dsn, pool_min=1, pool_max=2)
    policy_id = f"staging-deny-{uuid.uuid4().hex}"
    try:
        repo.add_policy(build_policy(policy_id, TENANT, {"allow_users": ["different-user"]}))
    finally:
        repo.close()

    restricted_dir = root / "restricted"
    restricted_dir.mkdir(exist_ok=True)
    sentinel = f"restricted-sentinel-{uuid.uuid4().hex}"
    (restricted_dir / "secret.txt").write_text(sentinel, encoding="utf-8")
    source = _request(
        "POST",
        "/sources",
        json={
            "tenant_id": TENANT,
            "source_type": "local_fs",
            "name": "staging-restricted",
            "config": {"path": str(restricted_dir)},
            "acl_policy_id": policy_id,
            "tags": ["staging", "restricted"],
        },
        expected=201,
    ).json()
    job = _request(
        "POST",
        "/ingest/jobs",
        json={"source_id": source["source_id"], "tenant_id": TENANT},
        expected=202,
    ).json()
    _wait_job(job["job_id"])

    result = _request(
        "POST",
        "/search",
        json={"query": sentinel, "tenant_id": TENANT, "user_id": USER, "top_k": 10},
    ).json()
    if any(sentinel in chunk.get("text", "") for chunk in result["chunks"]):
        raise AssertionError("ACL negative test leaked a restricted chunk")


def _prepare_git_repo(path: Path) -> None:
    path.mkdir(exist_ok=True)
    (path / "servo.py").write_text(
        "def ethercat_servo_control():\n"
        "    return 'staging git sentinel: cia402-sync-421'\n",
        encoding="utf-8",
    )
    if not (path / ".git").exists():
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "config", "user.email", "staging@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(path), "config", "user.name", "Ragbot Staging"], check=True)
    subprocess.run(["git", "-C", str(path), "add", "servo.py"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "staging fixture", "--allow-empty"], check=True)


def _wait_ready() -> None:
    deadline = time.monotonic() + 60
    last_error: Any = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{API_BASE}/admin/ready", timeout=5)
            if response.status_code == 200:
                return
            last_error = response.text
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"API did not become ready: {last_error}")


def _wait_job(job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        job = _request("GET", f"/ingest/jobs/{job_id}").json()
        if job["status"] == "completed":
            return job
        if job["status"] == "failed":
            raise RuntimeError(f"Ingestion job failed: {job_id}: {job.get('error')}")
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for ingestion job: {job_id}")


def _request(method: str, path: str, *, expected: int = 200, **kwargs: Any) -> requests.Response:
    headers = dict(kwargs.pop("headers", {}) or {})
    headers["X-API-Key"] = API_KEY
    response = requests.request(method, f"{API_BASE}{path}", headers=headers, timeout=60, **kwargs)
    if response.status_code != expected:
        raise RuntimeError(
            f"{method} {path} returned {response.status_code}, expected {expected}: {response.text[:500]}"
        )
    return response


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
