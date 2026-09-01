from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional
from urllib.parse import urlsplit

from .security import csv_values, validate_local_repo_path, validate_remote_url

logger = logging.getLogger(__name__)


def fetch_git(url_or_path: str, ref: Optional[str] = None, target_dir: Optional[str] = None) -> str:
    """Clone or open a Git repository under the configured source policy.

    Local repositories are restricted to ``RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS``
    in production. Remote repositories use HTTPS and the same SSRF protections
    as Web sources; ``RAGBOT_GIT_ALLOWED_HOSTS`` can further narrow hosts.
    """
    if os.path.isdir(os.path.join(url_or_path, ".git")):
        safe_path = validate_local_repo_path(url_or_path)
        return _checkout(safe_path, ref)

    parsed = urlsplit(url_or_path)
    if parsed.scheme:
        validate_remote_url(
            url_or_path,
            allowed_hosts=csv_values("RAGBOT_GIT_ALLOWED_HOSTS"),
            schemes=("https",),
        )
    elif os.getenv("RAGBOT_ENV", "development").strip().lower() in {"production", "prod"}:
        raise ValueError("Production Git sources must be an allowlisted local repository or HTTPS URL")

    try:
        import git as gitpython
    except ImportError as exc:
        raise RuntimeError("gitpython is required for Git source ingestion") from exc

    clone_dir = target_dir or tempfile.mkdtemp(prefix="ragbot-repo-")
    logger.info("Cloning Git source into %s", clone_dir)
    repo = gitpython.Repo.clone_from(url_or_path, clone_dir)
    if ref:
        repo.git.checkout(ref)
    return clone_dir


def _checkout(path: str, ref: Optional[str]) -> str:
    if not ref:
        return path
    try:
        import git as gitpython
    except ImportError as exc:
        raise RuntimeError("gitpython is required to checkout Git source refs") from exc
    repo = gitpython.Repo(path)
    repo.git.checkout(ref)
    return path
