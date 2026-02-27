from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


def fetch_git(url_or_path: str, ref: Optional[str] = None, target_dir: Optional[str] = None) -> str:
    """Clone or open a git repository and return the local path.

    If ``url_or_path`` is already a local directory containing a ``.git``
    folder, it is used directly (optionally checking out *ref*).
    Otherwise the repository is cloned into *target_dir* (or a temp
    directory).

    Requires the ``gitpython`` package.  Returns the path as-is when the
    package is not installed.
    """
    if os.path.isdir(os.path.join(url_or_path, ".git")):
        return _checkout(url_or_path, ref)

    try:
        import git as gitpython
    except ImportError:
        logger.warning("gitpython not installed; returning path as placeholder")
        return url_or_path

    clone_dir = target_dir or tempfile.mkdtemp(prefix="ragbot-repo-")
    logger.info("Cloning %s into %s", url_or_path, clone_dir)
    repo = gitpython.Repo.clone_from(url_or_path, clone_dir)
    if ref:
        repo.git.checkout(ref)
    return clone_dir


def _checkout(path: str, ref: Optional[str]) -> str:
    if not ref:
        return path
    try:
        import git as gitpython
    except ImportError:
        logger.warning("gitpython not installed; skipping checkout")
        return path
    repo = gitpython.Repo(path)
    repo.git.checkout(ref)
    return path
