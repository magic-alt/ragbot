from __future__ import annotations

from typing import Iterable


def ingest_repo(path: str) -> Iterable[str]:
    yield f"[REPO] {path}"

