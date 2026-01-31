from __future__ import annotations

from typing import Iterable


def ingest_web(url: str) -> Iterable[str]:
    yield f"[WEB] {url}"

