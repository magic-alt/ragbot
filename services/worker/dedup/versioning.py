from __future__ import annotations


def next_version(version: str) -> str:
    parts = version.split(".")
    if not parts:
        return "1"
    parts[-1] = str(int(parts[-1]) + 1)
    return ".".join(parts)

