from __future__ import annotations

import re


def next_version(version: str) -> str:
    """Increment the rightmost numeric segment of a version string.

    Supports formats like ``"1.0.0"``, ``"2.3"``, ``"1"``, ``"v1.2"``.
    Raises ``ValueError`` if no numeric segment is found.

    Examples::

        >>> next_version("1.0.0")
        '1.0.1'
        >>> next_version("v2.3")
        'v2.4'
        >>> next_version("1")
        '2'
    """
    if not version or not version.strip():
        raise ValueError(f"Invalid version string: {version!r}")
    match = re.search(r"(\d+)(?!.*\d)", version)
    if not match:
        raise ValueError(f"No numeric segment found in version: {version!r}")
    start, end = match.start(), match.end()
    current = int(match.group(1))
    return version[:start] + str(current + 1) + version[end:]


def parse_version(version: str) -> tuple[int, ...]:
    """Parse a version string into a tuple of integers for comparison.

    Strips a leading ``v`` prefix if present.

    Examples::

        >>> parse_version("1.2.3")
        (1, 2, 3)
        >>> parse_version("v2.0")
        (2, 0)
    """
    cleaned = version.lstrip("vV").strip()
    if not cleaned:
        return (0,)
    parts = cleaned.split(".")
    result = []
    for part in parts:
        try:
            result.append(int(part))
        except ValueError:
            result.append(0)
    return tuple(result) if result else (0,)


def is_newer(version_a: str, version_b: str) -> bool:
    """Return True if ``version_a`` is strictly newer than ``version_b``."""
    return parse_version(version_a) > parse_version(version_b)
