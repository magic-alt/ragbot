from __future__ import annotations

import unicodedata
from typing import Any, Iterable


# Identifier matching is deliberately stricter and separate from ordinary
# term matching. We normalize only separator variants that commonly differ
# across source text, PDF extraction, and user queries:
#
# - Unicode dash punctuation (including U+2011 NON-BREAKING HYPHEN)
# - ASCII hyphen
# - underscore
# - Unicode minus sign
# - soft hyphen
# - whitespace / PDF line wrapping
#
# Other punctuation is preserved so this helper does not silently broaden
# normal text relevance semantics.
_EXTRA_SEPARATORS = {"_", "\u00ad", "\u2212"}


def canonicalize_identifier(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    output: list[str] = []
    for char in text:
        if char.isspace() or char in _EXTRA_SEPARATORS or unicodedata.category(char) == "Pd":
            continue
        output.append(char)
    return "".join(output)


def identifier_matches(text: Any, identifier: Any) -> bool:
    needle = canonicalize_identifier(identifier)
    if not needle:
        return False
    return needle in canonicalize_identifier(text)


def any_identifier_matches(text: Any, identifiers: Iterable[Any]) -> bool:
    canonical_text = canonicalize_identifier(text)
    if not canonical_text:
        return False
    for identifier in identifiers:
        needle = canonicalize_identifier(identifier)
        if needle and needle in canonical_text:
            return True
    return False
