from __future__ import annotations

import re

_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+")


def contains_cjk(text: str) -> bool:
    return bool(_CJK_RUN_RE.search(text))


def lexicalize(text: str) -> str:
    """Build stable lexemes for PostgreSQL ``simple`` FTS.

    English/number tokens are preserved as lowercase terms. Contiguous CJK
    runs are expanded into overlapping character bigrams, with a unigram for
    one-character runs. This avoids requiring a PostgreSQL extension while
    materially improving Chinese lexical recall over treating a whole sentence
    as one token.
    """
    terms: list[str] = [token.lower() for token in _ASCII_TOKEN_RE.findall(text)]
    for match in _CJK_RUN_RE.finditer(text):
        run = match.group(0)
        if len(run) == 1:
            terms.append(run)
        else:
            terms.extend(run[index : index + 2] for index in range(len(run) - 1))
    return " ".join(terms)


def build_or_tsquery(text: str) -> str:
    """Return a safe OR tsquery for lexemes produced by :func:`lexicalize`."""
    terms = lexicalize(text).split()
    # lexicalize only emits alnum/_/- ASCII tokens or CJK characters, so terms
    # can be quoted safely for to_tsquery. De-duplicate while retaining order.
    unique = list(dict.fromkeys(terms))
    return " | ".join(_quote_tsquery_term(term) for term in unique)


def _quote_tsquery_term(term: str) -> str:
    return "'" + term.replace("'", "''") + "'"
