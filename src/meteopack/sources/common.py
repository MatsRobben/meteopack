"""Pieces every source needs, defined once.

Each source was written against the `Source` contract in isolation, which kept
them independent but left ten copies of the same scaffolding — and one of those
copies was a correctness problem rather than untidiness: ten separate
`SchemaDriftError` classes meant `except SchemaDriftError` caught exactly one
source's drift and silently missed the other nine.
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    import requests

    from meteopack.core.config import SourceConfig


class SourceError(RuntimeError):
    """Base for every error a source raises, so callers can catch one thing."""


class SchemaDriftError(SourceError):
    """What arrived does not match the schema pinned in the product config.

    Raised rather than coerced: a reader that guesses at an unfamiliar layout
    writes plausible numbers into a store nobody will re-check. Re-run `scan`,
    look at what moved, and re-pin deliberately.
    """


class CredentialError(SourceError):
    """A credential is missing, expired or rejected. Must name the file or var."""


def fingerprint(names: Iterable[str], *, width: int = 16) -> str:
    """Stable digest of a set of names, independent of order and whitespace.

    Order-independent because a CSV header and a JSON object key set describe
    the same schema regardless of how they are serialised.
    """
    cleaned = sorted({str(n).strip() for n in names if str(n).strip()})
    return hashlib.sha256("\u0000".join(cleaned).encode()).hexdigest()[:width]


def http_session(source: SourceConfig | None, existing: Any = None) -> requests.Session:
    """Return `existing`, or build a session carrying the source's user agent.

    Sources must work when constructed with no session — scripts and notebooks
    do that, and passing `None` straight through to the HTTP layer used to
    raise `AttributeError` deep inside a download.
    """
    if existing is not None:
        return existing
    import requests

    session = requests.Session()
    extra = getattr(source, "extra", None) or {}
    agent = extra.get("user_agent") or "meteopack/0.1"
    session.headers["User-Agent"] = str(agent)
    return session
