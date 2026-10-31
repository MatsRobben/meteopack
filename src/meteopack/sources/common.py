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
    """Build a session, or fit out one the caller supplied, with this source's headers.

    Sources must work when constructed with no session — scripts and notebooks
    do that, and passing `None` straight through to the HTTP layer used to
    raise `AttributeError` deep inside a download.

    They must also work when the caller *does* supply one, which is the case
    this got wrong: the CLI builds every source with a bare
    `requests.Session()`, and returning it untouched silently dropped whatever
    the source had meant to put on it. For six sources that cost them their
    User-Agent. For `midas` it dropped the bearer token, so every CLI run
    authenticated as nobody — and because CEDA answers a rejected token with
    HTTP 200 and an HTML login page, that looked like an expired credential for
    as long as anyone cared to look.

    So the headers are applied to whichever session is in hand. A header the
    caller set deliberately is left alone; `requests`' own default User-Agent
    is not treated as deliberate.
    """
    import requests

    session = existing if existing is not None else requests.Session()
    extra = getattr(source, "extra", None) or {}
    agent = extra.get("user_agent") or "meteopack/0.1"
    current = str(session.headers.get("User-Agent", ""))
    if not current or current.startswith("python-requests/"):
        session.headers["User-Agent"] = str(agent)
    return session
