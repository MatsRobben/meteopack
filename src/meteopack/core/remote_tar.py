"""Read members of a remote tar with HTTP Range requests, without fetching the tar.

A KNMI HARMONIE P1 run is one ~950 MB tar of 61 GRIB files, one per lead time;
a product that needs 12 of them would otherwise move 80 % of the bytes for
nothing. A tar is a sequence of 512-byte headers each followed by its member's
data padded to 512 bytes, so the member list is a walk: read a header, skip its
data, read the next. Each hop is one small ranged GET (here 1.5 KB, enough for
a PAX extended header, its records and the header after it in one go). The
members then come down with one ranged GET each.

Ranged GETs go to the pre-signed S3 URL, not to the API: they cost nothing
against a KDP key's quota (checked on the live API, see docs/sources).

Only what the HARMONIE tars use is understood: ustar and GNU headers, PAX
``x``/``g`` extended headers (``path`` and ``size`` records) and GNU long
names (``L``). A header that fails its checksum stops the walk with an error.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import requests

from meteopack.core import http
from meteopack.core.http import FetchError, IntegrityError

logger = logging.getLogger(__name__)

BLOCK = 512
#: Bytes read per hop: a PAX header block, up to two blocks of PAX records and
#: the member's own header land in one request.
HOP_BYTES = 3 * BLOCK


@dataclass(frozen=True, slots=True)
class Member:
    name: str
    #: Offset of the member's data (not its header) in the tar.
    offset: int
    size: int


def _octal(field: bytes) -> int:
    if field and field[0] & 0x80:  # GNU base-256 for sizes over 8 GB
        value = int.from_bytes(field[1:], "big")
        return value
    text = field.rstrip(b"\x00 ").strip()
    return int(text, 8) if text else 0


def _checksum_ok(header: bytes) -> bool:
    want = _octal(header[148:156])
    got = sum(header[:148]) + 8 * 32 + sum(header[156:512])
    return want == got


def _pax_records(data: bytes) -> dict[str, str]:
    out: dict[str, str] = {}
    pos = 0
    while pos < len(data):
        space = data.find(b" ", pos)
        if space < 0:
            break
        try:
            n = int(data[pos:space])
        except ValueError:
            break
        if n <= 0:
            break
        record = data[space + 1:pos + n - 1].decode("utf-8", "replace")
        key, _, value = record.partition("=")
        out[key] = value
        pos += n
    return out


def _padded(size: int) -> int:
    return -(-size // BLOCK) * BLOCK


class RangeReader:
    """Ranged GETs on one URL, with retries; counts requests and bytes."""

    def __init__(self, url: str, *, session: requests.Session | None = None,
                 retries: int = 5, backoff: float = 1.0, timeout: float = 60.0,
                 total_size: int | None = None) -> None:
        self.url = url
        self.session = session or requests.Session()
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout
        self.total_size = total_size
        self.requests = 0
        self.bytes = 0

    def read(self, start: int, length: int) -> bytes:
        """``length`` bytes at ``start`` (fewer only at the end of the object)."""
        if length <= 0:
            return b""
        end = start + length - 1
        if self.total_size is not None:
            end = min(end, self.total_size - 1)
        want = end - start + 1
        attempt = 0
        while True:
            self.requests += 1
            err: Exception
            try:
                resp = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                                        timeout=self.timeout)
            except http._TRANSIENT as exc:
                err = exc
            else:
                if resp.status_code == 206:
                    body = resp.content
                    if len(body) == want:
                        self.bytes += len(body)
                        return body
                    err = IntegrityError(f"range {start}-{end}: {len(body)} bytes, "
                                         f"expected {want}")
                elif resp.status_code == 200:
                    # A server that ignores Range would send the whole object.
                    raise FetchError(self.url.split("?")[0], "server ignored the Range header",
                                     200)
                elif resp.status_code in (408, 429) or resp.status_code >= 500:
                    err = FetchError(self.url.split("?")[0], f"HTTP {resp.status_code}",
                                     resp.status_code)
                else:
                    raise FetchError(self.url.split("?")[0], f"HTTP {resp.status_code}",
                                     resp.status_code)
            if attempt >= self.retries:
                if isinstance(err, FetchError):
                    raise err
                raise FetchError(self.url.split("?")[0], f"{type(err).__name__}: {err}", None,
                                 attempts=attempt + 1, error_class=type(err).__name__)
            delay = http._backoff_delay(attempt, self.backoff, None)
            logger.info("range read %d-%d: retrying in %.1f s: %s", start, end, delay, err)
            attempt += 1
            time.sleep(delay)


def list_members(read: Callable[[int, int], bytes], total_size: int,
                 *, stop: Callable[[list[Member]], bool] | None = None) -> list[Member]:
    """Walk the tar's headers through `read(start, length)`; regular files only.

    `stop`, if given, ends the walk early once it returns True for the members
    found so far (a caller that knows how many it needs).
    """
    members: list[Member] = []
    pos = 0
    pending: dict[str, str] = {}
    buf_at, buf = -1, b""

    def window(at: int) -> bytes:
        """HOP_BYTES from `at`, served from the last read when it already holds them
        (a PAX header, its records and the member header arrive in one read)."""
        nonlocal buf_at, buf
        n = min(HOP_BYTES, total_size - at)
        if buf_at >= 0 and buf_at <= at and at + min(n, BLOCK) <= buf_at + len(buf):
            return buf[at - buf_at:]
        buf_at, buf = at, read(at, n)
        return buf

    while pos + BLOCK <= total_size:
        chunk = window(pos)
        header = chunk[:BLOCK]
        if header == b"\x00" * BLOCK:
            break  # end-of-archive marker
        if not _checksum_ok(header):
            raise IntegrityError(f"tar header at offset {pos} fails its checksum")
        typeflag = header[156:157]
        size = _octal(header[124:136])
        data_at = pos + BLOCK
        if typeflag in (b"x", b"g", b"L"):
            need = _padded(size)
            body = chunk[BLOCK:BLOCK + size]
            if len(body) < size:
                body = read(data_at, size)
            if typeflag == b"x":
                pending = _pax_records(body)
            elif typeflag == b"L":
                pending = {"path": body.rstrip(b"\x00").decode("utf-8", "replace")}
            pos = data_at + need
            continue
        name = pending.get("path")
        if name is None:
            prefix = header[345:500].rstrip(b"\x00").decode("utf-8", "replace") \
                if header[257:262] == b"ustar" else ""
            base = header[:100].rstrip(b"\x00").decode("utf-8", "replace")
            name = f"{prefix}/{base}" if prefix else base
        if "size" in pending:
            size = int(pending["size"])
        pending = {}
        if typeflag in (b"0", b"\x00", b"7"):
            members.append(Member(name=name, offset=data_at, size=size))
            if stop is not None and stop(members):
                break
        pos = data_at + _padded(size)
    return members


__all__ = ["BLOCK", "Member", "RangeReader", "list_members"]
