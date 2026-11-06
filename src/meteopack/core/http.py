"""Rate-limited, retrying, atomic downloads; listing cache; per-bucket tar writer.

Shared by every downloader. Upstreams meter on request count, not bytes, and
RMI-style servers punish the whole egress IP for concurrent hammering, so
correctness under concurrency matters more than throughput here.
"""
from __future__ import annotations

import email.utils
import hashlib
import io
import json
import logging
import os
import random
import tarfile
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlparse

import requests

from meteopack.core.config import RateLimit
from meteopack.sources.base import Content, RawFile

logger = logging.getLogger(__name__)

_RETRY_STATUS = frozenset({408, 429})
#: Statuses that mean the server as a whole wants less from us. On these the
#: whole host is paused, not just the request that got the answer: one thread
#: backing off while seven others carry on is not backing off.
_THROTTLE_STATUS = frozenset({429, 503})
_MAX_RETRY_AFTER = 300.0
_CHUNK = 1 << 20

#: Transport failures worth another attempt. Everything here is something a
#: second request can plausibly not hit: a reset connection, a read that timed
#: out, a body cut short (urllib3 raises that as ChunkedEncodingError or, with
#: a Content-Length, as our own short-body check), a gzip stream cut mid-way.
_TRANSIENT = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ContentDecodingError,
)


class FetchError(Exception):
    """A download failed for good (retries exhausted or a non-retryable status).

    Carries what the ledger records about a failure, so that "failed" is never
    the whole story: the HTTP status if there was one, the class of the last
    error (`HTTPError`, `ReadTimeout`, `IntegrityError`, ...) and how many
    requests were made before giving up.
    """

    def __init__(
        self,
        url: str,
        message: str,
        status: int | None = None,
        *,
        attempts: int = 1,
        error_class: str | None = None,
    ) -> None:
        super().__init__(f"{url}: {message}")
        self.url = url
        self.status = status
        self.attempts = attempts
        self.error_class = error_class or ("HTTPError" if status is not None else "FetchError")
        self.message = message


class NotFoundError(FetchError):
    """HTTP 404. Distinct so callers can skip legitimately absent units."""


class IntegrityError(Exception):
    """A complete download that is not what it should be.

    Raised by a source's check (`fetch_file(check=...)`): a ZIP that does not
    open or fails its CRC, a CSV cut off mid-row, an HTML error page served
    with status 200. Treated as transient - the bytes are re-requested - since
    the usual cause is a connection that died without the server noticing.
    """


#: What a source's integrity check looks like: read the finished file, raise
#: IntegrityError if it is broken, otherwise say what it holds.
Check = Callable[[Path], Content | None]


class RateLimiter:
    """Per-host requests/second, safe across threads.

    Each caller reserves the next free slot under the lock and sleeps outside
    it, so concurrent callers are spaced `1/rps` apart rather than released as a
    burst when one sleep ends.
    """

    def __init__(
        self,
        requests_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be > 0")
        self._interval = 1.0 / requests_per_second
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next: dict[str, float] = {}

    @classmethod
    def from_config(cls, cfg: RateLimit) -> RateLimiter:
        return cls(cfg.requests_per_second)

    def acquire(self, host: str) -> None:
        with self._lock:
            now = self._clock()
            slot = max(now, self._next.get(host, now))
            self._next[host] = slot + self._interval
        delay = slot - now
        if delay > 0:
            self._sleep(delay)

    def pause(self, host: str, seconds: float) -> None:
        """Hold every caller for `host` back for `seconds` from now.

        Called when the server throttles (429, 503, or a source's configured
        throttle status): the next slot moves past the pause, and callers
        queue behind it at the usual spacing instead of resuming as a burst.
        """
        with self._lock:
            now = self._clock()
            self._next[host] = max(self._next.get(host, now), now + seconds)


def _part_path(dest: Path) -> Path:
    """A temp path private to this writer.

    `.part` + `os.replace` is only atomic if no two writers share the temp name.
    Slurm array tasks can legitimately target the same destination (an overlap
    in sharding, a re-run racing a straggler), and with a fixed `.part` name
    they would interleave writes into one file and then publish the mangled
    result. Tagging with pid + random makes the loser's write a harmless
    discard and leaves `os.replace` as plain last-writer-wins.
    """
    return dest.with_suffix(f"{dest.suffix}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part")


#: A `.part` file not written to for this long belongs to a dead writer: a
#: live download touches it every chunk, and gives up after `timeout` seconds
#: of silence.
STALE_PART_SECONDS = 3600.0


def remove_stale_parts(root: Path, *, older_than: float = STALE_PART_SECONDS,
                       now: float | None = None) -> int:
    """Delete `.part` files under `root` left by writers that were killed.

    `.part` + rename means a killed job never publishes a half file, but
    SIGKILL (Slurm's follow-up to SIGTERM, an OOM kill) gives no chance to
    remove the `.part` itself. Nothing reads them - `raw_files` skips the
    suffix - so they only cost space; they are removed by age rather than by
    owner, because the owner may be another node's array task still writing.
    """
    root = Path(root)
    if not root.is_dir():
        return 0
    cutoff = (time.time() if now is None else now) - older_than
    removed = 0
    for path in root.rglob("*.part"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def _sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


def _download_once(
    url: str, part: Path, session: requests.Session, timeout: float
) -> tuple[str, int]:
    """Stream to `part` hashing as we go. Raises on HTTP errors; cleans nothing up."""
    with session.get(url, stream=True, timeout=timeout) as resp:
        if resp.status_code >= 400:
            raise _HttpStatus(resp.status_code, _retry_after(resp.headers.get("Retry-After")))
        expected = resp.headers.get("Content-Length")
        h = hashlib.sha256()
        n = 0
        with part.open("wb") as f:
            for chunk in resp.iter_content(_CHUNK):
                f.write(chunk)
                h.update(chunk)
                n += len(chunk)
            f.flush()
            os.fsync(f.fileno())
    if expected is not None and resp.headers.get("Content-Encoding") is None and n != int(expected):
        raise requests.exceptions.ChunkedEncodingError(
            f"short body: got {n} of {expected} bytes"
        )
    return h.hexdigest(), n


class _HttpStatus(Exception):
    def __init__(self, status: int, retry_after: float | None) -> None:
        super().__init__(status)
        self.status = status
        self.retry_after = retry_after


def _backoff_delay(attempt: int, backoff: float, retry_after: float | None) -> float:
    """Exponential backoff with jitter; a server's Retry-After is a floor.

    Jitter matters because a Slurm array's tasks fail together (one upstream
    hiccup) and would otherwise retry in lockstep. Retry-After is capped so a
    misbehaving server cannot park a job for hours.
    """
    delay = backoff * (2**attempt) * random.uniform(0.5, 1.5)
    if retry_after is not None:
        delay = max(delay, min(retry_after, _MAX_RETRY_AFTER))
    return delay


def fetch_file(
    url: str,
    dest: Path,
    *,
    unit_key: str,
    session: requests.Session,
    limiter: RateLimiter | None = None,
    retries: int = 5,
    backoff: float = 1.0,
    timeout: float = 60.0,
    overwrite: bool = False,
    check: Check | None = None,
    retry_status: Iterable[int] = (),
) -> RawFile:
    """Download `url` to `dest` atomically; a no-op re-hash if `dest` exists.

    `overwrite` exists for volatile units — archives rewritten upstream under an
    unchanged filename. Without it the ledger re-queues them and this function
    then skips because the file is there, so the ledger refreshes and the bytes
    never do: the live tail silently freezes while everything looks healthy.

    `check` is the source's integrity check, run on the `.part` file before it
    is published: a file that fails it is never moved into place, and is
    downloaded again like any other transient failure. It also runs on a file
    already at `dest`, so a broken file left by an older fetcher (or by hand)
    is replaced rather than trusted because it exists. What the check returns
    rides back on the RawFile as `content`.

    Retried: connection errors, timeouts, truncated bodies, 408, 429, 5xx,
    the source's `retry_status` and integrity failures, `retries` times with
    exponential backoff, honouring Retry-After. A throttling answer (429, 503,
    `retry_status`) pauses the whole host in `limiter`, not just this request.
    Not retried: 404 (NotFoundError - absent upstream) and other 4xx. Every
    final failure is a FetchError saying how many attempts it took.

    The `.part` file is always removed on failure so an interrupted job can
    never leave something that looks like a finished download.
    """
    dest = Path(dest)
    if dest.exists() and not overwrite:
        digest, n = _sha256_file(dest)
        try:
            content = check(dest) if check is not None else None
        except IntegrityError as exc:
            logger.warning("%s exists but fails its check (%s); downloading it again", dest, exc)
        else:
            return RawFile(path=dest, sha256=digest, n_bytes=n, unit_key=unit_key,
                           content=content, attempts=0)

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = _part_path(dest)
    host = urlparse(url).netloc
    extra = frozenset(retry_status)

    attempt = 0
    while True:
        if limiter is not None:
            limiter.acquire(host)
        retry_after: float | None = None
        throttled = False
        try:
            digest, n = _download_once(url, part, session, timeout)
            content = check(part) if check is not None else None
        except _HttpStatus as e:
            part.unlink(missing_ok=True)
            if e.status == 404:
                raise NotFoundError(url, "404 not found", 404, attempts=attempt + 1) from None
            if e.status < 500 and e.status not in _RETRY_STATUS | extra:
                raise FetchError(url, f"HTTP {e.status}", e.status, attempts=attempt + 1) from None
            retry_after = e.retry_after
            err = FetchError(url, f"HTTP {e.status}", e.status, attempts=attempt + 1)
            throttled = e.status in _THROTTLE_STATUS | extra
        except _TRANSIENT as e:
            part.unlink(missing_ok=True)
            err = FetchError(url, f"{type(e).__name__}: {e}", attempts=attempt + 1,
                             error_class=type(e).__name__)
        except IntegrityError as e:
            part.unlink(missing_ok=True)
            err = FetchError(url, f"IntegrityError: {e}", attempts=attempt + 1,
                             error_class="IntegrityError")
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        else:
            os.replace(part, dest)
            return RawFile(path=dest, sha256=digest, n_bytes=n, unit_key=unit_key,
                           content=content, attempts=attempt + 1)

        if attempt >= retries:
            raise err
        delay = _backoff_delay(attempt, backoff, retry_after)
        if limiter is not None and throttled:
            limiter.pause(host, delay)
        logger.info("retrying %s in %.1f s (attempt %d of %d): %s",
                    url, delay, attempt + 2, retries + 1, err.message)
        attempt += 1
        time.sleep(delay)


def get_with_retry(
    url: str,
    *,
    session: requests.Session,
    limiter: RateLimiter | None = None,
    retries: int = 5,
    backoff: float = 1.0,
    timeout: float = 60.0,
    retry_status: Iterable[int] = (),
) -> requests.Response:
    """GET a small document - a directory listing, a station catalogue - with
    the same retry policy as `fetch_file`.

    Listings used to be a single bare GET whose failure was logged and turned
    into an empty list. A timeout on DWD's `historical/` listing then meant
    every closed station's archive was simply never planned, `fetch` reported
    nothing to do, and the gap was invisible. Retry, then raise.
    """
    host = urlparse(url).netloc
    extra = frozenset(retry_status)
    attempt = 0
    while True:
        if limiter is not None:
            limiter.acquire(host)
        retry_after: float | None = None
        throttled = False
        try:
            resp = session.get(url, timeout=timeout)
            resp.content  # noqa: B018 - read the body inside the retry
        except _TRANSIENT as e:
            err = FetchError(url, f"{type(e).__name__}: {e}", attempts=attempt + 1,
                             error_class=type(e).__name__)
        else:
            status = resp.status_code
            if status < 400:
                return resp
            if status == 404:
                raise NotFoundError(url, "404 not found", 404, attempts=attempt + 1)
            if status < 500 and status not in _RETRY_STATUS | extra:
                raise FetchError(url, f"HTTP {status}", status, attempts=attempt + 1)
            retry_after = _retry_after(resp.headers.get("Retry-After"))
            err = FetchError(url, f"HTTP {status}", status, attempts=attempt + 1)
            throttled = status in _THROTTLE_STATUS | extra
        if attempt >= retries:
            raise err
        delay = _backoff_delay(attempt, backoff, retry_after)
        if limiter is not None and throttled:
            limiter.pause(host, delay)
        logger.info("retrying %s in %.1f s: %s", url, delay, err.message)
        attempt += 1
        time.sleep(delay)


class ListingCache:
    """JSON-on-disk cache with a TTL, for slow remote directory/API listings."""

    def __init__(
        self, path: Path, ttl_seconds: float, *, clock: Callable[[], float] = time.time
    ) -> None:
        self.path = Path(path)
        self.ttl_seconds = ttl_seconds
        self._clock = clock

    def get(self) -> Any | None:
        """Cached value, or None if missing, expired or unreadable."""
        try:
            blob = json.loads(self.path.read_text())
            if self._clock() - float(blob["stored_at"]) > self.ttl_seconds:
                return None
            return blob["value"]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def set(self, value: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = _part_path(self.path)
        tmp.write_text(json.dumps({"stored_at": self._clock(), "value": value}))
        os.replace(tmp, self.path)


def listing_cache(
    path: Path, ttl_seconds: float, *, clock: Callable[[], float] = time.time
) -> ListingCache:
    return ListingCache(path, ttl_seconds, clock=clock)


class TarArchiveWriter:
    """Collects members into one tar per time bucket, published atomically.

    Members go to `<path>.part` and the tar only appears at `path` on a clean
    exit, so a killed job never leaves a truncated tar that looks complete. If
    `path` already exists its members are carried over first, making re-runs of a
    bucket additive (that costs a copy of the existing tar). Names already
    present are skipped, which keeps re-runs idempotent.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._part = _part_path(self.path)
        self._tar: tarfile.TarFile | None = None
        self._names: set[str] = set()

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tar = tarfile.open(self._part, "w")
        try:
            if self.path.exists():
                with tarfile.open(self.path, "r") as old:
                    for m in old:
                        if m.isfile():
                            self._tar.addfile(m, old.extractfile(m))
                            self._names.add(m.name)
        except BaseException:
            self._abort()
            raise
        return self

    def add_bytes(self, name: str, data: bytes, *, mtime: float | None = None) -> bool:
        """Add a member; False if `name` was already present."""
        assert self._tar is not None, "use as a context manager"
        if name in self._names:
            return False
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mtime = int(time.time() if mtime is None else mtime)
        self._tar.addfile(info, io.BytesIO(data))
        self._names.add(name)
        return True

    def __contains__(self, name: str) -> bool:
        return name in self._names

    def _abort(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None
        self._part.unlink(missing_ok=True)

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._abort()
            return
        assert self._tar is not None
        self._tar.close()
        self._tar = None
        with self._part.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(self._part, self.path)


def read_member(tar_path: Path, name: str) -> bytes:
    """Bytes of one member; KeyError if absent."""
    with tarfile.open(tar_path, "r") as tar:
        f = tar.extractfile(name)
        if f is None:
            raise KeyError(name)
        return f.read()


def list_members(tar_path: Path) -> list[str]:
    with tarfile.open(tar_path, "r") as tar:
        return [m.name for m in tar if m.isfile()]
