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
import os
import random
import tarfile
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from meteopack.core.config import RateLimit
from meteopack.sources.base import RawFile

_RETRY_STATUS = frozenset({408, 429})
_MAX_RETRY_AFTER = 300.0
_CHUNK = 1 << 20


class FetchError(Exception):
    """A download failed for good (retries exhausted or a non-retryable status)."""

    def __init__(self, url: str, message: str, status: int | None = None) -> None:
        super().__init__(f"{url}: {message}")
        self.url = url
        self.status = status


class NotFoundError(FetchError):
    """HTTP 404. Distinct so callers can skip legitimately absent units."""


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
) -> RawFile:
    """Download `url` to `dest` atomically; a no-op re-hash if `dest` exists.

    The `.part` file is always removed on failure so an interrupted job can
    never leave something that looks like a finished download.
    """
    dest = Path(dest)
    if dest.exists():
        digest, n = _sha256_file(dest)
        return RawFile(path=dest, sha256=digest, n_bytes=n, unit_key=unit_key)

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = _part_path(dest)
    host = urlparse(url).netloc

    attempt = 0
    while True:
        if limiter is not None:
            limiter.acquire(host)
        retry_after: float | None = None
        try:
            digest, n = _download_once(url, part, session, timeout)
        except _HttpStatus as e:
            part.unlink(missing_ok=True)
            if e.status == 404:
                raise NotFoundError(url, "404 not found", 404) from None
            if e.status < 500 and e.status not in _RETRY_STATUS:
                raise FetchError(url, f"HTTP {e.status}", e.status) from None
            retry_after = e.retry_after
            err: Exception = FetchError(url, f"HTTP {e.status}", e.status)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            part.unlink(missing_ok=True)
            err = FetchError(url, f"{type(e).__name__}: {e}")
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        else:
            os.replace(part, dest)
            return RawFile(path=dest, sha256=digest, n_bytes=n, unit_key=unit_key)

        if attempt >= retries:
            raise err
        delay = backoff * (2**attempt) * random.uniform(0.5, 1.5)
        if retry_after is not None:
            delay = max(delay, min(retry_after, _MAX_RETRY_AFTER))
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

    def __enter__(self) -> TarArchiveWriter:
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
