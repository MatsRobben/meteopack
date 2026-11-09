"""KNMI Data Platform (KDP) Open Data API: list a dataset, download its files.

The API meters *requests*, not bytes: listing a page costs one request and so
does asking for a file's temporary download URL; the download itself goes to
S3 and costs nothing against the key. What one key allows (developer docs,
checked 2026-11-09):

    registered key   200 requests/s, 1000 requests/hour
    bulk key         100 requests/s, quota = number of files in the dataset
    anonymous        50 requests/min, 3000 requests/hour, shared by everyone

Every API answer carries ``X-Ratelimit-Limit``, ``X-Ratelimit-Remaining`` and
``X-Ratelimit-Reset`` (epoch seconds). :class:`QuotaLimiter` reads them, so the
same code runs at whatever a key allows without being told the quota: it spaces
requests at the configured rate and, when the remaining quota reaches a small
reserve, waits for the reset instead of spending the request on a 429.

Which key is used is configuration, not code. `resolve_key` walks a list of key
*files* and takes the first that exists - by default the bulk key first, then
the regular one - and :meth:`KdpClient.refresh_key` repeats that walk, so a
long-running job picks up a newly placed bulk key at its next day boundary. The
key itself is never logged, printed or put in an exception; only the file it
came from is named.

Listing filters on the filename: ``orderBy=filename&begin=A&end=B`` returns the
files with A <= filename <= B (both ends inclusive, verified), so one day of a
product whose names carry a timestamp is a single request.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from meteopack.core import http
from meteopack.core.http import FetchError, IntegrityError
from meteopack.sources.base import RawFile
from meteopack.sources.common import CredentialError, ListingError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.dataplatform.knmi.nl/open-data/v1"
#: Environment variable naming a key file that wins over the configured list.
KEY_FILE_ENV = "KNMI_API_KEY_FILE"
#: 429s one request may wait out before it counts as a failure.
MAX_THROTTLES = 12
#: The most files one listing page may hold (the API's own maximum).
MAX_KEYS = 1000


@dataclass(frozen=True, slots=True)
class ApiKey:
    """A key and the file it came from. `repr` never shows the key."""

    path: Path
    value: str

    def __repr__(self) -> str:
        return f"ApiKey(path={str(self.path)!r})"

    __str__ = __repr__


def resolve_key(files: Iterable[str | os.PathLike[str]], *,
                env: Mapping[str, str] | None = None) -> ApiKey:
    """The first key file that exists and is non-empty: ``$KNMI_API_KEY_FILE``, then `files`."""
    env = os.environ if env is None else env
    candidates = [env[KEY_FILE_ENV]] if env.get(KEY_FILE_ENV) else []
    candidates += [str(f) for f in files]
    tried = []
    for candidate in candidates:
        path = Path(os.path.expandvars(str(candidate))).expanduser()
        tried.append(str(path))
        try:
            value = path.read_text().strip()
        except OSError:
            continue
        if value:
            return ApiKey(path=path, value=value)
    raise CredentialError(f"no KNMI Data Platform API key found; tried {tried} "
                          f"(set ${KEY_FILE_ENV} or extra.kdp.api_key_files)")


class QuotaLimiter:
    """Request spacing plus the server's own quota, safe across threads.

    `acquire` reserves the next slot at `requests_per_second`; if the quota the
    server last reported (minus what has been spent since) is down to `reserve`,
    it waits until the reported reset first. `observe` feeds it the headers of
    every answer, `throttled` a 429.
    """

    def __init__(self, requests_per_second: float, *, reserve: int = 5,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be > 0")
        self.interval = 1.0 / requests_per_second
        self.reserve = int(reserve)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next = 0.0
        self._remaining: int | None = None
        self._reset: float | None = None
        self._limit: int | None = None
        self.waited_for_quota = 0.0

    def set_rate(self, requests_per_second: float) -> None:
        with self._lock:
            self.interval = 1.0 / requests_per_second

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = self._clock()
                if (self._remaining is not None and self._reset is not None
                        and self._remaining <= self.reserve and now < self._reset):
                    wait = self._reset - now + 1.0
                else:
                    wait = 0.0
                    slot = max(now, self._next)
                    self._next = slot + self.interval
                    if self._remaining is not None:
                        self._remaining -= 1
            if wait > 0:
                logger.info("kdp: quota down to its reserve (%d of %s); waiting %.0f s for "
                            "the reset", self.reserve, self._limit, wait)
                self.waited_for_quota += wait
                self._sleep(wait)
                with self._lock:
                    # Past the reset the server's count is unknown until the
                    # next answer tells us; spend one request to find out.
                    if self._reset is not None and self._clock() >= self._reset:
                        self._remaining = None
                continue
            delay = slot - now
            if delay > 0:
                self._sleep(delay)
            return

    def observe(self, headers: Mapping[str, str]) -> None:
        lowered = {k.lower(): v for k, v in headers.items()}
        try:
            remaining = int(lowered["x-ratelimit-remaining"])
            reset = float(lowered["x-ratelimit-reset"])
        except (KeyError, ValueError):
            return
        with self._lock:
            # Answers arrive out of order across threads; the lower count is
            # the more recent one unless the window has reset.
            if self._reset is None or reset > self._reset or self._remaining is None:
                self._remaining = remaining
            else:
                self._remaining = min(self._remaining, remaining)
            self._reset = reset
            limit = lowered.get("x-ratelimit-limit")
            self._limit = int(limit) if limit and limit.isdigit() else self._limit

    def throttled(self, retry_after: float | None) -> None:
        """A 429: no more requests until the reset (or Retry-After)."""
        with self._lock:
            now = self._clock()
            until = now + (retry_after if retry_after is not None else 60.0)
            if self._reset is not None and self._reset > now:
                until = max(until, self._reset)
            self._remaining = 0
            self._reset = until

    @property
    def state(self) -> dict[str, Any]:
        with self._lock:
            return {"limit": self._limit, "remaining": self._remaining, "reset": self._reset}


@dataclass(frozen=True, slots=True)
class RemoteFile:
    filename: str
    size: int
    last_modified: str | None = None


class KdpClient:
    """One dataset version on the Open Data API.

    `key_files` is walked by `resolve_key`; `rates` maps a key file (as written
    in the config, or its expanded path) to that key's requests/second, with
    `default_rate` for any other.
    """

    def __init__(
        self,
        dataset: str,
        version: str,
        *,
        key_files: Iterable[str],
        base_url: str = DEFAULT_BASE_URL,
        rates: Mapping[str, float] | None = None,
        default_rate: float = 2.0,
        reserve: int = 5,
        retries: int = 6,
        backoff: float = 2.0,
        timeout: float = 60.0,
        session: requests.Session | None = None,
        download_session: requests.Session | None = None,
    ) -> None:
        self.dataset = dataset
        self.version = str(version)
        self.base_url = base_url.rstrip("/")
        self.key_files = [str(f) for f in key_files]
        self.rates = {str(Path(os.path.expandvars(k)).expanduser()): float(v)
                      for k, v in (rates or {}).items()}
        self.default_rate = float(default_rate)
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout
        self.session = session or requests.Session()
        # The S3 download must not carry the API key.
        self.download_session = download_session or requests.Session()
        self.key: ApiKey | None = None
        self.limiter = QuotaLimiter(self.default_rate, reserve=reserve)
        self.requests_made = 0
        self.refresh_key()

    # ------------------------------------------------------------------ keys

    def refresh_key(self) -> bool:
        """Re-resolve the key; True if it changed. Called at every day boundary."""
        key = resolve_key(self.key_files)
        if self.key is not None and key == self.key:
            return False
        changed = self.key is not None
        self.key = key
        self.session.headers["Authorization"] = key.value
        rate = self.rates.get(str(key.path), self.default_rate)
        self.limiter = QuotaLimiter(rate, reserve=self.limiter.reserve)
        logger.info("kdp: %s key from %s at up to %.1f requests/s",
                    "switched to the" if changed else "using the", key.path, rate)
        return True

    # ------------------------------------------------------------------ calls

    @property
    def files_url(self) -> str:
        return f"{self.base_url}/datasets/{self.dataset}/versions/{self.version}/files"

    def _get(self, url: str, params: Mapping[str, Any] | None = None) -> dict:
        attempt = 0
        throttles = 0
        while True:
            self.limiter.acquire()
            self.requests_made += 1
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except http._TRANSIENT as exc:
                err: Exception = exc
                status = None
            else:
                self.limiter.observe(resp.headers)
                status = resp.status_code
                if status < 400:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        err = exc
                elif status in (401, 403):
                    raise CredentialError(
                        f"KDP answered {status} for {url} with the key from "
                        f"{self.key.path if self.key else '?'}: the key is missing, expired "
                        f"or not valid for {self.dataset}")
                elif status == 404:
                    raise http.NotFoundError(url, "404 not found", 404, attempts=attempt + 1)
                elif status == 429:
                    retry_after = http._retry_after(resp.headers.get("Retry-After"))
                    self.limiter.throttled(retry_after)
                    logger.warning("kdp: 429 from %s; holding until the quota resets", url)
                    err = FetchError(url, "HTTP 429", 429, attempts=attempt + 1)
                    # A throttle is not a failure: wait it out without spending
                    # a retry, or a burst of 429s at the hour mark would fail a
                    # day. Bounded, so a key that is throttled for good fails.
                    throttles += 1
                    if throttles <= MAX_THROTTLES:
                        continue
                elif status < 500 and status != 408:
                    raise FetchError(url, f"HTTP {status}", status, attempts=attempt + 1)
                else:
                    err = FetchError(url, f"HTTP {status}", status, attempts=attempt + 1)
            if attempt >= self.retries:
                if isinstance(err, FetchError):
                    raise err
                raise FetchError(url, f"{type(err).__name__}: {err}", status,
                                 attempts=attempt + 1, error_class=type(err).__name__)
            delay = http._backoff_delay(attempt, self.backoff, None)
            logger.info("kdp: retrying %s in %.1f s: %s", url, delay, err)
            attempt += 1
            time.sleep(delay)

    def list_files(self, begin: str | None = None, end: str | None = None) -> list[RemoteFile]:
        """Every file with ``begin <= filename <= end``, in filename order, all pages."""
        params: dict[str, Any] = {"maxKeys": MAX_KEYS, "orderBy": "filename", "sorting": "asc"}
        if begin:
            params["begin"] = begin
        if end:
            params["end"] = end
        out: list[RemoteFile] = []
        while True:
            try:
                page = self._get(self.files_url, params)
            except FetchError as exc:
                raise ListingError(f"KDP listing of {self.dataset} {begin}..{end} failed: "
                                   f"{exc}") from exc
            for item in page.get("files") or []:
                out.append(RemoteFile(filename=str(item["filename"]), size=int(item["size"]),
                                      last_modified=item.get("lastModified")))
            token = page.get("nextPageToken")
            if not token or not page.get("isTruncated", True):
                return out
            params = {**params, "nextPageToken": token}

    def download_url(self, filename: str) -> str:
        page = self._get(f"{self.files_url}/{filename}/url")
        try:
            return str(page["temporaryDownloadUrl"])
        except KeyError:
            raise FetchError(filename, f"no temporaryDownloadUrl in {sorted(page)}") from None

    def download(self, remote: RemoteFile, dest: Path, *, attempts: int = 3) -> RawFile:
        """Download one file atomically, checking its size against the listing.

        A file already at `dest` with the listed size is kept (costing no
        request), which is what makes an interrupted day cheap to resume. The
        temporary URL is valid for an hour; if it lapses or S3 refuses it, a
        fresh one is requested, `attempts` times in all.
        """
        def check(path: Path):
            n = path.stat().st_size
            if n != remote.size:
                raise IntegrityError(f"{path.name}: {n} bytes, listing says {remote.size}")
            with path.open("rb") as f:
                if f.read(8) != b"\x89HDF\r\n\x1a\n":
                    raise IntegrityError(f"{path.name}: not an HDF5/NetCDF4 file")

        dest = Path(dest)
        if dest.exists():
            try:
                check(dest)
            except IntegrityError as exc:
                logger.warning("kdp: %s; downloading it again", exc)
            else:
                return RawFile(path=dest, sha256="", n_bytes=remote.size,
                               unit_key=remote.filename, attempts=0)
        last: Exception | None = None
        for _ in range(attempts):
            url = self.download_url(remote.filename)
            try:
                return http.fetch_file(url, dest, unit_key=remote.filename,
                                       session=self.download_session, limiter=None,
                                       retries=self.retries, backoff=self.backoff,
                                       timeout=self.timeout, overwrite=True, check=check)
            except FetchError as exc:
                last = exc
                if exc.status is not None and exc.status not in (403, 400):
                    break
        assert last is not None
        raise last
