"""Anonymous S3 over plain HTTPS: list a prefix, download whole objects.

For public buckets that need no credentials - the EUMETNET OPERA archive
(``https://s3.waw3-1.cloudferro.com/openradar-archive/``) is one - so no SDK and
no signing: a listing is ``GET <bucket>/?list-type=2&prefix=...`` (ListObjectsV2,
up to 1000 keys a page, continued with ``continuation-token``) and a download is
``GET <bucket>/<key>``.

:class:`S3Client` speaks the same small interface as
:class:`meteopack.core.kdp.KdpClient` where the day-by-day stream needs it
(`key`, `refresh_key`, `requests_made`, `limiter.state`, `stats`, `download`),
so `meteopack.jobs.stream_kdp` drives either. Downloads go through
:func:`meteopack.core.http.fetch_file` - temp file, atomic rename, retries with
backoff and jitter - and every file is checked before it is published: its size
against the listing, the HDF5 signature when asked, and its MD5 against the
listing's ETag where that ETag is a plain MD5 (not a multipart upload's).
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from meteopack.core import http
from meteopack.core.http import FetchError, IntegrityError
from meteopack.sources.base import RawFile
from meteopack.sources.common import ListingError

logger = logging.getLogger(__name__)

_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """One object: `filename` is the key's last component, `key` the whole key."""

    filename: str
    size: int
    last_modified: str | None = None
    key: str = ""
    etag: str = ""


class _Anonymous:
    """What the stream logs as "the key in use"."""

    path = "anonymous"

    def __repr__(self) -> str:
        return "anonymous"


class _Pacer:
    """Requests/second across threads, with the `state` the stream logs."""

    def __init__(self, requests_per_second: float) -> None:
        self.interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()
        self.reserve = 0

    def acquire(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        if slot > now:
            time.sleep(slot - now)

    @property
    def state(self) -> dict[str, Any]:
        return {"anonymous": True}


def parse_listing(xml: bytes | str) -> tuple[list[RemoteFile], str | None]:
    """One ListObjectsV2 page -> (objects, continuation token or None)."""
    root = ET.fromstring(xml)
    out = []
    for item in root.iter(f"{_NS}Contents"):
        key = item.findtext(f"{_NS}Key") or ""
        out.append(RemoteFile(
            filename=key.rsplit("/", 1)[-1],
            size=int(item.findtext(f"{_NS}Size") or 0),
            last_modified=item.findtext(f"{_NS}LastModified"),
            key=key,
            etag=(item.findtext(f"{_NS}ETag") or "").strip('"'),
        ))
    truncated = (root.findtext(f"{_NS}IsTruncated") or "false").lower() == "true"
    token = root.findtext(f"{_NS}NextContinuationToken") if truncated else None
    return out, token


class S3Client:
    """A public bucket: `endpoint` like ``https://s3.waw3-1.cloudferro.com``."""

    def __init__(self, endpoint: str, bucket: str, *, requests_per_second: float = 50.0,
                 retries: int = 5, backoff: float = 2.0, timeout: float = 60.0,
                 check_hdf5: bool = True, session: requests.Session | None = None) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.bucket = bucket.strip("/")
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout
        self.check_hdf5 = check_hdf5
        if session is None:
            session = requests.Session()
            # Enough pooled connections for every download thread, so 8 workers
            # reuse 8 TLS connections instead of opening one per file.
            adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=32)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
        self.session = session
        self.key = _Anonymous()
        self.limiter = _Pacer(requests_per_second)
        self.requests_made = 0
        self.stats = {"url_s": 0.0, "transfer_s": 0.0, "files": 0, "bytes": 0}
        self._lock = threading.Lock()

    @property
    def bucket_url(self) -> str:
        return f"{self.endpoint}/{self.bucket}"

    def refresh_key(self) -> bool:
        return False

    def _count(self) -> None:
        with self._lock:
            self.requests_made += 1

    def list_prefix(self, prefix: str) -> list[RemoteFile]:
        """Every object under `prefix`, all pages, in key order."""
        out: list[RemoteFile] = []
        token: str | None = None
        while True:
            url = f"{self.bucket_url}/?list-type=2&max-keys=1000&prefix={quote(prefix)}"
            if token:
                url += f"&continuation-token={quote(token)}"
            self.limiter.acquire()
            self._count()
            try:
                resp = http.get_with_retry(url, session=self.session, retries=self.retries,
                                           backoff=self.backoff, timeout=self.timeout)
                page, token = parse_listing(resp.content)
            except (FetchError, ET.ParseError) as exc:
                raise ListingError(f"S3 listing of {self.bucket}/{prefix} failed: {exc}") from exc
            out.extend(page)
            if not token:
                return sorted(out, key=lambda r: r.key)

    def object_url(self, key: str) -> str:
        return f"{self.bucket_url}/{quote(key)}"

    def _check(self, remote: RemoteFile):
        def check(path: Path):
            n = path.stat().st_size
            if n != remote.size:
                raise IntegrityError(f"{path.name}: {n} bytes, listing says {remote.size}")
            if self.check_hdf5:
                with path.open("rb") as f:
                    if f.read(8) != HDF5_SIGNATURE:
                        raise IntegrityError(f"{path.name}: not an HDF5 file")
            etag = remote.etag
            if etag and "-" not in etag and len(etag) == 32:
                md5 = hashlib.md5(usedforsecurity=False)
                with path.open("rb") as f:
                    while chunk := f.read(1 << 20):
                        md5.update(chunk)
                if md5.hexdigest() != etag:
                    raise IntegrityError(f"{path.name}: MD5 {md5.hexdigest()} != ETag {etag}")
        return check

    def download(self, remote: RemoteFile, dest: Path) -> RawFile:
        """Download one object atomically; a file already at `dest` that checks out is kept."""
        dest = Path(dest)
        check = self._check(remote)
        if dest.exists():
            try:
                check(dest)
            except IntegrityError as exc:
                logger.warning("s3: %s; downloading it again", exc)
            else:
                return RawFile(path=dest, sha256="", n_bytes=remote.size,
                               unit_key=remote.filename, attempts=0)
        self.limiter.acquire()
        self._count()
        t0 = time.perf_counter()
        raw = http.fetch_file(self.object_url(remote.key or remote.filename), dest,
                              unit_key=remote.filename, session=self.session, limiter=None,
                              retries=self.retries, backoff=self.backoff,
                              timeout=self.timeout, overwrite=True, check=check)
        with self._lock:
            self.requests_made += max(0, raw.attempts - 1)
            self.stats["transfer_s"] += time.perf_counter() - t0
            self.stats["files"] += 1
            self.stats["bytes"] += raw.n_bytes
        return raw


__all__ = ["RemoteFile", "S3Client", "parse_listing"]
