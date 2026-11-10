"""Anonymous S3: ListObjectsV2 paging and checked downloads, against a fake session."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import ClassVar

import pytest

from meteopack.core.http import FetchError
from meteopack.core.s3 import HDF5_SIGNATURE, RemoteFile, S3Client, parse_listing
from meteopack.sources.common import ListingError

NS = 'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"'


def page(keys, token=None):
    items = "".join(
        f"<Contents><Key>{k}</Key><LastModified>2026-02-06T10:00:00.000Z</LastModified>"
        f"<ETag>&quot;{e}&quot;</ETag><Size>{n}</Size></Contents>" for k, n, e in keys)
    trunc = (f"<IsTruncated>true</IsTruncated><NextContinuationToken>{token}"
             f"</NextContinuationToken>" if token else "<IsTruncated>false</IsTruncated>")
    return f'<?xml version="1.0"?><ListBucketResult {NS}>{trunc}{items}</ListBucketResult>'.encode()


class Resp:
    headers: ClassVar[dict] = {}

    def __init__(self, body: bytes, status: int = 200):
        self.content = body
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_content(self, n):
        yield self.content


class Session:
    def __init__(self, routes):
        self.routes = routes
        self.urls: list[str] = []

    def get(self, url, **kw):
        self.urls.append(url)
        for prefix, body in self.routes:
            if url.startswith(prefix) or prefix in url:
                return Resp(body() if callable(body) else body)
        return Resp(b"", 404)


def test_parse_listing_reads_keys_sizes_etags_and_the_token():
    files, token = parse_listing(page([("2024/07/05/OPERA/COMP/OPERA@20240705T0000@0@RATE.h5",
                                         123, "abc")], token="T1"))
    assert files == [RemoteFile("OPERA@20240705T0000@0@RATE.h5", 123,
                                "2026-02-06T10:00:00.000Z",
                                "2024/07/05/OPERA/COMP/OPERA@20240705T0000@0@RATE.h5", "abc")]
    assert token == "T1"
    assert parse_listing(page([]))[1] is None


def test_a_listing_follows_continuation_tokens():
    first = page([("p/a.h5", 1, "x")], token="NEXT")
    second = page([("p/b.h5", 2, "y")])
    s = Session([("continuation-token=NEXT", second), ("https://h/b/?list-type=2", first)])
    client = S3Client("https://h", "b", requests_per_second=0, session=s)
    got = client.list_prefix("p/")
    assert [f.filename for f in got] == ["a.h5", "b.h5"]
    assert client.requests_made == 2 and "prefix=p/" in s.urls[0]


def test_a_failing_listing_is_a_listing_error_not_an_empty_day():
    client = S3Client("https://h", "b", requests_per_second=0, retries=0,
                      session=Session([]))
    with pytest.raises(ListingError):
        client.list_prefix("2024/07/05/")


def test_downloads_are_checked_against_size_signature_and_md5(tmp_path: Path):
    body = HDF5_SIGNATURE + b"payload"
    md5 = hashlib.md5(body).hexdigest()
    s = Session([("https://h/b/k/f.h5", body)])
    client = S3Client("https://h", "b", requests_per_second=0, retries=0, backoff=0,
                      session=s)
    good = RemoteFile("f.h5", len(body), key="k/f.h5", etag=md5)
    raw = client.download(good, tmp_path / "f.h5")
    assert raw.attempts == 1 and (tmp_path / "f.h5").read_bytes() == body
    # Already there and intact: no request.
    n = len(s.urls)
    assert client.download(good, tmp_path / "f.h5").attempts == 0 and len(s.urls) == n

    for bad in (RemoteFile("g.h5", len(body) + 1, key="k/f.h5", etag=md5),
                RemoteFile("g.h5", len(body), key="k/f.h5", etag="0" * 32)):
        with pytest.raises(FetchError, match="IntegrityError"):
            client.download(bad, tmp_path / "g.h5")
        assert not (tmp_path / "g.h5").exists()
        assert not list(tmp_path.glob("*.part"))


def test_a_non_hdf5_body_is_refused(tmp_path: Path):
    body = b"<html>error</html>"
    client = S3Client("https://h", "b", requests_per_second=0, retries=0, backoff=0,
                      session=Session([("https://h/b/x.h5", body)]))
    with pytest.raises(FetchError, match="not an HDF5"):
        client.download(RemoteFile("x.h5", len(body), key="x.h5"), tmp_path / "x.h5")


class RangeSession:
    """Serves one object, honouring (or, with `ignore_range`, ignoring) a Range header."""

    def __init__(self, body: bytes, *, ignore_range: bool = False, short: int = 0):
        self.body = body
        self.ignore_range = ignore_range
        self.short = short
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url, headers=None, **kw):
        self.calls.append((url, headers))
        if url.endswith("/missing"):
            return Resp(b"", 404)
        if not headers or self.ignore_range:
            return Resp(self.body)
        lo, _, hi = headers["Range"].removeprefix("bytes=").partition("-")
        part = self.body[int(lo): None if hi == "" else int(hi) + 1]
        if self.short:
            part, self.short = part[:-1], self.short - 1
        return Resp(part, 206)


def test_byte_range_reads_return_exactly_the_bytes_asked_for():
    s = RangeSession(bytes(range(100)))
    client = S3Client("https://b.s3.example.com", "", requests_per_second=0, session=s,
                      backoff=0)
    assert client.get_bytes("k/f", 10, 19) == bytes(range(10, 20))
    assert client.get_bytes("k/f", 95) == bytes(range(95, 100))
    # A virtual-hosted endpoint: no bucket in the path.
    assert s.calls[0] == ("https://b.s3.example.com/k/f", {"Range": "bytes=10-19"})
    assert client.stats["bytes"] == 15
    assert client.get_text("missing") is None


def test_a_short_or_unranged_answer_is_retried_then_refused():
    s = RangeSession(bytes(range(100)), short=1)
    client = S3Client("https://h", "b", requests_per_second=0, session=s, backoff=0)
    assert client.get_bytes("f", 0, 9) == bytes(range(10))      # one short reply, retried
    from meteopack.core.http import IntegrityError

    s = RangeSession(bytes(range(100)), ignore_range=True)
    client = S3Client("https://h", "b", requests_per_second=0, session=s, backoff=0,
                      retries=1)
    with pytest.raises(IntegrityError):
        client.get_bytes("f", 10, 19)
