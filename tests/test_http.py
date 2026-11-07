from __future__ import annotations

import hashlib
import itertools
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest
import requests

from meteopack.core import http as h
from meteopack.core.http import (
    FetchError,
    NotFoundError,
    RateLimiter,
    TarArchiveWriter,
    fetch_file,
    listing_cache,
    read_member,
)

PAYLOAD = bytes(range(256)) * 5000


class _State:
    def __init__(self):
        self.hits: dict[str, int] = {}
        self.script: dict[str, list] = {}


@pytest.fixture
def server():
    state = _State()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            n = state.hits[self.path] = state.hits.get(self.path, 0) + 1
            steps = state.script.get(self.path, [])
            step = steps[n - 1] if n <= len(steps) else "ok"
            if step == "ok":
                self.send_response(200)
                self.send_header("Content-Length", str(len(PAYLOAD)))
                self.end_headers()
                self.wfile.write(PAYLOAD)
            elif step == "truncate":
                self.send_response(200)
                self.send_header("Content-Length", str(len(PAYLOAD)))
                self.end_headers()
                self.wfile.write(PAYLOAD[:1000])
                self.wfile.flush()
                self.close_connection = True
            elif isinstance(step, tuple):
                code, hdrs = step
                self.send_response(code)
                for k, v in hdrs.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self.send_response(step)
                self.send_header("Content-Length", "0")
                self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    state.base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield state
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def session():
    with requests.Session() as s:
        yield s


def _fetch(server, path, dest, session, **kw):
    kw.setdefault("retries", 3)
    kw.setdefault("backoff", 0.01)
    return fetch_file(server.base + path, dest, unit_key="k1", session=session, **kw)


def test_sha256_and_size(server, session, tmp_path):
    dest = tmp_path / "sub" / "f.bin"
    raw = _fetch(server, "/f", dest, session)
    assert raw.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    assert raw.n_bytes == len(PAYLOAD)
    assert raw.unit_key == "k1"
    assert raw.path == dest
    assert dest.read_bytes() == PAYLOAD
    assert not list(dest.parent.glob("*.part"))


def test_idempotent_skip_if_exists(server, session, tmp_path):
    dest = tmp_path / "f.bin"
    dest.write_bytes(b"already here")
    raw = _fetch(server, "/f", dest, session)
    assert server.hits == {}
    assert raw.sha256 == hashlib.sha256(b"already here").hexdigest()
    assert raw.n_bytes == len(b"already here")
    assert dest.read_bytes() == b"already here"


def test_retry_then_succeed_on_5xx_and_429(server, session, tmp_path):
    server.script["/f"] = [503, 500, (429, {"Retry-After": "0"})]
    dest = tmp_path / "f.bin"
    raw = _fetch(server, "/f", dest, session, retries=5)
    assert server.hits["/f"] == 4
    assert raw.sha256 == hashlib.sha256(PAYLOAD).hexdigest()


def test_retries_exhausted_raises(server, session, tmp_path):
    server.script["/f"] = [500] * 10
    dest = tmp_path / "f.bin"
    with pytest.raises(FetchError) as ei:
        _fetch(server, "/f", dest, session, retries=2)
    assert ei.value.status == 500
    assert server.hits["/f"] == 3
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))


def test_retry_after_is_honoured(server, session, tmp_path, monkeypatch):
    server.script["/f"] = [(429, {"Retry-After": "7"})]
    sleeps: list[float] = []
    monkeypatch.setattr(h.time, "sleep", sleeps.append)
    _fetch(server, "/f", tmp_path / "f.bin", session)
    assert sleeps and sleeps[0] >= 7


def test_404_raises_notfound_without_retry(server, session, tmp_path):
    server.script["/f"] = [404, 404]
    dest = tmp_path / "f.bin"
    with pytest.raises(NotFoundError):
        _fetch(server, "/f", dest, session)
    assert server.hits["/f"] == 1
    assert not dest.exists()
    assert isinstance(NotFoundError("u", "m"), FetchError)


@pytest.mark.parametrize("code", [400, 401, 403, 410])
def test_other_4xx_not_retried(server, session, tmp_path, code):
    server.script["/f"] = [code, code]
    with pytest.raises(FetchError) as ei:
        _fetch(server, "/f", tmp_path / "f.bin", session)
    assert not isinstance(ei.value, NotFoundError)
    assert server.hits["/f"] == 1


def test_408_is_retried(server, session, tmp_path):
    server.script["/f"] = [408]
    _fetch(server, "/f", tmp_path / "f.bin", session)
    assert server.hits["/f"] == 2


def test_midstream_truncation_retries_and_leaves_no_part(server, session, tmp_path):
    server.script["/f"] = ["truncate"]
    dest = tmp_path / "f.bin"
    raw = _fetch(server, "/f", dest, session)
    assert server.hits["/f"] == 2
    assert dest.read_bytes() == PAYLOAD
    assert raw.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    assert not list(tmp_path.glob("*.part"))


def test_midstream_failure_leaves_no_dest_and_no_part(server, session, tmp_path):
    server.script["/f"] = ["truncate"] * 10
    dest = tmp_path / "f.bin"
    with pytest.raises(FetchError):
        _fetch(server, "/f", dest, session, retries=1)
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))


def test_non_http_exception_midstream_cleans_up(tmp_path):
    class Resp:
        status_code = 200
        headers: ClassVar[dict] = {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_content(self, n):
            yield b"abc"
            raise KeyboardInterrupt

    class Sess:
        def get(self, *a, **k):
            return Resp()

    dest = tmp_path / "f.bin"
    with pytest.raises(KeyboardInterrupt):
        fetch_file("http://x/f", dest, unit_key="k", session=Sess(), retries=0, backoff=0)
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))


def test_limiter_used_per_attempt(server, session, tmp_path):
    calls: list[str] = []

    class L:
        def acquire(self, host):
            calls.append(host)

    server.script["/f"] = [500]
    _fetch(server, "/f", tmp_path / "f.bin", session, limiter=L())
    assert len(calls) == 2 and calls[0].startswith("127.0.0.1:")


def test_rate_limiter_spaces_concurrent_calls_deterministic():
    now = [0.0]
    lock = threading.Lock()
    slept: list[float] = []

    def sleep(d):
        with lock:
            slept.append(d)

    lim = RateLimiter(10.0, clock=lambda: now[0], sleep=sleep)
    for _ in range(5):
        lim.acquire("a")
    assert slept == pytest.approx([0.1, 0.2, 0.3, 0.4])
    lim.acquire("b")
    assert len(slept) == 4


def test_rate_limiter_spaces_real_threads():
    rps = 20.0
    lim = RateLimiter(rps)
    stamps: list[float] = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def work():
        barrier.wait()
        lim.acquire("host")
        with lock:
            stamps.append(time.monotonic())

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stamps.sort()
    gaps = [b - a for a, b in itertools.pairwise(stamps)]
    assert min(gaps) >= 1 / rps * 0.7
    assert stamps[-1] - stamps[0] >= 7 / rps * 0.9


def test_rate_limiter_hosts_independent():
    lim = RateLimiter(2.0)
    t0 = time.monotonic()
    lim.acquire("a")
    lim.acquire("b")
    assert time.monotonic() - t0 < 0.3


def test_rate_limiter_rejects_nonpositive():
    with pytest.raises(ValueError):
        RateLimiter(0)


def test_listing_cache_roundtrip_and_ttl(tmp_path):
    now = [1000.0]
    c = listing_cache(tmp_path / "d" / "l.json", 60, clock=lambda: now[0])
    assert c.get() is None
    c.set(["a", "b", {"k": 1}])
    assert c.get() == ["a", "b", {"k": 1}]
    now[0] += 59
    assert c.get() == ["a", "b", {"k": 1}]
    now[0] += 2
    assert c.get() is None
    assert not list((tmp_path / "d").glob("*.part"))


def test_listing_cache_corrupt_file_is_a_miss(tmp_path):
    p = tmp_path / "l.json"
    p.write_text("{not json")
    assert listing_cache(p, 60).get() is None


def test_tar_roundtrip(tmp_path):
    p = tmp_path / "2026-01-01.tar"
    with TarArchiveWriter(p) as w:
        assert w.add_bytes("a.nc", b"alpha")
        assert w.add_bytes("b.nc", b"\x00\x01beta")
        assert not p.exists()
        assert list(tmp_path.glob("*.part")), "a private temp file should exist while open"
    assert not list(tmp_path.glob("*.part"))
    assert read_member(p, "a.nc") == b"alpha"
    assert read_member(p, "b.nc") == b"\x00\x01beta"
    assert h.list_members(p) == ["a.nc", "b.nc"]
    with pytest.raises(KeyError):
        read_member(p, "missing")


def test_tar_failure_leaves_no_tar(tmp_path):
    p = tmp_path / "x.tar"
    with pytest.raises(RuntimeError), TarArchiveWriter(p) as w:
        w.add_bytes("a", b"1")
        raise RuntimeError("killed")
    assert not p.exists()
    assert not list(tmp_path.glob("*.part"))


def test_tar_failure_keeps_previous_tar_intact(tmp_path):
    p = tmp_path / "x.tar"
    with TarArchiveWriter(p) as w:
        w.add_bytes("a", b"1")
    with pytest.raises(RuntimeError), TarArchiveWriter(p) as w:
        w.add_bytes("b", b"2")
        raise RuntimeError
    assert h.list_members(p) == ["a"]


def test_tar_reopen_appends_and_dedupes(tmp_path):
    p = tmp_path / "x.tar"
    with TarArchiveWriter(p) as w:
        w.add_bytes("a", b"1")
    with TarArchiveWriter(p) as w:
        assert "a" in w
        assert not w.add_bytes("a", b"CHANGED")
        assert w.add_bytes("b", b"2")
    assert h.list_members(p) == ["a", "b"]
    assert read_member(p, "a") == b"1"
    with tarfile.open(p) as t:
        assert len(t.getmembers()) == 2


def test_part_paths_are_private_to_the_writer(tmp_path):
    """Two writers aiming at one destination must not share a temp file.

    Slurm array tasks can legitimately collide (overlapping shards, a re-run
    racing a straggler). With a fixed `.part` name they would interleave bytes
    into one file and publish the mangled result.
    """
    dest = tmp_path / "bucket.tar"
    assert h._part_path(dest) != h._part_path(dest)
    assert all(q.name.endswith(".part") for q in (h._part_path(dest), h._part_path(dest)))


def test_concurrent_writers_never_publish_a_mangled_tar(tmp_path):
    import threading

    dest = tmp_path / "day.tar"
    payloads = {f"w{i}": bytes([i]) * 4096 for i in range(6)}
    barrier = threading.Barrier(len(payloads))

    def write(name: str, data: bytes) -> None:
        barrier.wait()
        with h.TarArchiveWriter(dest) as w:
            w.add_bytes(name, data)

    threads = [threading.Thread(target=write, args=kv) for kv in payloads.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Last writer wins, but whatever landed must be a readable tar, not a splice
    # of several writers' bytes, and no temp files may survive.
    assert not list(tmp_path.glob("*.part"))
    members = h.list_members(dest)
    assert members, "a tar should have been published"
    for name in members:
        assert h.read_member(dest, name) == payloads[name]


# ------------------------------------------- why a download failed, and checks


def test_a_final_failure_says_how_many_attempts_and_what_failed(server, session, tmp_path):
    """The ledger used to get "FetchError: url: HTTP 503" and nothing else."""
    server.script["/f"] = [503] * 10
    with pytest.raises(FetchError) as ei:
        _fetch(server, "/f", tmp_path / "f.bin", session, retries=2)
    assert (ei.value.attempts, ei.value.status, ei.value.error_class) == (3, 503, "HTTPError")


def test_a_transient_transport_error_is_classed_by_its_type(server, session, tmp_path):
    server.script["/f"] = ["truncate"] * 10
    with pytest.raises(FetchError) as ei:
        _fetch(server, "/f", tmp_path / "f.bin", session, retries=1)
    assert ei.value.attempts == 2
    assert ei.value.status is None
    assert ei.value.error_class == "ChunkedEncodingError"


def _check_failing(times: int):
    calls = {"n": 0}

    def check(path):
        calls["n"] += 1
        assert path.exists()                      # runs on the finished .part
        if calls["n"] <= times:
            raise h.IntegrityError("not a zip")
        return h.Content(n_records=7)

    return check, calls


def test_a_file_failing_its_check_is_downloaded_again_and_not_published(server, session,
                                                                        tmp_path):
    check, calls = _check_failing(1)
    dest = tmp_path / "f.bin"
    raw = _fetch(server, "/f", dest, session, check=check)
    assert server.hits["/f"] == 2 and calls["n"] == 2
    assert raw.content == h.Content(n_records=7)
    assert raw.attempts == 2


def test_a_file_that_never_passes_its_check_fails_as_an_integrity_error(server, session,
                                                                        tmp_path):
    check, _ = _check_failing(99)
    dest = tmp_path / "f.bin"
    with pytest.raises(FetchError) as ei:
        _fetch(server, "/f", dest, session, retries=2, check=check)
    assert ei.value.error_class == "IntegrityError"
    assert ei.value.attempts == 3
    assert not dest.exists()                       # a broken file is never published
    assert not list(tmp_path.glob("*.part"))


def test_a_broken_file_already_on_disk_is_replaced_not_trusted(server, session, tmp_path):
    """Before, existence was enough: a truncated file left by an older fetcher
    was re-hashed and recorded as fetched."""
    dest = tmp_path / "f.bin"
    dest.write_bytes(b"truncated")

    def check(path):
        if path.read_bytes() == b"truncated":
            raise h.IntegrityError("cut short")
        return h.Content(n_records=1)

    raw = _fetch(server, "/f", dest, session, check=check)
    assert server.hits["/f"] == 1
    assert dest.read_bytes() == PAYLOAD
    assert raw.content.n_records == 1


def test_a_good_file_already_on_disk_is_checked_and_not_downloaded(server, session, tmp_path):
    dest = tmp_path / "f.bin"
    dest.write_bytes(b"fine")
    raw = _fetch(server, "/f", dest, session, check=lambda p: h.Content(n_records=4))
    assert server.hits == {}
    assert raw.attempts == 0
    assert raw.content.n_records == 4


def test_listings_are_retried_rather_than_read_as_empty(server, session):
    """A listing GET used to be one bare request; its failure became an empty list."""
    server.script["/dir/"] = [503, (429, {"Retry-After": "0"})]
    resp = h.get_with_retry(server.base + "/dir/", session=session, backoff=0.01)
    assert resp.status_code == 200 and server.hits["/dir/"] == 3


def test_a_listing_that_keeps_failing_raises(server, session):
    server.script["/dir/"] = [500] * 10
    with pytest.raises(FetchError) as ei:
        h.get_with_retry(server.base + "/dir/", session=session, retries=1, backoff=0.01)
    assert ei.value.attempts == 2
    server.script["/gone/"] = [404]
    with pytest.raises(NotFoundError):
        h.get_with_retry(server.base + "/gone/", session=session, backoff=0.01)
    assert server.hits["/gone/"] == 1


def test_a_source_s_throttle_status_is_retried(server, session, tmp_path):
    """EA refuses ~16% of requests with 403 under load; asked again, they succeed."""
    server.script["/f"] = [403, 403]
    raw = _fetch(server, "/f", tmp_path / "f.bin", session, retry_status=[403])
    assert server.hits["/f"] == 3 and raw.n_bytes == len(PAYLOAD)


class _RecordingLimiter:
    def __init__(self):
        self.paused: list[tuple[str, float]] = []

    def acquire(self, host):
        pass

    def pause(self, host, seconds):
        self.paused.append((host, seconds))


@pytest.mark.parametrize("step,extra", [((429, {"Retry-After": "0"}), ()), (503, ()),
                                        (403, (403,))])
def test_a_throttle_pauses_the_whole_host(server, session, tmp_path, step, extra):
    """One thread backing off while the other workers carry on is not backing off."""
    server.script["/f"] = [step]
    limiter = _RecordingLimiter()
    _fetch(server, "/f", tmp_path / "f.bin", session, limiter=limiter, retry_status=extra)
    assert len(limiter.paused) == 1 and limiter.paused[0][0].startswith("127.0.0.1")


def test_a_plain_server_error_does_not_pause_the_host(server, session, tmp_path):
    server.script["/f"] = [500]
    limiter = _RecordingLimiter()
    _fetch(server, "/f", tmp_path / "f.bin", session, limiter=limiter)
    assert limiter.paused == []


def test_a_pause_holds_back_every_later_caller():
    now = [100.0]
    sleeps: list[float] = []
    rl = RateLimiter(10.0, clock=lambda: now[0], sleep=sleeps.append)
    rl.pause("h", 5.0)
    rl.acquire("h")
    rl.acquire("h")
    assert sleeps == [pytest.approx(5.0), pytest.approx(5.1)]


def test_stale_part_files_are_removed_and_fresh_ones_kept(tmp_path):
    old = tmp_path / "a" / "x.zip.123.abcd.part"
    fresh = tmp_path / "a" / "y.zip.456.ef01.part"
    real = tmp_path / "a" / "z.zip"
    for p in (old, fresh, real):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    import os

    os.utime(old, (1000.0, 1000.0))
    os.utime(real, (1000.0, 1000.0))
    assert h.remove_stale_parts(tmp_path, now=1000.0 + h.STALE_PART_SECONDS + 1) == 1
    assert not old.exists() and fresh.exists() and real.exists()
