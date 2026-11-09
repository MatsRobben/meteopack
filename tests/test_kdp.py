"""KNMI Data Platform client: key resolution, quota handling, listing, downloads."""
from __future__ import annotations

from pathlib import Path

import pytest

from meteopack.core import kdp
from meteopack.sources.common import CredentialError

SECRET = "super-secret-key-value"


def test_the_first_existing_key_file_wins_and_the_env_overrides(tmp_path: Path):
    bulk, regular, env_key = tmp_path / "bulk", tmp_path / "regular", tmp_path / "env"
    regular.write_text(SECRET + "\n")
    assert kdp.resolve_key([bulk, regular], env={}).path == regular
    bulk.write_text("bulk-key")
    assert kdp.resolve_key([bulk, regular], env={}).value == "bulk-key"
    env_key.write_text("env-key")
    assert kdp.resolve_key([bulk, regular],
                           env={kdp.KEY_FILE_ENV: str(env_key)}).value == "env-key"
    (tmp_path / "empty").write_text("  \n")
    assert kdp.resolve_key([tmp_path / "empty", regular], env={}).path == regular


def test_a_missing_key_names_the_files_and_never_the_key(tmp_path: Path):
    with pytest.raises(CredentialError, match="no KNMI Data Platform API key"):
        kdp.resolve_key([tmp_path / "nope"], env={})
    (tmp_path / "k").write_text(SECRET)
    key = kdp.resolve_key([tmp_path / "k"], env={})
    assert SECRET not in repr(key) and SECRET not in str(key)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_the_limiter_spaces_requests_and_waits_for_the_quota_reset():
    clock = Clock()
    lim = kdp.QuotaLimiter(2.0, reserve=2, clock=clock, sleep=clock.sleep)
    lim.acquire()
    lim.acquire()
    assert clock.slept == [0.5]
    lim.observe({"X-Ratelimit-Remaining": "3", "X-Ratelimit-Reset": str(clock.now + 100),
                 "X-Ratelimit-Limit": "1000"})
    lim.acquire()                      # 3 -> 2, still allowed
    assert lim.state["remaining"] == 2
    before = clock.now
    lim.acquire()                      # at the reserve: wait for the reset first
    assert clock.now >= before + 100
    assert lim.waited_for_quota > 0


def test_a_429_holds_every_caller_until_the_reset():
    clock = Clock()
    lim = kdp.QuotaLimiter(100.0, reserve=0, clock=clock, sleep=clock.sleep)
    lim.throttled(retry_after=30.0)
    lim.acquire()
    assert clock.now >= 1030.0


class Response:
    def __init__(self, status: int, payload=None, headers=None) -> None:
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload


class Session:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self.responses.pop(0)


def client(tmp_path: Path, responses) -> tuple[kdp.KdpClient, Session]:
    (tmp_path / "key").write_text(SECRET)
    session = Session(responses)
    c = kdp.KdpClient("3d-radar-composite", "1.0", key_files=[str(tmp_path / "key")],
                      session=session, default_rate=1000.0, backoff=0.0)
    return c, session


def test_listing_follows_pages_and_filters_by_filename(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(kdp.KEY_FILE_ENV, raising=False)
    page1 = {"files": [{"filename": "A_1.nc", "size": 3}], "isTruncated": True,
             "nextPageToken": "tok"}
    page2 = {"files": [{"filename": "A_2.nc", "size": 4, "lastModified": "x"}],
             "isTruncated": False}
    c, session = client(tmp_path, [Response(200, page1), Response(200, page2)])
    files = c.list_files(begin="A_0", end="A_9")
    assert [f.filename for f in files] == ["A_1.nc", "A_2.nc"]
    assert session.calls[0][1]["begin"] == "A_0" and session.calls[0][1]["end"] == "A_9"
    assert session.calls[0][1]["orderBy"] == "filename"
    assert session.calls[1][1]["nextPageToken"] == "tok"
    assert session.headers["Authorization"] == SECRET
    assert c.requests_made == 2


def test_429_is_waited_out_and_5xx_retried(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(kdp.KEY_FILE_ENV, raising=False)
    monkeypatch.setattr(kdp.time, "sleep", lambda s: None)
    ok = Response(200, {"temporaryDownloadUrl": "https://s3/x"})
    c, _ = client(tmp_path, [Response(429, headers={"Retry-After": "0"}), Response(503), ok])
    c.limiter = kdp.QuotaLimiter(1000.0, reserve=0, sleep=lambda s: None)
    assert c.download_url("f.nc") == "https://s3/x"


def test_a_rejected_key_is_a_credential_error_without_the_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(kdp.KEY_FILE_ENV, raising=False)
    c, _ = client(tmp_path, [Response(403)])
    with pytest.raises(CredentialError) as info:
        c.download_url("f.nc")
    assert SECRET not in str(info.value)


def test_a_new_key_file_is_picked_up_on_refresh(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(kdp.KEY_FILE_ENV, raising=False)
    regular = tmp_path / "regular"
    bulk = tmp_path / "bulk"
    regular.write_text("r")
    c = kdp.KdpClient("d", "1", key_files=[str(bulk), str(regular)], session=Session([]),
                      rates={str(bulk): 50.0}, default_rate=2.0)
    assert c.key.path == regular and c.limiter.interval == pytest.approx(0.5)
    assert not c.refresh_key()
    bulk.write_text("b")
    assert c.refresh_key()
    assert c.key.path == bulk and c.session.headers["Authorization"] == "b"
    assert c.limiter.interval == pytest.approx(0.02)


def test_download_checks_the_size_and_keeps_a_complete_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(kdp.KEY_FILE_ENV, raising=False)
    c, _ = client(tmp_path, [])
    remote = kdp.RemoteFile("f.nc", size=12)
    dest = tmp_path / "f.nc"
    dest.write_bytes(b"\x89HDF\r\n\x1a\n" + b"1234")
    calls = []
    monkeypatch.setattr(c, "download_url", lambda name: calls.append(name) or "https://s3/x")
    raw = c.download(remote, dest)
    assert raw.attempts == 0 and calls == []          # already whole: no request spent

    seen = {}

    def fake_fetch(url, dest, *, check, **kw):
        dest.write_bytes(b"\x89HDF\r\n\x1a\n" + b"12")  # short
        seen["error"] = None
        try:
            check(dest)
        except kdp.IntegrityError as exc:
            seen["error"] = str(exc)
        raise kdp.FetchError(url, "IntegrityError", attempts=1, error_class="IntegrityError")

    dest.write_bytes(b"short")
    monkeypatch.setattr(kdp.http, "fetch_file", fake_fetch)
    with pytest.raises(kdp.FetchError):
        c.download(remote, dest, attempts=2)
    assert "listing says 12" in seen["error"]
    assert calls == ["f.nc", "f.nc"]                  # a fresh URL per attempt
