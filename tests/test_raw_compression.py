from __future__ import annotations

import hashlib
import lzma
from pathlib import Path
from types import SimpleNamespace

import pytest

from meteopack.core import raw_compression as rc
from meteopack.core.config import RawLayout
from meteopack.sources.base import RawFile

CONTENT = b"time,station,value\n2026-01-01T00:00:00Z,abc,1.25\n" * 100


def _raw(path: Path, content: bytes = CONTENT) -> RawFile:
    path.write_bytes(content)
    return RawFile(path=path, sha256=hashlib.sha256(content).hexdigest(),
                   n_bytes=len(content), unit_key="u", attempts=1)


def test_compression_is_opt_in():
    assert RawLayout().compression == "none"
    assert RawLayout(compression="xz").compression == "xz"


def test_names():
    assert rc.compressed_path("a/2019.csv") == Path("a/2019.csv.xz")
    assert rc.compressed_path("a/2019.csv.xz") == Path("a/2019.csv.xz")
    assert rc.logical_path("a/day.tar.xz") == Path("a/day.tar")
    assert rc.logical_path("a/day.tar") == Path("a/day.tar")


def test_compress_raw_appends_suffix_and_keeps_the_original_checksum(tmp_path):
    raw = _raw(tmp_path / "2019.csv")

    stored = rc.compress_raw(raw, "xz")

    assert stored.path == tmp_path / "2019.csv.xz"
    assert not raw.path.exists()
    assert lzma.decompress(stored.path.read_bytes()) == CONTENT
    # The ledger describes what the provider served, not the storage bytes.
    assert (stored.sha256, stored.n_bytes, stored.attempts) == (raw.sha256, raw.n_bytes, 1)
    assert rc.digest(stored.path) == (raw.sha256, raw.n_bytes)
    assert not list(tmp_path.glob("*.part"))


def test_compress_raw_none_leaves_the_file_alone(tmp_path):
    raw = _raw(tmp_path / "2019.csv")
    assert rc.compress_raw(raw, "none") is raw
    assert raw.path.read_bytes() == CONTENT


def test_compress_is_idempotent_and_handles_empty_files(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    stored = rc.compress(empty)
    assert stored.name == "empty.csv.xz"
    assert rc.digest(stored) == (hashlib.sha256(b"").hexdigest(), 0)
    assert rc.compress(stored) == stored


def test_failed_verification_keeps_the_original(tmp_path, monkeypatch):
    path = tmp_path / "day.tar"
    path.write_bytes(CONTENT)
    monkeypatch.setattr(rc, "_hash", lambda f: ("bad", 0))

    with pytest.raises(OSError, match="round trip"):
        rc.compress(path)

    assert path.read_bytes() == CONTENT
    assert sorted(p.name for p in tmp_path.iterdir()) == ["day.tar"]


def test_materialized_restores_the_provider_name_and_bytes(tmp_path):
    stored = rc.compress(_raw(tmp_path / "2019.csv").path)

    with rc.materialized(stored) as readable:
        assert readable.name == "2019.csv"
        assert readable.read_bytes() == CONTENT
        temp = readable
    assert not temp.exists()

    plain = tmp_path / "plain.csv"
    plain.write_bytes(b"x")
    with rc.materialized(plain) as readable:
        assert readable == plain


def test_stored_path_prefers_the_compressed_copy(tmp_path):
    dest = tmp_path / "2019.csv"
    assert rc.stored_path(dest) == dest
    rc.compress(_raw(dest).path)
    assert rc.stored_path(dest) == tmp_path / "2019.csv.xz"


def test_fetch_file_reuses_an_existing_xz_and_checks_the_original_bytes(tmp_path):
    from meteopack.core.http import fetch_file

    dest = tmp_path / "measure.csv"
    rc.compress(_raw(dest, b"header\nvalue\n").path)
    checked = []

    def check(readable):
        checked.append(readable.read_bytes())
        return "content"

    raw = fetch_file("https://unused.invalid/measure.csv", dest, unit_key="u",
                     session=object(), check=check)

    assert checked == [b"header\nvalue\n"]
    assert raw.path == tmp_path / "measure.csv.xz"
    assert raw.attempts == 0
    assert (raw.sha256, raw.n_bytes) == (hashlib.sha256(b"header\nvalue\n").hexdigest(), 13)
    assert raw.content == "content"


def test_fetch_and_store_compresses_a_new_unit(tmp_path):
    from meteopack.cli import _fetch_and_store

    class Source:
        def fetch(self, unit, raw_root):
            return _raw(Path(raw_root) / "day.tar")

    raw = _fetch_and_store(Source(), object(), tmp_path, "xz")

    assert raw.path == tmp_path / "day.tar.xz"
    assert not (tmp_path / "day.tar").exists()
    assert rc.digest(raw.path) == (raw.sha256, raw.n_bytes)


def test_fetch_and_store_passes_a_404_through(tmp_path):
    from meteopack.cli import _fetch_and_store

    class Source:
        def fetch(self, unit, raw_root):
            return None

    assert _fetch_and_store(Source(), object(), tmp_path, "xz") is None


def test_raw_files_lists_one_name_per_unit(tmp_path):
    """A crash between publishing the .xz and removing the original leaves both."""
    from meteopack.cli import Context

    root = tmp_path / "raw"
    (root / "a").mkdir(parents=True)
    rc.compress(_raw(root / "a" / "2019.csv").path)
    _raw(root / "a" / "2019.csv")                      # the left-over original
    _raw(root / "a" / "2020.csv")                      # not compressed yet
    (root / "a" / ".2021.csv.xz.1.ab.part").write_bytes(b"partial")

    ctx = SimpleNamespace(raw_root=root, SIDECARS=Context.SIDECARS)
    names = [p.name for p in Context.raw_files(ctx)]
    assert names == ["2019.csv.xz", "2020.csv"]


def test_midas_reads_station_headers_from_compressed_files(tmp_path):
    from meteopack.sources.stations.midas import _head

    stored = rc.compress(_raw(tmp_path / "2019.csv", b"Conventions,G,BADC-CSV,1\n").path)
    assert _head(stored).startswith("Conventions,G,BADC-CSV")


def test_a_ledger_path_compressed_later_has_not_vanished(tmp_path):
    """Rows written before compression name the provider path; that is not a loss."""
    from meteopack.core.catalog import Catalog
    from meteopack.jobs.plan import _vanished

    raw = _raw(tmp_path / "2019.csv")
    catalog = Catalog(tmp_path / "data")
    catalog.record_file(raw, source="s", product="p", url="https://x")
    cfg = SimpleNamespace(source="s", product="p")

    rc.compress(raw.path)
    assert _vanished(catalog, cfg, {"u"}, set(), "fetch") == set()

    rc.compressed_path(raw.path).unlink()
    assert _vanished(catalog, cfg, {"u"}, set(), "fetch") == {"fetch:u"}
