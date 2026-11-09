"""Walking a tar's member index with ranged reads, as on the KDP P1 tars."""
from __future__ import annotations

import io
import tarfile

import pytest

from meteopack.core import remote_tar
from meteopack.core.http import IntegrityError


class BytesReader:
    """`read(start, length)` over a bytes object, counting requests like RangeReader."""

    def __init__(self, blob: bytes) -> None:
        self.blob = blob
        self.requests = 0

    def read(self, start: int, length: int) -> bytes:
        self.requests += 1
        return self.blob[start:start + length]


def make_tar(members: dict[str, bytes], fmt=tarfile.PAX_FORMAT, pax_headers=None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=fmt, pax_headers=pax_headers) as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            if fmt == tarfile.PAX_FORMAT:
                info.pax_headers = {"mtime": "1767830400.123"}  # forces a PAX header per member
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


MEMBERS = {
    "HA43_N20_202601080000_00000_GB": b"GRIB" + b"a" * 1000,
    "HA43_N20_202601080000_00600_GB": b"GRIB" + b"b" * 3000,
    "HA43_N20_202601080000_00100_GB": b"GRIB" + b"c" * 513,
    "HA43_N20_202601080000_00200_GB": b"GRIB" + b"d" * 1,
}


@pytest.mark.parametrize("fmt", [tarfile.PAX_FORMAT, tarfile.USTAR_FORMAT, tarfile.GNU_FORMAT])
def test_members_are_found_with_offsets_that_read_back_exactly(fmt):
    blob = make_tar(MEMBERS, fmt=fmt)
    reader = BytesReader(blob)
    members = remote_tar.list_members(reader.read, len(blob))
    assert [m.name for m in members] == list(MEMBERS)
    for m in members:
        assert blob[m.offset:m.offset + m.size] == MEMBERS[m.name]
    # One read per member: a PAX header, its records and the member header
    # arrive together (plus the end-of-archive block).
    assert reader.requests <= len(MEMBERS) + 1


def test_long_names_and_early_stop():
    long = "x" * 150 + "_GB"
    blob = make_tar({long: b"1234", "short": b"56"}, fmt=tarfile.GNU_FORMAT)
    members = remote_tar.list_members(BytesReader(blob).read, len(blob))
    assert [m.name for m in members] == [long, "short"]
    reader = BytesReader(make_tar(MEMBERS))
    found = remote_tar.list_members(reader.read, len(reader.blob),
                                    stop=lambda ms: len(ms) == 2)
    assert len(found) == 2


def test_a_corrupt_header_stops_the_walk():
    blob = bytearray(make_tar(MEMBERS, fmt=tarfile.USTAR_FORMAT))
    blob[0] ^= 0xFF
    with pytest.raises(IntegrityError, match="checksum"):
        remote_tar.list_members(BytesReader(bytes(blob)).read, len(blob))


def test_range_reader_requires_partial_content():
    class Resp:
        def __init__(self, status, content):
            self.status_code, self.content = status, content

    class Session:
        def __init__(self, status, content):
            self.answer = Resp(status, content)
            self.headers = []

        def get(self, url, headers=None, timeout=None):
            self.headers.append(headers)
            return self.answer

    ok = Session(206, b"abcd")
    reader = remote_tar.RangeReader("https://s3/x?sig=1", session=ok, total_size=100)
    assert reader.read(10, 4) == b"abcd"
    assert ok.headers == [{"Range": "bytes=10-13"}]
    assert reader.requests == 1 and reader.bytes == 4
    from meteopack.core.http import FetchError

    with pytest.raises(FetchError, match="ignored the Range"):
        remote_tar.RangeReader("https://s3/x", session=Session(200, b"x" * 100)).read(0, 4)
    with pytest.raises(FetchError, match="403"):
        remote_tar.RangeReader("https://s3/x", session=Session(403, b"")).read(0, 4)
