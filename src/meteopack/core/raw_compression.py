"""Lossless storage compression for raw files (`raw_layout.compression`).

A product configured with ``compression: xz`` keeps every fetched file as
``<provider file name>.xz`` -- `2019.csv` becomes `2019.csv.xz`, a daily
`...tar` becomes `...tar.xz` -- so the name says what the bytes are and
``xz -dc`` recovers the download exactly.

The contract, in three parts:

* **The suffix is the marker.** A raw file is compressed if and only if its
  name ends in ``.xz``; nothing sniffs magic bytes.
* **Checksums describe the provider's bytes.** `RawFile.sha256`/`n_bytes` and
  the ledger rows are those of the file as downloaded, before compression, so
  they stay comparable across storage settings (xz carries its own CRC for the
  stored bytes).
* **Parsers never see xz.** `materialized` restores the original bytes under
  the original name in temporary space for the duration of a parse.
"""
from __future__ import annotations

import dataclasses
import hashlib
import lzma
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import BinaryIO

from meteopack.sources.base import RawFile

SUFFIX = ".xz"
#: xz -3: on the station archives this was within a few percent of -6's ratio
#: at a fraction of the time and memory (2026-11-08 benchmark).
PRESET = 3
_CHUNK = 4 * 1024 * 1024


def is_compressed(path: Path | str) -> bool:
    return Path(path).name.endswith(SUFFIX)


def compressed_path(path: Path | str) -> Path:
    """Where the compressed copy of `path` lives."""
    path = Path(path)
    return path if is_compressed(path) else path.with_name(path.name + SUFFIX)


def logical_path(path: Path | str) -> Path:
    """The provider's file name: `path` without the storage suffix."""
    path = Path(path)
    return path.with_name(path.name[: -len(SUFFIX)]) if is_compressed(path) else path


def stored_path(path: Path | str) -> Path:
    """The file actually on disk for a provider path: its `.xz` copy if present."""
    path = Path(path)
    packed = compressed_path(path)
    return packed if packed.exists() else path


def open_logical(path: Path | str) -> BinaryIO:
    """Open a raw file for reading its original bytes, compressed or not."""
    return lzma.open(path, "rb") if is_compressed(path) else Path(path).open("rb")


def _hash(f: BinaryIO) -> tuple[str, int]:
    h, n = hashlib.sha256(), 0
    while block := f.read(_CHUNK):
        h.update(block)
        n += len(block)
    return h.hexdigest(), n


def digest(path: Path | str) -> tuple[str, int]:
    """sha256 and size of the original bytes, decompressing on the fly."""
    with open_logical(path) as f:
        return _hash(f)


def compress(path: Path | str, *, preset: int = PRESET) -> Path:
    """Replace `path` by `path.xz`, removing the original only once verified.

    The compressed copy is written to a hidden ``.part`` file (which
    `http.remove_stale_parts` clears after a crash), decompressed and compared
    with the original's sha256 and size, and only then moved into place. A
    crash at any point leaves either the original alone, or a complete `.xz`
    beside it -- never only a partial copy. Already-compressed paths are
    returned unchanged.
    """
    path = Path(path)
    if is_compressed(path):
        return path
    target = compressed_path(path)
    temp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part")
    want = hashlib.sha256()
    try:
        with path.open("rb") as src, lzma.open(temp, "wb", preset=preset) as dst:
            while block := src.read(_CHUNK):
                want.update(block)
                dst.write(block)
        shutil.copymode(path, temp)
        with lzma.open(temp, "rb") as f:
            restored = _hash(f)
        if restored != (want.hexdigest(), path.stat().st_size):
            raise OSError(f"xz round trip of {path} does not reproduce the original bytes")
        with temp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    path.unlink()
    return target


def compress_raw(raw: RawFile, codec: str) -> RawFile:
    """Apply a product's storage compression to one freshly fetched file."""
    if codec == "none":
        return raw
    if codec != "xz":
        raise ValueError(f"unsupported raw compression {codec!r}")
    return dataclasses.replace(raw, path=compress(raw.path))


def _temp_root() -> Path | None:
    """Node-local space for decompressed copies; $METEOPACK_TEMP_DIR wins."""
    configured = os.environ.get("METEOPACK_TEMP_DIR")
    if configured:
        root = Path(configured).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return root
    return None  # tempfile's default: $TMPDIR, else /tmp


@contextmanager
def materialized(path: Path | str) -> Iterator[Path]:
    """A readable uncompressed path with the provider's file name.

    Uncompressed files are yielded as they are; compressed ones are restored
    into a temporary directory that is removed on exit.
    """
    path = Path(path)
    if not is_compressed(path):
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="meteopack-raw-", dir=_temp_root()) as tmp:
        target = Path(tmp) / logical_path(path).name
        with lzma.open(path, "rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=_CHUNK)
        yield target


@contextmanager
def materialized_all(paths: Iterable[Path | str]) -> Iterator[list[Path]]:
    """`materialized` for several files at once (all restored until exit)."""
    with ExitStack() as stack:
        yield [stack.enter_context(materialized(p)) for p in paths]


@contextmanager
def materialized_raw(raw: RawFile) -> Iterator[RawFile]:
    """`raw` with its path swapped for a readable uncompressed one."""
    with materialized(raw.path) as path:
        yield raw if path == Path(raw.path) else dataclasses.replace(raw, path=path)
