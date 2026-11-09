"""Stream a remote product into a gridded store one UTC day at a time.

Two remotes (``extra.stream``): ``kdp``, the KNMI Data Platform (a key, a
quota, :class:`meteopack.core.kdp.KdpClient`), and ``s3``, a public bucket read
anonymously (:class:`meteopack.core.s3.S3Client`). What the files are - how a
day is listed, which file fills which time step, how one is checked and
packed - is the product's *format* (``extra.stream_format``, default
``knmi_radar_3d``; see :func:`format_for`), so the loop below is the same for
every product.

For each day: list -> download that day's files (bounded concurrency, one
shared quota-aware limiter, retries, temp file + atomic rename, size check) ->
check every file's schema -> pack -> delete the day's raw files -> write a
per-day status record. Raw data therefore never exceeds about two days (the
day being packed and the next one, which is downloaded meanwhile).

Resumable and idempotent:

* A day whose status is ``done`` is skipped. Any other day is run again, and
  only for the time steps not yet packed - a day left ``partial`` by files
  missing upstream costs one listing request to re-check.
* Files already on disk with the listed size are not downloaded again, so a
  job killed mid-day resumes without spending its quota twice.
* Writing a day is a region write of that day's shard; re-running it writes
  the same codes. ``redo`` re-packs a day from scratch.

Status records live beside the store, ``days/<YYYY>/<YYYY-MM-DD>.json``, and
the manifest carries the per-day counts and the schema fingerprints seen
across the archive (drift report).

A time step's "file" may be a bundle: a format that defines ``download``
fetches it itself (an NWP run is nine NetCDF files, or twelve members of a
remote tar, into one directory), a format that defines ``variables`` packs
several fields per time step, and a check that carries ``file_errors`` /
``quarantine`` fails single files of a bundle without failing the rest.

Exit codes of :func:`run`: 0 the window is finished (every day done, partial
or empty), 75 stopped early (time budget or SIGTERM) with days left - run it
again, 3 stopped on schema drift, 2 a credential problem, 1 anything else.
"""
from __future__ import annotations

import concurrent.futures as cf
import contextlib
import fcntl
import json
import logging
import os
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from meteopack.core import kdp
from meteopack.core.config import ProductConfig, layer_path
from meteopack.core.encoding import Encoding
from meteopack.core.store import gridded, manifest
from meteopack.core.timeline import Timeline
from meteopack.sources.radar import knmi_radar_3d as r3d

logger = logging.getLogger(__name__)

EXIT_DONE = 0
EXIT_ERROR = 1
EXIT_CREDENTIALS = 2
EXIT_SCHEMA = 3
EXIT_MORE = 75

STATUS_DIR = "days"
LOCK_NAME = ".stream.lock"
#: Consecutive days with schema failures after which the stream stops: one bad
#: file is recorded and skipped; a run of bad days is upstream drift that a
#: human has to look at before more of the archive is skipped.
MAX_SCHEMA_DAYS = 2


#: The remotes a product can be streamed from (``extra.stream``).
REMOTES = ("kdp", "s3")


@dataclass(slots=True)
class Settings:
    base_url: str
    dataset: str
    version: str
    prefix: str
    key_files: list[str]
    rates: dict[str, float]
    default_rate: float
    reserve: int
    download_workers: int
    origin: datetime
    #: ``kdp`` or ``s3``.
    remote: str = "kdp"
    #: s3 only: the bucket, and the day's key prefix as a format string of `day`.
    bucket: str = ""
    day_prefix: str = ""

    @classmethod
    def from_config(cls, cfg: ProductConfig) -> Settings:
        remote = cfg.extra.get("stream")
        origin = datetime.fromisoformat(str(cfg.extra["timeline_origin"])) \
            if cfg.extra.get("timeline_origin") else None
        if remote == "s3":
            spec = dict(cfg.extra.get("s3") or {})
            if not spec or origin is None:
                raise ValueError(f"{cfg.store_id}: an s3 stream needs extra.s3 and "
                                 f"extra.timeline_origin")
            return cls(
                base_url=str(spec["endpoint"]), dataset=str(spec["bucket"]), version="",
                prefix="", key_files=[], rates={},
                default_rate=float(spec.get("requests_per_second", 50.0)), reserve=0,
                download_workers=int(spec.get("download_workers", 8)),
                origin=origin if origin.tzinfo else origin.replace(tzinfo=UTC),
                remote="s3", bucket=str(spec["bucket"]), day_prefix=str(spec["day_prefix"]))
        spec = dict(cfg.extra.get("kdp") or {})
        if remote != "kdp" or not spec or origin is None:
            raise ValueError(f"{cfg.store_id} is not a streamed product "
                             f"(extra.stream: one of {REMOTES}, with its settings)")
        return cls(
            base_url=str(spec.get("base_url", kdp.DEFAULT_BASE_URL)),
            dataset=str(spec["dataset"]),
            version=str(spec["version"]),
            prefix=str(spec.get("filename_prefix", r3d.PREFIX)),
            key_files=[str(f) for f in spec.get("api_key_files", ["~/.knmi_api_key"])],
            rates={str(k): float(v) for k, v in (spec.get("requests_per_second") or {}).items()},
            default_rate=float(spec.get("default_requests_per_second", 2.0)),
            reserve=int(spec.get("quota_reserve", 5)),
            download_workers=int(spec.get("download_workers", 8)),
            origin=origin if origin.tzinfo else origin.replace(tzinfo=UTC),
        )


@dataclass(slots=True)
class Fetched:
    """One day's listing and downloads."""

    day: datetime
    listed: dict[datetime, kdp.RemoteFile] = field(default_factory=dict)
    paths: dict[datetime, Path] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    download_s: list[float] = field(default_factory=list)
    list_s: float = 0.0
    wall_s: float = 0.0
    bytes: int = 0
    requests: int = 0
    key_file: str = ""
    skipped_known: int = 0
    url_s: float = 0.0
    transfer_s: float = 0.0
    dl_wall_s: float = 0.0


def _day(value: datetime) -> datetime:
    value = value.astimezone(UTC)
    return datetime(value.year, value.month, value.day, tzinfo=UTC)


def store_dir(cfg: ProductConfig, root: Path) -> Path:
    return gridded.store_path(root, cfg.store_id)


def raw_dir(cfg: ProductConfig, root: Path, day: datetime) -> Path:
    return layer_path(root, "raw", cfg.source, cfg.product) / f"{day:%Y/%m/%d}"


def status_path(cfg: ProductConfig, root: Path, day: datetime) -> Path:
    return store_dir(cfg, root) / STATUS_DIR / f"{day:%Y}" / f"{day:%Y-%m-%d}.json"


def read_status(cfg: ProductConfig, root: Path, day: datetime) -> dict | None:
    path = status_path(cfg, root, day)
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


class Knmi3dFormat:
    """The KNMI 3D radar composite on the KDP: what `Stream` asks of a format.

    A format answers, for one product: how one day is listed, which listed file
    fills which time step, how one file is checked and packed, which
    `source_layout_id` a packed frame gets, and what the manifest says about
    the encoding. Another product supplies its own (`format_for`); the loop in
    `Stream` never looks inside a file.
    """

    def __init__(self, cfg: ProductConfig, settings: Settings) -> None:
        self.cfg = cfg
        self.settings = settings
        self.schema = r3d.Schema.from_config(cfg)
        self.target: Encoding = r3d.target_encoding(cfg)
        self.layout_names = r3d.LAYOUT_NAMES
        self.pinned = frozenset({cfg.schema_fingerprint} if cfg.schema_fingerprint else ())
        self.frame_shape = (len(cfg.levels.values), self.schema.ny, self.schema.nx)

    def list_day(self, client, day: datetime) -> list:
        begin = f"{self.settings.prefix}{day:%Y%m%d}0000"
        end = f"{self.settings.prefix}{day:%Y%m%d}2359~"
        return client.list_files(begin=begin, end=end)

    def select(self, files, day: datetime) -> dict[datetime, object]:
        out = {}
        for remote in files:
            when = r3d.filename_time(remote.filename)
            if when is None or _day(when) != day:
                continue
            out[when] = remote
        return out

    def pack(self, path: Path, when: datetime):
        return r3d.pack_file(path, self.cfg, self.schema, self.target)

    def layout(self, packed) -> int:
        return (r3d.LAYOUT_V1 if not self.pinned or packed.check.fingerprint in self.pinned
                else r3d.LAYOUT_UNKNOWN)

    def frame_values(self, packed) -> dict:
        return {}

    def provenance(self, packed) -> dict:
        return {"scale_factor": packed.check.scale_factor,
                "add_offset": packed.check.add_offset}

    def store_attrs(self) -> dict:
        return {}

    def manifest_extra(self) -> dict:
        enc = self.target
        return {
            "encoding": {"dtype": str(enc.dtype), "scale": enc.scale, "offset": enc.offset,
                         "no_echo_code": 0, "not_observed_code": enc.out_of_image,
                         "missing_frame_code": enc.missing,
                         "formula": "dBZ = 0.5 * code - 32"},
            "levels": self.cfg.levels.model_dump(),
            "orientation": "north-up (y descending), flipped from the south-up source",
            "source": {"api": self.settings.base_url, "dataset": self.settings.dataset,
                       "version": self.settings.version,
                       "files": f"{self.settings.prefix}*.nc"},
        }


#: ``extra.stream_format`` -> "module:class". Built lazily: a format's reader
#: is imported only by the product that uses it.
FORMATS = {
    "knmi_radar_3d": "meteopack.jobs.stream_kdp:Knmi3dFormat",
    "opera_odim": "meteopack.sources.radar.opera:OperaFormat",
}


def format_for(cfg: ProductConfig, settings: Settings):
    import importlib

    name = str(cfg.extra.get("stream_format", "knmi_radar_3d"))
    if name not in FORMATS:
        raise ValueError(f"{cfg.store_id}: unknown stream_format {name!r}; known: "
                         f"{sorted(FORMATS)}")
    module, _, attr = FORMATS[name].partition(":")
    return getattr(importlib.import_module(module), attr)(cfg, settings)


def make_client(settings: Settings, requests_per_second: float | None = None):
    """The remote client the settings describe."""
    if settings.remote == "s3":
        from meteopack.core import s3

        return s3.S3Client(settings.base_url, settings.bucket,
                           requests_per_second=requests_per_second or settings.default_rate)
    # An explicit rate applies to whichever key is in use.
    rates = {} if requests_per_second else settings.rates
    return kdp.KdpClient(
        settings.dataset, settings.version,
        key_files=settings.key_files, base_url=settings.base_url,
        rates=rates, default_rate=requests_per_second or settings.default_rate,
        reserve=settings.reserve)


class Stream:
    """The per-day loop. One instance per process; one process per store (locked)."""

    def __init__(self, cfg: ProductConfig, root: Path, *,
                 client: kdp.KdpClient | None = None,
                 download_workers: int | None = None,
                 pack_workers: int | None = None,
                 redo: bool = False,
                 requests_per_second: float | None = None,
                 fmt=None) -> None:
        self.cfg = cfg
        self.root = Path(root)
        self.settings = Settings.from_config(cfg)
        self.fmt = fmt or format_for(cfg, self.settings)
        self.client = client or make_client(self.settings, requests_per_second)
        self.download_workers = download_workers or self.settings.download_workers
        self.pack_workers = pack_workers or max(1, min(8, os.cpu_count() or 1))
        self.redo = redo
        self.target = self.fmt.target
        #: Field name -> target encoding. One field unless the format packs several.
        self.variables: dict[str, Encoding] = dict(
            getattr(self.fmt, "variables", None) or {cfg.variables[0].name: self.target})
        self.var = next(iter(self.variables))
        self.cadence = cfg.cadence
        self.per_day = int(timedelta(days=1) / self.cadence)
        self.pinned = cfg.schema_fingerprint
        #: Every fingerprint that is not drift (a format may pin several).
        self.pinned_all: frozenset[str] = frozenset(self.fmt.pinned)
        self._readme_written = False

    # ------------------------------------------------------------ download

    def _slots_wanted(self, day: datetime) -> set[datetime] | None:
        """Time steps still to pack, or None for all of them."""
        if self.redo:
            return None
        status = read_status(self.cfg, self.root, day)
        if not status or not status.get("packed_times"):
            return None
        done = {datetime.fromisoformat(t) for t in status["packed_times"]}
        return set(r3d.day_slots(day, self.cadence)) - done

    def fetch_day(self, day: datetime) -> Fetched:
        t0 = time.perf_counter()
        self.client.refresh_key()
        before = self.client.requests_made
        out = Fetched(day=day, key_file=str(self.client.key.path if self.client.key else ""))
        t1 = time.perf_counter()
        files = self.fmt.list_day(self.client, day)
        out.list_s = time.perf_counter() - t1
        out.listed.update(self.fmt.select(files, day))
        wanted = self._slots_wanted(day)
        todo = {w: f for w, f in out.listed.items() if wanted is None or w in wanted}
        out.skipped_known = len(out.listed) - len(todo)
        target = raw_dir(self.cfg, self.root, day)
        target.mkdir(parents=True, exist_ok=True)

        stats0 = dict(getattr(self.client, "stats", {}) or {})
        t_dl = time.perf_counter()

        fetch = getattr(self.fmt, "download", None)

        def one(item):
            when, remote = item
            s = time.perf_counter()
            if fetch is not None:  # a bundle: the format knows how to fetch it
                raw = fetch(self.client, remote, target)
            else:
                raw = self.client.download(remote, target / remote.filename)
            return when, raw, time.perf_counter() - s

        with cf.ThreadPoolExecutor(self.download_workers, thread_name_prefix="dl") as pool:
            futures = {pool.submit(one, item): item for item in sorted(todo.items())}
            for fut in cf.as_completed(futures):
                when, remote = futures[fut]
                try:
                    _, raw, seconds = fut.result()
                except kdp.CredentialError:
                    raise
                except Exception as exc:  # noqa: BLE001 - recorded, day becomes partial
                    out.failed[remote.filename] = f"{type(exc).__name__}: {exc}"
                    logger.warning("stream: download of %s failed: %s", remote.filename, exc)
                    continue
                out.paths[when] = Path(raw.path)
                if raw.attempts:
                    out.download_s.append(seconds)
                    out.bytes += raw.n_bytes
        out.dl_wall_s = time.perf_counter() - t_dl
        stats1 = dict(getattr(self.client, "stats", {}) or {})
        out.url_s = stats1.get("url_s", 0.0) - stats0.get("url_s", 0.0)
        out.transfer_s = stats1.get("transfer_s", 0.0) - stats0.get("transfer_s", 0.0)
        out.requests = self.client.requests_made - before
        out.wall_s = time.perf_counter() - t0
        return out

    # ---------------------------------------------------------------- pack

    def _ensure_store(self, day: datetime) -> None:
        end = day + timedelta(days=1)
        origin = self.settings.origin
        if day < origin:
            raise ValueError(f"{day:%Y-%m-%d} is before the store's origin {origin:%Y-%m-%d}")
        path = gridded.zarr_path(self.root, self.cfg.store_id)
        if not self._readme_written:
            from meteopack.core import readme

            readme.store_readme(self.cfg, store_dir(self.cfg, self.root))
            self._readme_written = True
        if not (path / "zarr.json").exists():
            gridded.initialise(self.cfg, self.root, Timeline(origin, end, self.cadence))
            gridded.refresh_attrs(self.cfg, self.root, layout_names=self.fmt.layout_names)
            extra_attrs = self.fmt.store_attrs()
            array_attrs = getattr(self.fmt, "array_attrs", dict)()
            if extra_attrs or array_attrs:
                import zarr

                group = zarr.open_group(path, mode="r+")
                group.attrs.update(extra_attrs)
                for name, attrs in array_attrs.items():
                    group[name].attrs.update(attrs)
        else:
            gridded.extend(self.cfg, self.root, Timeline(origin, end, self.cadence))

    def pack_day(self, fetched: Fetched) -> dict:
        day = fetched.day
        t0 = time.perf_counter()
        slots = r3d.day_slots(day, self.cadence)
        wanted = self._slots_wanted(day)
        fresh = wanted is None
        write_slots = slots if fresh else sorted(wanted)
        index = {w: i for i, w in enumerate(write_slots)}
        n = len(write_slots)

        codes = {name: np.full((n, *self.fmt.frame_shape), enc.missing, dtype=enc.dtype)
                 for name, enc in self.variables.items()}
        missing = np.ones(n, dtype=bool)
        corrupt = np.zeros(n, dtype=bool)
        layout = np.full(n, r3d.LAYOUT_MISSING, dtype="int16")
        record = np.full(n, -1, dtype="int64")
        # The product's own per-frame values; a frame not packed keeps the fill.
        frames = {v.name: np.full(n, v.fill_value, dtype=v.dtype)
                  for v in self.cfg.frame_variables}
        checks: dict[datetime, r3d.FileCheck] = {}
        pack_s: list[float] = []
        provenance: list[dict] = []

        def one(item):
            when, path = item
            return when, self.fmt.pack(path, when)

        jobs = [(w, p) for w, p in sorted(fetched.paths.items()) if w in index]
        with cf.ThreadPoolExecutor(self.pack_workers, thread_name_prefix="pack") as pool:
            for when, packed in pool.map(one, jobs):
                i = index[when]
                checks[when] = packed.check
                pack_s.append(packed.read_s)
                remote = fetched.listed[when]
                record[i] = gridded.record_id(remote.filename)
                if packed.codes is None:
                    corrupt[i] = True
                    missing[i] = False
                    layout[i] = r3d.LAYOUT_CORRUPT
                    logger.error("stream: %s NOT packed: %s", remote.filename,
                                 "; ".join(packed.check.errors))
                else:
                    # A dict for a multi-field format; a field it lacks stays missing.
                    fields = packed.codes if isinstance(packed.codes, dict) \
                        else {self.var: packed.codes}
                    for name, frame in fields.items():
                        if frame is not None:
                            codes[name][i] = frame
                    missing[i] = False
                    layout[i] = self.fmt.layout(packed)
                    for name, value in self.fmt.frame_values(packed).items():
                        frames[name][i] = value
                provenance.append({
                    "source_record_id": int(record[i]), "filename": remote.filename,
                    "first_time": when.replace(tzinfo=None), "n_bytes": remote.size,
                    "last_modified": remote.last_modified or "",
                    **self.fmt.provenance(packed),
                    "schema_fingerprint": packed.check.fingerprint or "",
                    "packed": packed.codes is not None,
                    "errors": "; ".join(packed.check.errors),
                })
        t_packed = time.perf_counter()

        # Unwritten steps of a re-run (still missing upstream) are left alone.
        keep = np.ones(n, dtype=bool) if fresh else ~missing
        result = None
        write_s = 0.0
        if keep.any():
            import xarray as xr

            self._ensure_store(day)
            times = np.array([np.datetime64(w.replace(tzinfo=None), "ns")
                              for w in write_slots])[keep]
            dims = gridded.dims_for(self.cfg)
            ds = xr.Dataset(
                # A view, not a copy, when the whole day is written: 4.9 GB.
                {**{name: (dims, block if keep.all() else block[keep])
                    for name, block in codes.items()},
                 "is_missing_frame": (("time",), missing[keep]),
                 "is_corrupt_frame": (("time",), corrupt[keep]),
                 "source_layout_id": (("time",), layout[keep]),
                 "source_record_id": (("time",), record[keep]),
                 **{name: (("time",), values[keep]) for name, values in frames.items()}},
                coords={"time": times})
            t_w = time.perf_counter()
            result = gridded.write(ds, self.cfg, self.root)
            write_s = time.perf_counter() - t_w
            if result.outside or result.off_cadence or result.written != int(keep.sum()):
                raise RuntimeError(f"{day:%Y-%m-%d}: wrote {result.written} of {int(keep.sum())}"
                                   f" frames ({result.outside} outside the axis, "
                                   f"{result.off_cadence} off cadence)")
            self._verify(day, write_slots, keep, codes, missing, corrupt)
            if provenance:
                gridded.write_provenance_table(self.root, self.cfg.store_id, provenance)
        del codes

        status = self._status(fetched, slots, checks, pack_s, write_s,
                              time.perf_counter() - t0, t_packed - t0)
        self._clean_raw(fetched, checks)
        _write_json(status_path(self.cfg, self.root, day), status)
        self._update_manifest(status, checks)
        return status

    def _verify(self, day, write_slots, keep, codes, missing, corrupt) -> None:
        """Read back the day's flags and two frames: a write that lost data fails here."""
        import zarr

        group = zarr.open_group(gridded.zarr_path(self.root, self.cfg.store_id), mode="r")
        timeline = Timeline(self.settings.origin, day + timedelta(days=1), self.cadence)
        stamps = np.array([np.datetime64(w.replace(tzinfo=None), "ns") for w in write_slots])
        rows = timeline.index_of(stamps)
        got_missing = np.asarray(group["is_missing_frame"].get_coordinate_selection(rows[keep]))
        got_corrupt = np.asarray(group["is_corrupt_frame"].get_coordinate_selection(rows[keep]))
        if not (np.array_equal(got_missing, missing[keep])
                and np.array_equal(got_corrupt, corrupt[keep])):
            raise RuntimeError(f"{day:%Y-%m-%d}: provenance flags did not read back as written")
        packed = np.flatnonzero(keep & ~missing & ~corrupt)
        for i in {int(packed[0]), int(packed[-1])} if packed.size else ():
            for name, block in codes.items():
                back = np.asarray(group[name][int(rows[i])])
                if not np.array_equal(back, block[i]):
                    raise RuntimeError(f"{day:%Y-%m-%d}: {name} at {write_slots[i]} did not "
                                       f"read back")

    def _shard_bytes(self, day: datetime) -> int:
        row = (day - self.settings.origin) // self.cadence
        shard = row // self.per_day
        total = 0
        for name in self.variables:
            base = gridded.zarr_path(self.root, self.cfg.store_id) / name / "c" / str(shard)
            if base.exists():
                total += sum(p.stat().st_size for p in base.rglob("*") if p.is_file())
        return total

    def _status(self, fetched: Fetched, slots, checks, pack_s, write_s, wall_s, pack_wall_s):
        previous = None if self.redo else read_status(self.cfg, self.root, fetched.day)
        packed = {datetime.fromisoformat(t) for t in (previous or {}).get("packed_times", [])}
        packed |= {w for w, c in checks.items() if c.ok}
        schema_failed = {c.name: c.errors for c in checks.values() if not c.ok}
        # Single files of a bundle that failed while the rest was packed.
        for c in checks.values():
            schema_failed.update(getattr(c, "file_errors", None) or {})
        missing_upstream = [w for w in slots if w not in fetched.listed]
        not_packed = [w for w in slots if w in fetched.listed and w not in packed]
        fingerprints: dict[str, int] = {}
        for c in checks.values():
            if c.fingerprint:
                fingerprints[c.fingerprint] = fingerprints.get(c.fingerprint, 0) + 1
        drift = sorted(fp for fp in fingerprints if self.pinned_all and fp not in self.pinned_all)
        storage: dict[str, int] = {}
        for c in checks.values():
            if c.storage:
                storage[c.storage] = storage.get(c.storage, 0) + 1
        scales = [c.scale_factor for c in checks.values() if c.scale_factor is not None]
        if len(packed) == len(slots):
            state = "done"
        elif not fetched.listed:
            state = "empty"
        elif packed:
            state = "partial"
        else:
            state = "failed"
        n_packed_now = sum(1 for c in checks.values() if c.ok)
        n_dl = len(fetched.download_s)
        return {
            "date": f"{fetched.day:%Y-%m-%d}",
            "state": state,
            "expected": len(slots),
            "listed": len(fetched.listed),
            "packed": len(packed),
            "packed_this_run": n_packed_now,
            "missing_upstream": [f"{w:%H%M}" for w in missing_upstream],
            "listed_not_packed": [f"{w:%H%M}" for w in not_packed],
            "failed_downloads": fetched.failed,
            "schema_failed": schema_failed,
            "schema_fingerprints": fingerprints,
            "schema_drift": drift,
            "storage_layouts": storage,
            "scale_factor_min": min(scales) if scales else None,
            "scale_factor_max": max(scales) if scales else None,
            "packed_times": sorted(w.isoformat() for w in packed),
            "key_file": fetched.key_file,
            "api_requests": fetched.requests + 0,
            "timings": {
                "list_s": round(fetched.list_s, 3),
                "fetch_wall_s": round(fetched.wall_s, 2),
                "downloaded_files": n_dl,
                "download_s_per_file_mean": round(sum(fetched.download_s) / n_dl, 3)
                if n_dl else None,
                "download_mb": round(fetched.bytes / 1e6, 1),
                # Per file: waiting for a URL (the key's rate and quota) vs
                # moving the bytes (the network), and the day's aggregate rate.
                "url_s_per_file_mean": round(fetched.url_s / n_dl, 3) if n_dl else None,
                "transfer_s_per_file_mean": round(fetched.transfer_s / n_dl, 3)
                if n_dl else None,
                "download_wall_s": round(fetched.dl_wall_s, 2),
                "download_mb_per_s": round(fetched.bytes / 1e6 / fetched.dl_wall_s, 1)
                if fetched.dl_wall_s > 0 and n_dl else None,
                "pack_s_per_file_mean": round(sum(pack_s) / len(pack_s), 3) if pack_s else None,
                "pack_wall_s": round(pack_wall_s, 2),
                "write_s": round(write_s, 2),
                "day_wall_s": round(wall_s, 2),
            },
            "shard_bytes": self._shard_bytes(fetched.day),
            **(getattr(self.fmt, "status_extra", lambda c: {})(checks)),
            "code_version": manifest.code_version(),
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    def _clean_raw(self, fetched: Fetched, checks) -> None:
        """Delete the day's raw files; keep the ones that failed their check, aside."""
        quarantine = layer_path(self.root, "raw", self.cfg.source, self.cfg.product) / "quarantine"
        for when, path in fetched.paths.items():
            check = checks.get(when)
            if path.is_dir():
                # A bundle: a failed time step goes aside whole; otherwise the
                # files that failed go to quarantine/<bundle>/ and the rest is deleted.
                if check is not None and not check.ok:
                    quarantine.mkdir(parents=True, exist_ok=True)
                    with contextlib.suppress(OSError):
                        shutil.move(str(path), quarantine / path.name)
                    continue
                bad = set(getattr(check, "quarantine", None) or ())
                for item in sorted(path.iterdir()):
                    if item.name in bad:
                        (quarantine / path.name).mkdir(parents=True, exist_ok=True)
                        with contextlib.suppress(OSError):
                            shutil.move(str(item), quarantine / path.name / item.name)
                    elif item.is_file():
                        item.unlink(missing_ok=True)
                with contextlib.suppress(OSError):
                    path.rmdir()
            elif check is not None and not check.ok:
                quarantine.mkdir(parents=True, exist_ok=True)
                with contextlib.suppress(OSError):
                    shutil.move(str(path), quarantine / path.name)
            else:
                path.unlink(missing_ok=True)
        directory = raw_dir(self.cfg, self.root, fetched.day)
        if directory.exists():
            for leftover in directory.glob("*.part"):
                leftover.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                directory.rmdir()

    def _update_manifest(self, status: dict, checks) -> None:
        store = store_dir(self.cfg, self.root)
        old = manifest.read(store)
        extra = dict(old.extra) if old else {}
        days = dict(extra.get("days") or {})
        days[status["date"]] = {k: status[k] for k in ("state", "packed", "listed", "expected")}
        days[status["date"]]["corrupt"] = len(status["schema_failed"])
        days[status["date"]]["shard_bytes"] = status["shard_bytes"]
        layouts = dict(extra.get("storage_layouts") or {})
        for name, count in status.get("storage_layouts", {}).items():
            entry = layouts.setdefault(name, {"files": 0, "first_day": status["date"],
                                              "last_day": status["date"]})
            entry["files"] += count
            entry["first_day"] = min(entry["first_day"], status["date"])
            entry["last_day"] = max(entry["last_day"], status["date"])
        schemas = dict(extra.get("schema_fingerprints") or {})
        for c in checks.values():
            if not c.fingerprint:
                continue
            entry = schemas.setdefault(c.fingerprint, {
                "count": 0, "first": c.name, "last": c.name, "descriptor": c.descriptor,
                "pinned": c.fingerprint in self.pinned_all})
            entry["count"] += 1
            entry["first"] = min(entry["first"], c.name)
            entry["last"] = max(entry["last"], c.name)
        extra.update({
            "days": dict(sorted(days.items())),
            "schema_fingerprints": schemas,
            "storage_layouts": layouts,
            "pinned_schema_fingerprint": self.pinned,
            "grid_id": self.cfg.grid_id,
            **self.fmt.manifest_extra(),
        })
        n_records = sum(int(d.get("packed", 0)) for d in days.values())
        times = sorted(days)
        m = manifest.Manifest.new(self.cfg.store_id, "packed", "gridded", self.cfg,
                                  n_records=n_records, extra=extra,
                                  time_min=times[0] if times else None,
                                  time_max=times[-1] if times else None)
        if old is not None:
            m.created_at = old.created_at
        manifest.write(store, m)


@contextlib.contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(f"another stream holds {path}; one stream per store") from None
        yield


def run(cfg: ProductConfig, root: Path, start: datetime, end: datetime, *,
        download_workers: int | None = None, pack_workers: int | None = None,
        redo: bool = False, time_budget_s: float | None = None, max_days: int | None = None,
        stop: threading.Event | None = None, client: kdp.KdpClient | None = None,
        requests_per_second: float | None = None,
        log: Callable[[str], None] = print) -> int:
    """Stream every day in ``[start, end)``. See the module docstring for exit codes."""
    t_start = time.monotonic()
    stop = stop or threading.Event()
    try:
        stream = Stream(cfg, root, client=client, download_workers=download_workers,
                        pack_workers=pack_workers, redo=redo,
                        requests_per_second=requests_per_second)
    except kdp.CredentialError as exc:
        log(f"error: {exc}")
        return EXIT_CREDENTIALS
    first = max(_day(start), stream.settings.origin)
    days = []
    d = first
    while d < end:
        days.append(d)
        d += timedelta(days=1)

    def todo(day: datetime) -> bool:
        if redo:
            return True
        status = read_status(cfg, root, day)
        return not status or status.get("state") != "done"

    pending = [d for d in days if todo(d)]
    if max_days is not None:
        pending = pending[:max_days]
    log(f"stream {cfg.store_id}: {len(days)} day(s) in {first:%Y-%m-%d}..{end:%Y-%m-%d}, "
        f"{len(pending)} to do; key file {stream.client.key.path}")

    schema_days = 0
    code = EXIT_DONE
    with _locked(store_dir(cfg, root) / LOCK_NAME), \
            cf.ThreadPoolExecutor(1, thread_name_prefix="prefetch") as prefetcher:
        future: cf.Future | None = None
        for i, day in enumerate(pending):
            if stop.is_set() or (time_budget_s and time.monotonic() - t_start > time_budget_s):
                code = EXIT_MORE
                break
            try:
                fetched = future.result() if future is not None else stream.fetch_day(day)
            except kdp.CredentialError as exc:
                log(f"error: {exc}")
                return EXIT_CREDENTIALS
            except Exception as exc:  # noqa: BLE001 - listing failed: stop, rerun resumes
                log(f"error: {day:%Y-%m-%d}: {type(exc).__name__}: {exc}")
                return EXIT_ERROR
            future = None
            # Download the next day while this one packs.
            last_one = i + 1 >= len(pending)
            out_of_time = bool(time_budget_s) and time.monotonic() - t_start > time_budget_s
            if not last_one and not stop.is_set() and not out_of_time:
                future = prefetcher.submit(stream.fetch_day, pending[i + 1])
            try:
                status = stream.pack_day(fetched)
            except Exception as exc:  # noqa: BLE001 - reported; the rerun redoes the day
                log(f"error: packing {day:%Y-%m-%d} failed: {type(exc).__name__}: {exc}")
                if future is not None:
                    with contextlib.suppress(Exception):
                        future.result()
                return EXIT_ERROR
            t = status["timings"]
            log(f"{status['date']} {status['state']:8s} packed {status['packed']:3d}/"
                f"{status['expected']} listed {status['listed']:3d} "
                f"missing {len(status['missing_upstream']):3d} "
                f"failed_dl {len(status['failed_downloads'])} "
                f"schema_fail {len(status['schema_failed'])} "
                f"req {status['api_requests']} dl {t['download_mb']} MB "
                f"dl/file {t['download_s_per_file_mean']} s (url {t['url_s_per_file_mean']} s, "
                f"transfer {t['transfer_s_per_file_mean']} s, {t['download_mb_per_s']} MB/s) "
                f"pack/file "
                f"{t['pack_s_per_file_mean']} s write {t['write_s']} s "
                f"shard {status['shard_bytes'] / 1e6:.1f} MB day {t['day_wall_s']} s "
                f"(+fetch {t['fetch_wall_s']} s) quota {stream.client.limiter.state}")
            if status["schema_failed"] or status["schema_drift"]:
                schema_days += 1 if status["schema_failed"] else 0
                log(f"SCHEMA: {status['date']}: {len(status['schema_failed'])} file(s) failed "
                    f"their check, fingerprints {status['schema_fingerprints']} "
                    f"(pinned {sorted(stream.pinned_all)})")
                if schema_days >= MAX_SCHEMA_DAYS:
                    log(f"error: schema failures on {schema_days} consecutive days; stopping. "
                        f"See {status_path(cfg, root, day)}")
                    if future is not None:
                        with contextlib.suppress(Exception):
                            future.result()
                    return EXIT_SCHEMA
            else:
                schema_days = 0
        if future is not None:
            # Stopped early with a prefetch in flight: let it finish so its
            # files are whole on disk for the next run.
            with contextlib.suppress(Exception):
                future.result()
            code = EXIT_MORE
    remaining = [d for d in days if todo(d)]
    if code == EXIT_DONE and remaining and max_days is None:
        # Days that ran but are not `done` are partial/empty/failed: finished
        # as far as this window goes. Only days never attempted count as more.
        never = [d for d in remaining if read_status(cfg, root, d) is None]
        code = EXIT_MORE if never else EXIT_DONE
    log(f"stream {cfg.store_id}: exit {code}; {len(days) - len(remaining)} of {len(days)} "
        f"day(s) done, {len(remaining)} not done; {stream.client.requests_made} API request(s)")
    return code


__all__ = ["EXIT_CREDENTIALS", "EXIT_DONE", "EXIT_ERROR", "EXIT_MORE", "EXIT_SCHEMA",
           "FORMATS", "Knmi3dFormat", "Settings", "Stream", "format_for", "make_client",
           "read_status", "run", "status_path"]
