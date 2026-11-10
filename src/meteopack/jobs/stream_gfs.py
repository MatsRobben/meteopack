"""Stream NOAA GFS 0.25 degree forecasts into two small gridded stores, one UTC day at a time.

``meteopack stream noaa_gfs --product state_1h_0p25`` fills both:

Both keep GFS's native 0.25 degree latitude/longitude grid, cropped to a fixed
window (``grid_id``); nothing is interpolated in the store.

* **state** (the product named): an hourly valid-time series of instantaneous
  fields. Valid hour H comes from run ``floor6(H)`` at lead ``H - floor6(H)``
  (0-5 h), i.e. from the newest run started at or before H. When that run's
  file for the lead is absent upstream, the frame comes from the previous run
  at lead + 6 and is flagged (`is_fallback`); per frame the store keeps the run
  and lead it came from (`source_run_time`, `source_lead_h`).
* **precipitation** (``extra.gfs.prate_product``): per run (time = run start,
  every 6 h), the one-hour mean precipitation rate over each lead hour
  ``(L-1, L]``, one variable per lead (``prate_f01`` ...), de-averaged from
  GFS's 6-hourly-resetting averages (`meteopack.sources.nwp.gfs.hourly_means`).

Per day: read the ``.idx`` of every file needed (4 runs; state leads 0-5,
precipitation leads 1-15) -> fetch only the wanted messages with byte-range
reads (adjacent messages in one request) -> check every message's schema ->
decode -> crop to the store grid -> encode -> write both stores -> read
them back and compare -> status record. Raw bytes live only in memory; a
message that fails its check is written to ``raw/<category>/<source>/<product>/
quarantine/`` with the reason and never packed.

Resumable and idempotent: a day whose status is ``done`` is skipped; any other
day is redone whole (a day is ~2 GB of transfer, about a minute). Writes are
region writes, so redoing a day rewrites the same codes. One process per store
pair (a lock in the state store).

A day is ``done`` when every frame is packed or confirmed absent upstream
(fields absent from an index, files absent from the bucket - the archive's own
gaps); ``partial`` when a transfer failed (rerun it); ``empty`` when nothing
for the day exists upstream. Exit codes as `meteopack.jobs.stream_kdp`.
"""
from __future__ import annotations

import concurrent.futures as cf
import contextlib
import json
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from meteopack.core import grids
from meteopack.core.config import ProductConfig, layer_path
from meteopack.core.encoding import Encoding
from meteopack.core.http import FetchError
from meteopack.core.store import gridded, manifest
from meteopack.core.timeline import Timeline
from meteopack.jobs.stream_kdp import (
    EXIT_DONE,
    EXIT_ERROR,
    EXIT_MORE,
    EXIT_SCHEMA,
    LOCK_NAME,
    MAX_SCHEMA_DAYS,
    STATUS_DIR,
    _locked,
    _write_json,
)
from meteopack.sources.nwp import gfs

logger = logging.getLogger(__name__)

HOUR = timedelta(hours=1)
CYCLE = timedelta(hours=gfs.CYCLE_HOURS)
#: Key of the precipitation-rate message (both its instantaneous and mean forms).
PRATE_KEY = ("PRATE", "surface")


@dataclass(slots=True)
class Settings:
    endpoint: str
    bucket: str
    origin: datetime
    state_leads: list[int]
    prate_leads: list[int]
    fallback: bool
    grid: gfs.SourceGrid
    download_workers: int
    requests_per_second: float
    prate_product: str
    prate_factor: float
    prate_signature: gfs.GribSignature

    @classmethod
    def from_config(cls, cfg: ProductConfig) -> Settings:
        spec = dict(cfg.extra.get("gfs") or {})
        s3 = dict(cfg.extra.get("s3") or {})
        if not spec or not s3 or not cfg.extra.get("timeline_origin"):
            raise ValueError(f"{cfg.store_id}: a GFS stream needs extra.gfs, extra.s3 and "
                             f"extra.timeline_origin")
        origin = datetime.fromisoformat(str(cfg.extra["timeline_origin"]))
        prate = dict(spec.get("prate") or {})
        return cls(
            endpoint=str(s3.get("endpoint", gfs.ENDPOINT)),
            bucket=str(s3.get("bucket", gfs.BUCKET)),
            origin=origin if origin.tzinfo else origin.replace(tzinfo=UTC),
            state_leads=[int(x) for x in spec.get("state_leads", range(6))],
            prate_leads=[int(x) for x in prate.get("leads", range(1, 16))],
            fallback=bool(spec.get("fallback_previous_run", True)),
            grid=gfs.SourceGrid.from_config(spec.get("source_grid")),
            download_workers=int(s3.get("download_workers", 16)),
            requests_per_second=float(s3.get("requests_per_second", 100.0)),
            prate_product=str(prate.get("product", "")),
            prate_factor=float(prate.get("factor", 3600.0)),
            prate_signature=gfs.GribSignature.parse(prate.get("grib", [0, 1, 7, 1, 0, 255, None])),
        )


@dataclass(slots=True)
class RunData:
    """What one run gave: cropped fields per lead, and what was absent or failed."""

    run: datetime
    layout: int | None = None
    #: lead -> field name -> cropped physical values (NaN = undefined here).
    state: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    #: leads whose pgrb2 index exists upstream.
    present: set[int] = field(default_factory=set)
    #: lead -> fields not in the index (absent upstream).
    absent_fields: dict[int, list[str]] = field(default_factory=dict)
    #: lead -> fields whose message failed its check.
    failed_fields: dict[int, list[str]] = field(default_factory=dict)
    #: lead -> (window start, cropped mean rate in the store's units).
    prate_avg: dict[int, tuple[int, np.ndarray]] = field(default_factory=dict)
    prate_absent: set[int] = field(default_factory=set)
    #: transfer failures: "fFFF kind" -> message. A run with any is not packed.
    errors: dict[str, str] = field(default_factory=dict)
    #: files whose .idx did not match them; their messages were located by header
    stale_indexes: list[str] = field(default_factory=list)
    schema_failures: list[dict] = field(default_factory=list)
    fingerprints: Counter = field(default_factory=Counter)
    bytes: int = 0
    requests: int = 0
    messages: int = 0
    idx_seen: dict[tuple[int, str], bool] = field(default_factory=dict)

    def lead_ok(self, lead: int) -> bool:
        return lead in self.present and not any(k.startswith(f"f{lead:03d} ")
                                                 for k in self.errors)


@dataclass(slots=True)
class _Want:
    """One message to fetch and what it is for."""

    run: datetime
    lead: int
    kind: str
    entry: gfs.IdxEntry
    spec: gfs.FieldSpec | None    # None: the precipitation mean
    average: bool


class GfsStream:
    """Fetch, check, crop and pack. One instance per process."""

    def __init__(self, state_cfg: ProductConfig, prate_cfg: ProductConfig, root: Path, *,
                 client=None, download_workers: int | None = None, redo: bool = False,
                 requests_per_second: float | None = None) -> None:
        self.cfg = state_cfg
        self.pcfg = prate_cfg
        self.root = Path(root)
        self.settings = Settings.from_config(state_cfg)
        self.redo = redo
        self.specs = gfs.field_specs(state_cfg)
        self.prate_vars = [v.name for v in prate_cfg.variables]
        self.prate_enc = {v.name: Encoding.from_config(v.encoding, name=v.name)
                          for v in prate_cfg.variables}
        leads_cfg = [int(v.name.rsplit("_f", 1)[1]) for v in prate_cfg.variables]
        if leads_cfg != self.settings.prate_leads:
            raise ValueError(f"{prate_cfg.store_id} variables are leads {leads_cfg}, "
                             f"extra.gfs.prate.leads says {self.settings.prate_leads}")
        self.prate_spec = gfs.FieldSpec(
            name="prate_mean", var=PRATE_KEY[0], level=PRATE_KEY[1],
            signature=self.settings.prate_signature, encoding=next(iter(self.prate_enc.values())),
            factor=self.settings.prate_factor, files=("pgrb2",))
        if state_cfg.grid_id != prate_cfg.grid_id:
            raise ValueError("the state and precipitation stores must share a grid")
        grid = grids.get(state_cfg.grid_id)
        lon, lat = grid.lonlat()
        self.crop = gfs.Crop(lon, lat, self.settings.grid)
        self.frame_shape = grid.shape
        if client is None:
            from meteopack.core import s3

            client = s3.S3Client(self.settings.endpoint, self.settings.bucket,
                                 requests_per_second=requests_per_second
                                 or self.settings.requests_per_second, check_hdf5=False)
            # One pooled connection per worker.
            import requests

            adapter = requests.adapters.HTTPAdapter(
                pool_connections=4,
                pool_maxsize=max(32, (download_workers or self.settings.download_workers) + 8))
            client.session.mount("https://", adapter)
        self.client = client
        self.download_workers = download_workers or self.settings.download_workers
        self.pool = cf.ThreadPoolExecutor(self.download_workers, thread_name_prefix="gfs")
        self.quarantine = layer_path(self.root, "raw", state_cfg.source, state_cfg.product) \
            / "quarantine"
        self._readme_written = False
        self._lock = threading.Lock()
        #: Where `run` will stop: the store axes are grown to here in one go.
        self.axis_end: datetime | None = None

    def close(self) -> None:
        self.pool.shutdown(wait=True)

    # ------------------------------------------------------------------ fetch

    def _index(self, run: datetime, lead: int, kind: str) -> tuple[list | None, int, str]:
        """The parsed index of one file, trying the expected path layout first."""
        primary = gfs.layout_for(run)
        other = gfs.LAYOUT_V15 if primary == gfs.LAYOUT_V16 else gfs.LAYOUT_V16
        for layout in (primary, other):
            key = gfs.file_key(run, lead, kind, layout)
            text = self.client.get_text(key + ".idx")
            if text is not None:
                return gfs.parse_idx(text), layout, key
        return None, primary, gfs.file_key(run, lead, kind, primary)

    def fetch_runs(self, wanted: dict[datetime, tuple[list[int], list[int]]]) -> dict:
        """Fetch runs: run -> (state leads, precipitation leads). Returns run -> RunData."""
        out = {run: RunData(run=run) for run in wanted}
        req0 = self.client.requests_made

        # Phase 1: the pgrb2 index of every file.
        jobs = {}
        for run, (s_leads, p_leads) in wanted.items():
            for lead in sorted(set(s_leads) | set(p_leads)):
                jobs[self.pool.submit(self._index, run, lead, "pgrb2")] = (run, lead, "pgrb2")
        indexes: dict[tuple[datetime, int, str], tuple[list | None, int, str]] = {}
        self._gather_indexes(jobs, indexes, out)

        # Phase 2: pgrb2b where a state field is not in pgrb2.
        jobs = {}
        for run, (s_leads, _) in wanted.items():
            for lead in s_leads:
                got = indexes.get((run, lead, "pgrb2"))
                if not got or got[0] is None:
                    continue
                if any(gfs.find_entry(got[0], s.var, s.level) is None and "pgrb2b" in s.files
                       for s in self.specs):
                    jobs[self.pool.submit(self._index, run, lead, "pgrb2b")] = \
                        (run, lead, "pgrb2b")
        self._gather_indexes(jobs, indexes, out)

        # Phase 3: choose messages, coalesce, fetch, check, decode, crop.
        groups: list[tuple[str, list[_Want]]] = []
        for run, (s_leads, p_leads) in wanted.items():
            data = out[run]
            for lead in sorted(set(s_leads) | set(p_leads)):
                a = indexes.get((run, lead, "pgrb2"))
                if not a or a[0] is None:
                    continue
                data.present.add(lead)
                data.layout = a[1] if data.layout is None else data.layout
                per_key: dict[str, list[_Want]] = {}
                if lead in s_leads:
                    for spec in self.specs:
                        hit = None
                        for kind in spec.files:
                            idx = indexes.get((run, lead, kind))
                            if idx and idx[0] is not None:
                                e = gfs.find_entry(idx[0], spec.var, spec.level)
                                if e is not None:
                                    hit = (kind, idx[2], e)
                                    break
                        if hit is None:
                            data.absent_fields.setdefault(lead, []).append(spec.name)
                            continue
                        per_key.setdefault(hit[1], []).append(
                            _Want(run, lead, hit[0], hit[2], spec, False))
                if lead in p_leads:
                    e = gfs.find_entry(a[0], *PRATE_KEY, average=True)
                    if e is None:
                        data.prate_absent.add(lead)
                    else:
                        per_key.setdefault(a[2], []).append(_Want(run, lead, "pgrb2", e, None,
                                                                  True))
                for key, wants in per_key.items():
                    by_entry = {w.entry.start: w for w in wants}
                    for group in gfs.coalesce([w.entry for w in wants]):
                        groups.append((key, [by_entry[e.start] for e in group]))

        stale = self._fetch_groups(groups, out)

        # A file whose .idx no longer matches it (re-uploaded upstream): walk the
        # file's own message headers instead, then fetch what is wanted again.
        if stale:
            rebuilt = list(self.pool.map(lambda kw: self._rebuild(*kw),
                                         [(key, wants) for key, wants in stale.items()]))
            groups = []
            for key, (fixed, absent) in zip(stale, rebuilt, strict=True):
                for w in absent:
                    target = out[w.run]
                    if w.spec is None:
                        target.prate_absent.add(w.lead)
                    else:
                        target.absent_fields.setdefault(w.lead, []).append(w.spec.name)
                if fixed is None:
                    continue
                out[stale[key][0].run].stale_indexes.append(key)
                by_entry = {w.entry.start: w for w in fixed}
                for group in gfs.coalesce([w.entry for w in fixed]):
                    groups.append((key, [by_entry[e.start] for e in group]))
            still = self._fetch_groups(groups, out)
            for key, wants in still.items():
                out[wants[0].run].errors[f"f{wants[0].lead:03d} {wants[0].kind}"] = \
                    "the file's messages do not match its own headers"
        for data in out.values():
            data.requests = 0
        # Requests are counted for the whole call; attribute them to the first run.
        if out:
            next(iter(out.values())).requests = self.client.requests_made - req0
        return out

    def _fetch_groups(self, groups, out) -> dict[str, list[_Want]]:
        """Fetch, check and crop groups; return the wants of files with a stale index."""
        futures = {self.pool.submit(self._fetch_group, key, wants): (key, wants)
                   for key, wants in groups}
        stale: dict[str, list[_Want]] = {}
        for fut in cf.as_completed(futures):
            key, wants = futures[fut]
            data = out[wants[0].run]
            try:
                results, n_bytes = fut.result()
            except gfs.StaleIndex:
                stale.setdefault(key, []).extend(wants)
                continue
            except Exception as exc:  # noqa: BLE001 - a transfer failure; the day is partial
                tag = f"f{wants[0].lead:03d} {wants[0].kind}"
                data.errors[tag] = f"{type(exc).__name__}: {exc}"
                logger.warning("gfs: %s %s failed: %s", key, tag, exc)
                continue
            data.bytes += n_bytes
            for want, check, values, window in results:
                data.messages += 1
                if check.fingerprint:
                    data.fingerprints[check.fingerprint] += 1
                name = want.spec.name if want.spec else "prate_mean"
                if values is None:
                    data.failed_fields.setdefault(want.lead, []).append(name)
                    data.schema_failures.append({
                        "run": f"{want.run:%Y%m%d%H}", "lead": want.lead, "kind": want.kind,
                        "field": name, "idx": f"{want.entry.var}:{want.entry.level}:"
                                              f"{want.entry.step}",
                        "errors": check.errors})
                    continue
                if want.spec is None:
                    data.prate_avg[want.lead] = (window[0], values)
                else:
                    data.state.setdefault(want.lead, {})[want.spec.name] = values
        return stale

    def _rebuild(self, key: str, wants: list[_Want]):
        """Locate `wants` in `key` from the file's own headers: (found wants, absent wants).

        Returns (None, []) if the file cannot be walked; the caller records that.
        """
        try:
            listed = [r for r in self.client.list_prefix(key) if r.key == key]
            if not listed:
                return None, []
            headers = gfs.scan_headers(lambda a, b: self.client.get_bytes(key, a, b),
                                       listed[0].size)
        except Exception as exc:  # noqa: BLE001 - reported as a failed file
            logger.warning("gfs: %s: could not walk the file's headers: %s", key, exc)
            return None, []
        logger.warning("gfs: %s: stale .idx; located %d messages from the file's own headers",
                       key, len(headers))
        found, absent = [], []
        for w in wants:
            spec = w.spec or self.prate_spec
            h = gfs.find_header(headers, spec.signature, w.lead, average=w.average)
            if h is None:
                absent.append(w)
                continue
            entry = gfs.IdxEntry(number="header", start=h.start, end=h.end,
                                 run=w.entry.run, var=w.entry.var, level=w.entry.level,
                                 step=w.entry.step)
            found.append(_Want(w.run, w.lead, w.kind, entry, w.spec, w.average))
        return found, absent

    def _gather_indexes(self, jobs, indexes, out) -> None:
        for fut in cf.as_completed(jobs):
            run, lead, kind = jobs[fut]
            try:
                indexes[(run, lead, kind)] = fut.result()
            except Exception as exc:  # noqa: BLE001 - a transfer failure; the day is partial
                out[run].errors[f"f{lead:03d} {kind}"] = f"index: {type(exc).__name__}: {exc}"
                logger.warning("gfs: index %s f%03d %s failed: %s", run, lead, kind, exc)
                continue
            out[run].idx_seen[(lead, kind)] = indexes[(run, lead, kind)][0] is not None

    def _fetch_group(self, key: str, wants: list[_Want]):
        """One ranged GET for adjacent messages; check, decode and crop each."""
        first, last = wants[0].entry, wants[-1].entry
        try:
            blob = self.client.get_bytes(key, first.start, last.end)
        except FetchError as exc:
            if exc.status == 416:            # the index points past the end of the file
                raise gfs.StaleIndex(key) from exc
            raise
        slices = []
        for w in wants:
            lo = w.entry.start - first.start
            hi = None if w.entry.end is None else w.entry.end - first.start + 1
            slices.append(blob[lo:hi])
        if not all(gfs.message_ok(m) for m in slices):
            raise gfs.StaleIndex(key)
        results = []
        for w, message in zip(wants, slices, strict=True):
            spec = w.spec or self.prate_spec
            check, values, window = gfs.check_message(
                message, spec, run=w.run, lead=w.lead, average=w.average,
                grid=self.settings.grid)
            if values is None:
                self._quarantine(w, message, check)
                results.append((w, check, None, None))
                continue
            out = self.crop(values) * spec.factor
            results.append((w, check, out, window))
        return results, len(blob)

    def _quarantine(self, want: _Want, message: bytes, check) -> None:
        self.quarantine.mkdir(parents=True, exist_ok=True)
        name = (f"{gfs.file_name(want.run, want.lead, want.kind)}."
                f"{(want.spec.name if want.spec else 'prate_mean')}")
        (self.quarantine / f"{name}.grib2").write_bytes(message)
        (self.quarantine / f"{name}.json").write_text(json.dumps(
            {"errors": check.errors, "descriptor": check.descriptor,
             "idx": f"{want.entry.var}:{want.entry.level}:{want.entry.step}"},
            indent=1, default=str))
        logger.error("gfs: %s failed its check, quarantined: %s", name, "; ".join(check.errors))

    def fetch_day(self, day: datetime) -> dict:
        """All runs a day needs, including the previous runs any fallback frame needs."""
        t0 = time.perf_counter()
        b0 = self.client.stats.get("bytes", 0) if hasattr(self.client, "stats") else 0
        r0 = self.client.requests_made
        s = self.settings
        runs = gfs.run_times(day)
        data = self.fetch_runs({r: (s.state_leads, s.prate_leads) for r in runs})
        fallback: dict[datetime, list[int]] = {}
        if s.fallback:
            for h in range(24):
                when = day + h * HOUR
                run = gfs.floor_run(when)
                lead = int((when - run) / HOUR)
                d = data[run]
                if not d.lead_ok(lead) and not any(k.startswith(f"f{lead:03d} ")
                                                   for k in d.errors):
                    prev = run - CYCLE
                    fallback.setdefault(prev, []).append(lead + gfs.CYCLE_HOURS)
        extra = self.fetch_runs({r: (sorted(set(leads)), []) for r, leads in fallback.items()}) \
            if fallback else {}
        return {"day": day, "runs": data, "fallback": extra,
                "wall_s": time.perf_counter() - t0,
                "bytes": (self.client.stats.get("bytes", 0) - b0)
                if hasattr(self.client, "stats") else sum(d.bytes for d in data.values()),
                "requests": self.client.requests_made - r0}

    # ------------------------------------------------------------------- pack

    def _ensure_store(self, cfg: ProductConfig, day: datetime) -> None:
        # Grow the axis to the end of the whole window at once: one resize of every
        # array per job instead of one per day (unwritten steps cost nothing).
        end = max(day + timedelta(days=1), self.axis_end or day)
        origin = self.settings.origin
        if day < origin:
            raise ValueError(f"{day:%Y-%m-%d} is before the store's origin {origin:%Y-%m-%d}")
        path = gridded.zarr_path(self.root, cfg.store_id)
        if not (path / "zarr.json").exists():
            gridded.initialise(cfg, self.root, Timeline(origin, end, cfg.cadence))
            gridded.refresh_attrs(cfg, self.root, layout_names=gfs.LAYOUT_NAMES)
        else:
            gridded.extend(cfg, self.root, Timeline(origin, end, cfg.cadence))

    def _readmes(self) -> None:
        if self._readme_written:
            return
        from meteopack.core import readme

        for cfg in (self.cfg, self.pcfg):
            readme.store_readme(cfg, gridded.store_path(self.root, cfg.store_id))
        self._readme_written = True

    def pack_day(self, fetched: dict) -> dict:
        t0 = time.perf_counter()
        day: datetime = fetched["day"]
        runs: dict[datetime, RunData] = fetched["runs"]
        extra: dict[datetime, RunData] = fetched["fallback"]
        ny, nx = self.frame_shape
        n_fields = len(self.specs)

        # ---- state: 24 valid hours
        codes = {s.name: np.full((24, ny, nx), s.encoding.missing, dtype=s.encoding.dtype)
                 for s in self.specs}
        missing = np.ones(24, dtype=bool)
        corrupt = np.zeros(24, dtype=bool)
        written = np.ones(24, dtype=bool)
        layout = np.full(24, gfs.LAYOUT_MISSING, dtype="int16")
        record = np.full(24, -1, dtype="int64")
        run_s = np.full(24, -1, dtype="int64")
        lead_h = np.full(24, -1, dtype="int16")
        is_fb = np.full(24, -1, dtype="int8")
        n_missing = np.full(24, -1, dtype="int16")
        clipped: Counter = Counter()
        hours: dict[str, dict] = {}
        provenance: list[dict] = []
        for h in range(24):
            when = day + h * HOUR
            run = gfs.floor_run(when)
            lead = int((when - run) / HOUR)
            src = runs[run]
            entry: dict = {"run": f"{run:%Y%m%d%H}", "lead": lead, "fallback": False}
            if any(k.startswith(f"f{lead:03d} ") for k in src.errors):
                written[h] = False
                entry["state"] = "transfer_failed"
                hours[f"{h:02d}"] = entry
                continue
            if not src.lead_ok(lead):
                prev = extra.get(run - CYCLE)
                if prev is not None and prev.lead_ok(lead + gfs.CYCLE_HOURS):
                    src, lead = prev, lead + gfs.CYCLE_HOURS
                    run = prev.run
                    entry.update(run=f"{run:%Y%m%d%H}", lead=lead, fallback=True)
                elif prev is not None and any(k.startswith(f"f{lead + gfs.CYCLE_HOURS:03d} ")
                                              for k in prev.errors):
                    written[h] = False
                    entry["state"] = "transfer_failed"
                    hours[f"{h:02d}"] = entry
                    continue
                else:
                    entry["state"] = "missing_upstream"
                    hours[f"{h:02d}"] = entry
                    continue
            fields = src.state.get(lead, {})
            for spec in self.specs:
                values = fields.get(spec.name)
                if values is None:
                    continue
                enc = gfs.encode_field(values, spec.encoding)
                codes[spec.name][h] = enc.codes
                if enc.n_below or enc.n_above:
                    clipped[spec.name] += enc.n_below + enc.n_above
            absent = src.absent_fields.get(lead, [])
            failed = src.failed_fields.get(lead, [])
            missing[h] = False
            corrupt[h] = bool(failed)
            layout[h] = src.layout or gfs.LAYOUT_MISSING
            name = gfs.file_name(run, lead)
            record[h] = gridded.record_id(name)
            run_s[h] = int(run.timestamp())
            lead_h[h] = lead
            is_fb[h] = int(entry["fallback"])
            n_missing[h] = n_fields - len(fields)
            entry.update(state="packed", fields_absent_upstream=absent,
                         fields_failed_check=failed)
            hours[f"{h:02d}"] = entry
            provenance.append({
                "source_record_id": int(record[h]), "filename": name,
                "first_time": when.replace(tzinfo=None), "run": run.replace(tzinfo=None),
                "lead_h": lead, "fallback": bool(entry["fallback"]),
                "n_fields": len(fields), "fields_absent_upstream": ",".join(absent),
                "fields_failed_check": ",".join(failed), "packed": True})

        # ---- precipitation: one frame per run
        p_runs = gfs.run_times(day)
        p_codes = {v: np.full((len(p_runs), ny, nx), self.prate_enc[v].missing,
                              dtype=self.prate_enc[v].dtype) for v in self.prate_vars}
        p_missing = np.ones(len(p_runs), dtype=bool)
        p_corrupt = np.zeros(len(p_runs), dtype=bool)
        p_written = np.ones(len(p_runs), dtype=bool)
        p_layout = np.full(len(p_runs), gfs.LAYOUT_MISSING, dtype="int16")
        p_record = np.full(len(p_runs), -1, dtype="int64")
        p_nmiss = np.full(len(p_runs), -1, dtype="int16")
        p_info: dict[str, dict] = {}
        p_prov: list[dict] = []
        neg = {"count": 0, "min": 0.0}
        for i, run in enumerate(p_runs):
            src = runs[run]
            leads = self.settings.prate_leads
            if any(int(k[1:4]) in leads for k in src.errors):
                p_written[i] = False
                p_info[f"{run:%Y%m%d%H}"] = {"state": "transfer_failed"}
                continue
            hourly = gfs.hourly_means(src.prate_avg)
            got = []
            for lead, var in zip(leads, self.prate_vars, strict=True):
                values = hourly.get(lead)
                if values is None:
                    continue
                n_neg = int((values < 0).sum())
                if n_neg:
                    neg["count"] += n_neg
                    neg["min"] = min(neg["min"], float(np.nanmin(values)))
                enc = gfs.encode_field(np.maximum(values, 0.0), self.prate_enc[var])
                p_codes[var][i] = enc.codes
                if enc.n_above:
                    clipped[var] += enc.n_above
                got.append(lead)
            lost = [lead for lead in leads if lead not in got]
            failed = src.failed_fields.get
            if got:
                p_missing[i] = False
                p_corrupt[i] = any("prate_mean" in (failed(lead) or []) for lead in leads)
                p_layout[i] = src.layout or gfs.LAYOUT_MISSING
                p_record[i] = gridded.record_id(f"gfs.{run:%Y%m%d%H}.prate")
                p_nmiss[i] = len(lost)
            p_info[f"{run:%Y%m%d%H}"] = {"state": "packed" if got else "missing_upstream",
                                         "leads_missing": lost}
            if got:
                p_prov.append({"source_record_id": int(p_record[i]),
                               "filename": f"gfs.{run:%Y%m%d%H}.pgrb2.0p25 PRATE ave",
                               "first_time": run.replace(tzinfo=None),
                               "leads": ",".join(map(str, got)),
                               "leads_missing": ",".join(map(str, lost)), "packed": True})
        t_packed = time.perf_counter()

        # ---- write, then read back
        self._readmes()
        write_s = 0.0
        tw = time.perf_counter()
        if written.any():
            self._write(self.cfg, day, [day + h * HOUR for h in range(24)], written, codes,
                        missing, corrupt, layout, record,
                        {"source_run_time": run_s, "source_lead_h": lead_h,
                         "is_fallback": is_fb, "n_fields_missing": n_missing})
            if provenance:
                gridded.write_provenance_table(self.root, self.cfg.store_id, provenance)
        if p_written.any():
            self._write(self.pcfg, day, p_runs, p_written, p_codes, p_missing, p_corrupt,
                        p_layout, p_record, {"n_leads_missing": p_nmiss})
            if p_prov:
                gridded.write_provenance_table(self.root, self.pcfg.store_id, p_prov)
        write_s = time.perf_counter() - tw

        status = self._status(fetched, hours, p_info, written, p_written, missing, clipped, neg,
                              t_packed - t0, write_s, time.perf_counter() - t0)
        _write_json(self.status_path(day), status)
        self._update_manifests(status)
        return status

    def _write(self, cfg, day, times, keep, codes, missing, corrupt, layout, record, frames):
        import xarray as xr

        self._ensure_store(cfg, day)
        stamps = np.array([np.datetime64(t.replace(tzinfo=None), "ns") for t in times])[keep]
        dims = gridded.dims_for(cfg)
        ds = xr.Dataset(
            {**{name: (dims, arr[keep]) for name, arr in codes.items()},
             "is_missing_frame": (("time",), missing[keep]),
             "is_corrupt_frame": (("time",), corrupt[keep]),
             "source_layout_id": (("time",), layout[keep]),
             "source_record_id": (("time",), record[keep]),
             **{name: (("time",), values[keep]) for name, values in frames.items()}},
            coords={"time": stamps})
        result = gridded.write(ds, cfg, self.root)
        if result.outside or result.off_cadence or result.written != int(keep.sum()):
            raise RuntimeError(f"{cfg.store_id} {day:%Y-%m-%d}: wrote {result.written} of "
                               f"{int(keep.sum())} frames ({result.outside} outside the axis, "
                               f"{result.off_cadence} off cadence)")
        self._verify(cfg, stamps, codes, keep, missing, frames)

    def _verify(self, cfg, stamps, codes, keep, missing, frames) -> None:
        """Read every written frame of every variable back and compare."""
        import zarr

        group = zarr.open_group(gridded.zarr_path(self.root, cfg.store_id), mode="r")
        origin = np.datetime64(self.settings.origin.replace(tzinfo=None), "ns")
        rows = ((stamps - origin) // np.timedelta64(cfg.cadence_seconds, "s")).astype("int64")
        lo, hi = int(rows.min()), int(rows.max()) + 1
        pick = rows - lo            # the written rows inside the day's range [lo, hi)

        def back(name):
            return np.asarray(group[name][lo:hi])[pick]

        if not np.array_equal(back("is_missing_frame"), missing[keep]):
            raise RuntimeError(f"{cfg.store_id}: is_missing_frame did not read back")
        for name, values in frames.items():
            if not np.array_equal(back(name), values[keep]):
                raise RuntimeError(f"{cfg.store_id}: {name} did not read back")
        for name, arr in codes.items():
            if not np.array_equal(back(name), arr[keep]):
                raise RuntimeError(f"{cfg.store_id}: {name} did not read back as written")

    # ----------------------------------------------------------------- status

    def status_path(self, day: datetime) -> Path:
        return gridded.store_path(self.root, self.cfg.store_id) / STATUS_DIR / f"{day:%Y}" \
            / f"{day:%Y-%m-%d}.json"

    def read_status(self, day: datetime) -> dict | None:
        try:
            return json.loads(self.status_path(day).read_text())
        except (OSError, ValueError):
            return None

    def _shard_bytes(self, cfg: ProductConfig, day: datetime) -> int:
        """Bytes on disk of the shard files holding this day, all variables."""
        base = gridded.zarr_path(self.root, cfg.store_id)
        per_shard = int((cfg.store.shard or {}).get("time", 86400 // cfg.cadence_seconds))
        row = int((day - self.settings.origin) / cfg.cadence)
        shard = row // per_shard
        total = 0
        for v in cfg.variables:
            p = base / v.name / "c" / str(shard)
            if p.is_dir():
                total += sum(q.stat().st_size for q in p.rglob("*") if q.is_file())
            elif p.exists():
                total += p.stat().st_size
        return total

    def _status(self, fetched, hours, p_info, written, p_written, missing, clipped, neg,
                pack_s, write_s, wall_s) -> dict:
        day = fetched["day"]
        runs = {**fetched["fallback"], **fetched["runs"]}
        errors = {f"{r:%Y%m%d%H} {k}": v for r, d in runs.items() for k, v in d.errors.items()}
        schema = [f for d in runs.values() for f in d.schema_failures]
        absent_files = sorted(f"{r:%Y%m%d%H} f{lead:03d} {kind}"
                              for r, d in runs.items() for (lead, kind), seen in d.idx_seen.items()
                              if not seen)
        fps: Counter = Counter()
        for d in runs.values():
            fps.update(d.fingerprints)
        n_packed = int((~missing & written).sum())
        if errors or not written.all() or not p_written.all():
            state = "partial"
        elif n_packed == 0 and all(i.get("state") != "packed" for i in p_info.values()):
            state = "empty"
        else:
            state = "done"
        mb = fetched["bytes"] / 1e6
        return {
            "date": f"{day:%Y-%m-%d}",
            "state": state,
            "state_frames_packed": n_packed,
            "state_frames_fallback": sum(1 for h in hours.values() if h.get("fallback")
                                         and h.get("state") == "packed"),
            "state_frames_missing_upstream": sorted(k for k, h in hours.items()
                                                    if h.get("state") == "missing_upstream"),
            "prate_runs_packed": sum(1 for i in p_info.values() if i.get("state") == "packed"),
            "hours": hours,
            "prate_runs": p_info,
            "files_absent_upstream": absent_files,
            "failed_downloads": errors,
            "stale_indexes": sorted(k for d in runs.values() for k in d.stale_indexes),
            "schema_failed": schema,
            "schema_fingerprints": dict(fps),
            "clipped_values": dict(clipped),
            "prate_negative_hourly_means": neg,
            "transfer": {"mb": round(mb, 1), "requests": fetched["requests"],
                         "messages": sum(d.messages for d in runs.values()),
                         "fetch_wall_s": round(fetched["wall_s"], 2),
                         "mb_per_s": round(mb / fetched["wall_s"], 1)
                         if fetched["wall_s"] > 0 else None},
            "timings": {"pack_s": round(pack_s, 2), "write_verify_s": round(write_s, 2),
                        "day_pack_wall_s": round(wall_s, 2)},
            "state_shard_bytes": self._shard_bytes(self.cfg, day),
            "prate_shard_bytes": self._shard_bytes(self.pcfg, day),
            "code_version": manifest.code_version(),
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    def _update_manifests(self, status: dict) -> None:
        for cfg, keys in ((self.cfg, ("state", "state_frames_packed", "state_frames_fallback")),
                          (self.pcfg, ("state", "prate_runs_packed"))):
            store = gridded.store_path(self.root, cfg.store_id)
            old = manifest.read(store)
            extra = dict(old.extra) if old else {}
            days = dict(extra.get("days") or {})
            days[status["date"]] = {k: status[k] for k in keys}
            fps = dict(extra.get("schema_fingerprints") or {})
            for fp, n in status["schema_fingerprints"].items():
                entry = fps.setdefault(fp, {"count": 0, "first": status["date"],
                                            "last": status["date"]})
                entry["count"] += n
                entry["first"] = min(entry["first"], status["date"])
                entry["last"] = max(entry["last"], status["date"])
            extra.update({"days": dict(sorted(days.items())), "schema_fingerprints": fps,
                          "grid_id": cfg.grid_id,
                          "source": {"bucket": f"s3://{self.settings.bucket}",
                                     "endpoint": self.settings.endpoint,
                                     "files": "gfs.YYYYMMDD/HH/[atmos/]gfs.tHHz."
                                              "pgrb2[b].0p25.fFFF (+ .idx)"}})
            count_key = keys[1]
            n_records = sum(int(d.get(count_key, 0)) for d in days.values())
            times = sorted(days)
            m = manifest.Manifest.new(cfg.store_id, "packed", "gridded", cfg,
                                      n_records=n_records, extra=extra,
                                      time_min=times[0] if times else None,
                                      time_max=times[-1] if times else None)
            if old is not None:
                m.created_at = old.created_at
            manifest.write(store, m)


def run(state_cfg: ProductConfig, prate_cfg: ProductConfig, root: Path, start: datetime,
        end: datetime, *, download_workers: int | None = None, redo: bool = False,
        time_budget_s: float | None = None, max_days: int | None = None,
        stop: threading.Event | None = None, client=None,
        requests_per_second: float | None = None,
        log: Callable[[str], None] = print) -> int:
    """Stream every day in ``[start, end)`` into both stores. Exit codes as stream_kdp."""
    t_start = time.monotonic()
    stop = stop or threading.Event()
    stream = GfsStream(state_cfg, prate_cfg, root, client=client,
                       download_workers=download_workers, redo=redo,
                       requests_per_second=requests_per_second)
    first = max(start.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0),
                stream.settings.origin)
    days = []
    d = first
    while d < end:
        days.append(d)
        d += timedelta(days=1)

    def todo(day: datetime) -> bool:
        if redo:
            return True
        status = stream.read_status(day)
        return not status or status.get("state") not in ("done", "empty")

    pending = [d for d in days if todo(d)]
    if max_days is not None:
        pending = pending[:max_days]
    if pending:
        stream.axis_end = pending[-1] + timedelta(days=1)
    log(f"stream {state_cfg.store_id} + {prate_cfg.store_id}: {len(days)} day(s) in "
        f"{first:%Y-%m-%d}..{end:%Y-%m-%d}, {len(pending)} to do; "
        f"{stream.download_workers} download workers")
    code = EXIT_DONE
    schema_days = 0
    lock = gridded.store_path(root, state_cfg.store_id) / LOCK_NAME
    try:
        with _locked(lock), cf.ThreadPoolExecutor(1, thread_name_prefix="prefetch") as pre:
            future: cf.Future | None = None
            for i, day in enumerate(pending):
                if stop.is_set() or (time_budget_s and time.monotonic() - t_start > time_budget_s):
                    code = EXIT_MORE
                    break
                try:
                    fetched = future.result() if future is not None else stream.fetch_day(day)
                except Exception as exc:  # noqa: BLE001 - stop; a rerun resumes
                    log(f"error: {day:%Y-%m-%d}: fetch failed: {type(exc).__name__}: {exc}")
                    return EXIT_ERROR
                future = None
                out_of_time = bool(time_budget_s) and time.monotonic() - t_start > time_budget_s
                if i + 1 < len(pending) and not stop.is_set() and not out_of_time:
                    future = pre.submit(stream.fetch_day, pending[i + 1])
                try:
                    status = stream.pack_day(fetched)
                except Exception as exc:
                    logger.exception("packing %s", day)
                    log(f"error: packing {day:%Y-%m-%d} failed: {type(exc).__name__}: {exc}")
                    if future is not None:
                        with contextlib.suppress(Exception):
                            future.result()
                    return EXIT_ERROR
                t, x = status["timings"], status["transfer"]
                log(f"{status['date']} {status['state']:8s} state {status['state_frames_packed']:2d}"
                    f"/24 (fallback {status['state_frames_fallback']}, missing upstream "
                    f"{len(status['state_frames_missing_upstream'])}) prate "
                    f"{status['prate_runs_packed']}/4 | {x['mb']} MB in {x['requests']} req, "
                    f"{x['messages']} msg, fetch {x['fetch_wall_s']} s ({x['mb_per_s']} MB/s) "
                    f"| pack {t['pack_s']} s write+verify {t['write_verify_s']} s | shards "
                    f"{status['state_shard_bytes'] / 1e6:.2f} + "
                    f"{status['prate_shard_bytes'] / 1e6:.2f} MB | "
                    f"schema_fail {len(status['schema_failed'])} "
                    f"dl_fail {len(status['failed_downloads'])}")
                if status["schema_failed"]:
                    schema_days += 1
                    log(f"SCHEMA: {status['date']}: {len(status['schema_failed'])} message(s) "
                        f"failed their check and were quarantined")
                    if schema_days >= MAX_SCHEMA_DAYS:
                        log(f"error: schema failures on {schema_days} consecutive days; "
                            f"stopping. See {stream.status_path(day)}")
                        if future is not None:
                            with contextlib.suppress(Exception):
                                future.result()
                        return EXIT_SCHEMA
                else:
                    schema_days = 0
            if future is not None:
                with contextlib.suppress(Exception):
                    future.result()
                code = EXIT_MORE
    finally:
        stream.close()
    remaining = [d for d in days if todo(d)]
    if code == EXIT_DONE and remaining and max_days is None:
        never = [d for d in remaining if stream.read_status(d) is None]
        code = EXIT_MORE if never else EXIT_DONE
    log(f"stream {state_cfg.store_id}: exit {code}; {len(days) - len(remaining)} of {len(days)} "
        f"day(s) done, {len(remaining)} not done; {stream.client.requests_made} request(s)")
    return code


__all__ = ["GfsStream", "RunData", "Settings", "run"]
