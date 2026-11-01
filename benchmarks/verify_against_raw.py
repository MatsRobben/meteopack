"""Verify a packed radar store against the raw HDF5 files, independently.

Comparing against radar_v3 shows the two pipelines agree; it cannot show
either is right. This reads the source files and decodes them *here*, with the
calibration arithmetic written out longhand rather than imported, so a bug in
`meteopack.core.encoding` cannot hide by being used on both sides of the
comparison.

Checked per sampled frame:

* every pixel's physical value against the file's own `calibration_formulas`,
  to within half a quantisation step of the store's own calibration
* validity: a pixel the file calls real must not read back as missing, and a
  pixel the file calls missing must not read back as a number
* the frame sits at the timestamp its member name claims
* the grid matches the file's `geo_number_rows`/`geo_number_columns`

Usage: python benchmarks/verify_against_raw.py --frames 40
"""
from __future__ import annotations

import argparse
import io
import os
import random
import re
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ARCHIVE = Path(os.environ.get(
    "KNMI_RADAR_ARCHIVE",
    "/vol/knmimo-nobackup/users/mrobben/data_raw/data/raw/knmi/nl_rdr_data_rtcor_5m_tar",
))
STORE = os.environ.get(
    "STORE",
    "/vol/knmimo-nobackup/users/mrobben/data_full/packed/knmi_radar/rtcor_5min/v1/dataset.zarr",
)

# Any of the three radar products: rtcor members are RAD_NL25_RAC_RT_,
# echotop RAD_NL25_ETH_NA_, the reflectivity composites RAD_NL25_PCP_NA_.
MEMBER_RE = re.compile(r"RAD_NL25_\w+?_(\d{12})\.h5$")
#: store variable -> the image_geo_parameter values that carry it, across all
#: three radar products.
WANTED = {
    "accumulation_mm": ("PRECIP_[MM]", "ACCUMULATION_[MM]"),
    "quality": ("QUALITY_[-]",),
    "adjustment_factor_db": ("ADJUSTMENT_FACTOR_[DB]",),
    "echotop_height_km": ("HEIGHT_[KM]",),
    "reflectivity_dbz": ("REFLECTIVITY_[DBZ]",),
}


def text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        return text(value.reshape(-1)[0]) if value.size else ""
    return str(value)


def parse_formula(raw: str) -> tuple[float, float]:
    """`GEO = 0.01 * PV + 0.0` -> (0.01, 0.0). Written out here on purpose."""
    m = re.search(r"GEO\s*=\s*([-+0-9.eE]+)\s*\*\s*PV\s*\+\s*([-+]*[0-9.eE]+)", raw)
    if not m:
        raise ValueError(f"unreadable calibration {raw!r}")
    offset_text = m.group(2).replace("+-", "-").lstrip("+")
    return float(m.group(1)), float(offset_text)


def codes_of(calibration, key) -> set[int]:
    value = calibration.attrs.get(key)
    if value is None:
        return set()
    return {int(v) for v in np.asarray(value).reshape(-1)}


def no_echo_values(product: str) -> dict[str, float]:
    """The value each variable's "no echo" takes, read from the product config.

    The config is the specification of the store, so this reads it rather than
    asking the pipeline's code what it did - and not from the store's own
    attrs either, which the pipeline wrote.
    """
    import yaml

    path = Path(__file__).resolve().parents[1] / "configs" / "products" / "knmi_radar" / \
        f"{product}.yaml"
    cfg = yaml.safe_load(path.read_text())
    return {v["name"]: float(v["no_echo"]) for v in cfg["variables"]
            if v.get("no_echo") is not None}


def decode_raw(dataset, calibration, no_echo: float | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Physical values and a validity mask, straight from the file."""
    counts = np.asarray(dataset[:])
    scale, offset = parse_formula(text(calibration.attrs["calibration_formulas"]))
    out = counts.astype("float64") * scale + offset

    missing = codes_of(calibration, "calibration_missing_data")
    out_of_image = codes_of(calibration, "calibration_out_of_image")
    info = np.iinfo(counts.dtype) if counts.dtype.kind in "ui" else None

    def representable(values: set[int]) -> set[int]:
        # A sentinel the dtype cannot hold is clamped to the dtype maximum,
        # which is what the product means and what later generations write.
        kept = set()
        for v in values:
            if info is None or info.min <= v <= info.max:
                kept.add(v)
            elif v > info.max:
                kept.add(int(info.max))
        return kept

    keep_missing = representable(missing)
    keep_out = representable(out_of_image)
    # A declared missing code of 0 means "no echo" for the intensity variables,
    # and takes the value the product config states for no echo (0 mm, 0 km,
    # -32 dBZ). This is the one place this script states an interpretation of
    # the data rather than just reading it, so here is the evidence, from a
    # census of 40 tars per archive spread over 2008-2026
    # (benchmarks/README_findings.md):
    #
    #   PRECIP_[MM]      miss=0    occurs on 75-80% of pixels
    #   REFLECTIVITY     miss=0    occurs on 70%
    #   HEIGHT_[KM]      miss=0    occurs on 45%
    #   QUALITY_[-]      miss=0    occurs on  0.00%
    #   ADJUSTMENT_[DB]  miss=0    occurs on  0.00%
    #
    # A code of 0 is only ever populated for the intensity variables, where it
    # is the bottom of the scale - the radar looked and saw no echo - and never
    # for the variables where 0 would genuinely mean "no measurement". The
    # non-zero sentinels (254, 65534) are honoured normally.
    #
    # It is NOT decoded with the file's calibration. That gives the stated
    # value only where the offset is 0; KNMI shifted the offset by one count
    # between generations, and this script used to call -0.102 km echo tops
    # correct because the pipeline made the same mistake.
    if no_echo is not None and 0 in keep_missing:
        keep_missing.discard(0)
        out[counts == 0] = no_echo
    reserved = sorted(keep_missing | keep_out)
    invalid = np.isin(counts, reserved) if reserved else np.zeros(counts.shape, bool)
    return out, ~invalid


def main() -> int:
    import h5py
    import zarr

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    group = zarr.open(STORE, mode="r")
    no_echo = no_echo_values(str(group.attrs["product"]))
    stamps = np.asarray(group["time"][:]).astype("datetime64[ns]")
    index = {t: i for i, t in enumerate(stamps)}
    written = ~np.asarray(group["is_missing_frame"][:])
    present = {k for k, _ in group.arrays()}

    # Only tars that overlap what the store actually holds. Sampling the whole
    # archive against a two-day store spends every sample on a miss and then
    # reports "nothing compared", which looks like a pass.
    first, last = stamps[0], stamps[-1]
    tars = []
    for path in sorted(ARCHIVE.glob("*.tar")):
        m = re.search(r"__(\d{8})T", path.name)
        if not m:
            continue
        day = np.datetime64(f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}", "ns")
        if first - np.timedelta64(2, "D") <= day <= last + np.timedelta64(1, "D"):
            tars.append(path)
    if not tars:
        print(f"no tars in {ARCHIVE} overlap the store's {first}..{last}")
        return 1
    rng = random.Random(args.seed)
    picks = rng.sample(tars, min(args.frames, len(tars)))

    checked = failures = skipped = 0
    for path in picks:
        with tarfile.open(path, "r") as tar:
            members = [m for m in tar if MEMBER_RE.search(m.name)]
            if not members:
                continue
            member = rng.choice(members)
            blob = tar.extractfile(member).read()

        when = datetime.strptime(
            MEMBER_RE.search(member.name).group(1), "%Y%m%d%H%M"
        ).replace(tzinfo=UTC)
        key = np.datetime64(when.replace(tzinfo=None), "ns")
        if key not in index or not written[index[key]]:
            skipped += 1
            continue
        row = index[key]

        with h5py.File(io.BytesIO(blob), "r") as frame:
            geo = frame["geographic"].attrs
            ny = int(np.asarray(geo["geo_number_rows"]).reshape(-1)[0])
            nx = int(np.asarray(geo["geo_number_columns"]).reshape(-1)[0])
            for name in sorted(k for k in frame if k.startswith("image")):
                node = frame[name]
                if "image_data" not in node or "calibration" not in node:
                    continue
                parameter = text(node.attrs["image_geo_parameter"])
                target = next((v for v, ps in WANTED.items() if parameter in ps), None)
                if target is None or target not in present:
                    continue

                truth, valid = decode_raw(node["image_data"], node["calibration"],
                                          no_echo.get(target))
                array = group[target]
                if tuple(array.shape[1:]) != (ny, nx):
                    print(f"FAIL grid {tuple(array.shape[1:])} != file {(ny, nx)}")
                    failures += 1
                    continue

                codes = np.asarray(array[row])
                attrs = array.attrs
                scale = float(attrs["scale_factor"])
                offset = float(attrs["add_offset"])
                reserved = {int(attrs["calibration_missing_data"]),
                            int(attrs["calibration_out_of_image"])}
                stored_valid = ~np.isin(codes, sorted(reserved))
                stored = codes.astype("float64") * scale + offset

                checked += 1
                if not np.array_equal(stored_valid, valid):
                    n = int((stored_valid != valid).sum())
                    print(f"FAIL {member.name} {target}: validity differs on {n} px")
                    failures += 1
                    continue
                if not valid.any():
                    continue
                delta = np.abs(stored[valid] - truth[valid])
                tolerance = scale / 2 + 1e-9
                worst = float(delta.max())
                if worst > tolerance:
                    bad = int((delta > tolerance).sum())
                    print(f"FAIL {member.name} {target}: {bad} px off by up to "
                          f"{worst:g} (tolerance {tolerance:g} = half a step)")
                    failures += 1

    print(f"\nframes compared: {checked}, skipped (not in store): {skipped}, "
          f"failures: {failures}")
    print("RESULT:", "store matches the raw files" if checked and not failures
          else "DISCREPANCY" if failures else "nothing compared")
    return 0 if checked and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
