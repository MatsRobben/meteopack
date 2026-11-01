"""Every frame of a KNMI radar archive whose calibration is not one of the real ones.

    python benchmarks/odd_calibration_census.py ARCHIVE "FORMULA" ["FORMULA" ...]

For each unexpected calibration: how many frames, on which days, and how many
of them hold ANY echo pixel (a code other than 0 and 255). A placeholder
calibration over an empty image is a failed product and belongs in
extra.corrupt_layouts; one over real echo would need a different answer.
"""
import io
import os
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

ARCHIVE = Path(sys.argv[1])
REAL = set(sys.argv[2:])


def text(v) -> str:
    a = np.asarray(v).reshape(-1)
    x = a[0] if a.size else v
    return x.decode() if isinstance(x, bytes) else str(x)


def one(tar: Path) -> list[tuple[str, str, int]]:
    out = []
    with tarfile.open(tar) as t:
        try:
            members = [m for m in t if m.isfile()]
        except tarfile.ReadError:
            return [(tar.name, "<truncated tar>", -1)]
        for m in members:
            try:
                with h5py.File(io.BytesIO(t.extractfile(m).read())) as h:
                    g = h["image1"]
                    cal = text(g["calibration"].attrs["calibration_formulas"])
                    if cal in REAL:
                        continue
                    d = np.asarray(g["image_data"][:])
            except Exception:  # noqa: BLE001 - counted, not fatal
                out.append((m.name[-15:-3], "<unreadable>", -1))
                continue
            out.append((m.name[-15:-3], cal, int(((d != 0) & (d != 255)).sum())))
    return out


if __name__ == "__main__":
    tars = sorted(ARCHIVE.glob("*.tar"))
    with ProcessPoolExecutor(len(os.sched_getaffinity(0))) as ex:
        rows = [r for rs in ex.map(one, tars, chunksize=8) for r in rs]
    by: dict[str, list[tuple[str, int]]] = {}
    for stamp, cal, echo in rows:
        by.setdefault(cal, []).append((stamp, echo))
    print(f"{ARCHIVE.name}: {len(tars)} tars; {len(rows)} frames outside the real calibrations")
    for cal, frames in sorted(by.items(), key=lambda kv: -len(kv[1])):
        days = sorted({s[:8] for s, _ in frames})
        with_echo = [(s, e) for s, e in frames if e > 0]
        print(f"  {cal!r}: {len(frames)} frames on {len(days)} day(s), "
              f"{days[0]}..{days[-1]}; frames with ANY echo pixel: {len(with_echo)}"
              + (f" e.g. {with_echo[:5]}" if with_echo else ""))
