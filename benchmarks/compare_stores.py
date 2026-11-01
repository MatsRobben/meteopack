"""Are two gridded stores the same output? Exit 0 if so, 1 with the differences.

The gate for every performance change: a candidate store packed by new code
must decode to exactly what the reference holds - same arrays, same dtypes,
same chunking and codecs, same attributes, same values in every written frame.
Only `created_at` may differ.

    python benchmarks/compare_stores.py REF.zarr CANDIDATE.zarr

Values are compared shard by shard, and only shards that hold a written frame
in either store: a sparse sample spanning 18 years is ~6,700 shards of which a
few dozen exist, and reading the rest would only read the fill value back.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr

#: Root attributes allowed to differ between two packs of the same input.
VOLATILE = frozenset({"created_at"})


def _layout(arr) -> dict:
    return {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "chunks": tuple(arr.chunks),
        "shards": tuple(arr.shards) if getattr(arr, "shards", None) else None,
        "fill_value": arr.fill_value.item() if hasattr(arr.fill_value, "item")
        else arr.fill_value,
        "codecs": json.dumps([c.to_dict() for c in arr.metadata.codecs], sort_keys=True,
                             default=str),
        "attrs": json.dumps(dict(arr.attrs), sort_keys=True, default=str),
    }


def compare(ref_path: Path, cand_path: Path) -> list[str]:
    ref = zarr.open_group(str(ref_path), mode="r")
    cand = zarr.open_group(str(cand_path), mode="r")
    problems: list[str] = []

    ra = {k: v for k, v in ref.attrs.items() if k not in VOLATILE}
    ca = {k: v for k, v in cand.attrs.items() if k not in VOLATILE}
    for key in sorted(set(ra) | set(ca)):
        if json.dumps(ra.get(key), sort_keys=True, default=str) != \
                json.dumps(ca.get(key), sort_keys=True, default=str):
            problems.append(f"root attr {key!r}: {ra.get(key)!r} != {ca.get(key)!r}")

    rnames = {n for n, _ in ref.arrays()}
    cnames = {n for n, _ in cand.arrays()}
    for name in sorted(rnames ^ cnames):
        problems.append(f"array {name!r} only in {'ref' if name in rnames else 'candidate'}")

    # Which time steps either store wrote. Frames missing in both read back as
    # the fill value from both, which `_layout` has already compared.
    written = None
    if "is_missing_frame" in rnames & cnames:
        written = ~np.asarray(ref["is_missing_frame"][:]) | \
            ~np.asarray(cand["is_missing_frame"][:])

    for name in sorted(rnames & cnames):
        r, c = ref[name], cand[name]
        lr, lc = _layout(r), _layout(c)
        for key in lr:
            if lr[key] != lc[key]:
                problems.append(f"{name}: {key} differs: {lr[key]} != {lc[key]}")
        if lr["shape"] != lc["shape"]:
            continue
        if r.ndim < 3 or written is None:
            if not np.array_equal(np.asarray(r[...]), np.asarray(c[...]), equal_nan=True):
                problems.append(f"{name}: values differ")
            continue
        step = (r.shards or r.chunks)[0]
        n_frames = n_diff_frames = 0
        first_bad = None
        for s0 in range(0, r.shape[0], step):
            s1 = min(s0 + step, r.shape[0])
            if not written[s0:s1].any():
                continue
            a, b = np.asarray(r[s0:s1]), np.asarray(c[s0:s1])
            n_frames += int(written[s0:s1].sum())
            if not np.array_equal(a, b):
                bad = np.flatnonzero((a != b).reshape(s1 - s0, -1).any(axis=1))
                n_diff_frames += bad.size
                if first_bad is None:
                    first_bad = s0 + int(bad[0])
        if n_diff_frames:
            problems.append(f"{name}: {n_diff_frames} of {n_frames} written frames differ "
                            f"(first at index {first_bad})")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    args = ap.parse_args()
    problems = compare(args.reference, args.candidate)
    if problems:
        print(f"DIFFERENT: {len(problems)} problem(s)")
        for p in problems:
            print(f"  {p}")
        return 1
    print("IDENTICAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
