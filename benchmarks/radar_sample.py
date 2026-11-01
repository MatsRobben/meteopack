"""Build a fixed sample of the radar archives as a meteopack data root.

    python benchmarks/radar_sample.py --root /path/to/sample --per-product 24

Each product gets `--per-product` tars spaced evenly over its archive, plus the
first tar (the irregular archive start), plus one example tar for every image
layout a previous full pack recorded in its scan_report.json. That last part
is what makes the sample worth comparing on: every calibration generation is
in it, including the ones that occur on a handful of days - an optimisation
that only breaks a 14-frame layout would otherwise pass.

The tars are symlinked, not copied, under <root>/raw/knmi_radar/<product>/,
and the selection is written to <root>/sample.json so a run can be reproduced.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

KNMI = Path(os.environ.get(
    "KNMI_RAW", "/vol/knmimo-nobackup/users/mrobben/data_raw/data/raw/knmi"))

ARCHIVES = {
    "rtcor_5min": ("KNMI_RADAR_ARCHIVE", KNMI / "nl_rdr_data_rtcor_5m_tar",
                   "RAD_NL25_RAC_RT_"),
    "echotop_5min": ("KNMI_ECHOTOP_ARCHIVE", KNMI / "radar_tar_echotopheight_5min",
                     "RAD_NL25_ETH_NA_"),
    "refl_5min": ("KNMI_REFL_ARCHIVE", KNMI / "radar_tar_refl_composites",
                  "RAD_NL25_PCP_NA_"),
}

#: Where earlier full packs left their layout census.
REPORTS = Path(os.environ.get(
    "METEOPACK_REPORT_ROOT", "/vol/knmimo-nobackup/users/mrobben/data_v2/packed/knmi_radar"))


def archive_of(product: str) -> Path:
    env, default, _ = ARCHIVES[product]
    return Path(os.environ.get(env, default))


def _start(tar: Path) -> str | None:
    """A tar's first instant as YYYYmmddHHMM, comparable as a string."""
    m = re.search(r"__(\d{8})T(\d{4})\d\d_", tar.name)
    return m.group(1) + m.group(2) if m else None


def layout_example_tars(product: str, tars: list[Path]) -> list[Path]:
    """The tar holding each layout's example frame, from a previous scan report.

    A frame belongs to the tar with the latest start at or before it. That is
    right for both naming conventions - rtcor's 08:05->08:00 and the exclusive
    midnight->midnight of the other two - where matching on the end token is
    off by one tar exactly at midnight.
    """
    report = REPORTS / product / "v1" / "scan_report.json"
    if not report.exists():
        return []
    prefix = ARCHIVES[product][2]
    starts = sorted((st, t) for t in tars if (st := _start(t)))
    out = []
    for entry in json.loads(report.read_text()).get("layouts", []):
        name = str(entry.get("example", ""))
        if not name.startswith(prefix):
            continue
        stamp = name[len(prefix):len(prefix) + 12]        # YYYYmmddHHMM
        holders = [t for st, t in starts if st <= stamp]
        if holders:
            out.append(holders[-1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--per-product", type=int, default=24)
    ap.add_argument("--products", default=",".join(ARCHIVES))
    ap.add_argument("--offset", type=int, default=0,
                    help="shift the evenly spaced picks; offsets 0,1,2 give disjoint samples, "
                         "which is what a scaling run needs - the NFS server caches what an "
                         "earlier run read")
    ap.add_argument("--no-examples", action="store_true",
                    help="evenly spaced tars only: no first tar, no layout examples (they "
                         "would be shared between otherwise disjoint samples)")
    args = ap.parse_args()

    chosen: dict[str, list[str]] = {}
    for product in args.products.split(","):
        tars = sorted(archive_of(product).glob("*.tar"))
        if not tars:
            raise SystemExit(f"no tars for {product} under {archive_of(product)}")
        stride = max(len(tars) // args.per_product, 1)
        picks = set(tars[args.offset::stride][: args.per_product])
        examples = [] if args.no_examples else layout_example_tars(product, tars)
        if not args.no_examples:
            picks.add(tars[0])
        picks.update(examples)
        dest = args.root / "raw" / "knmi_radar" / product
        dest.mkdir(parents=True, exist_ok=True)
        for old in dest.iterdir():
            old.unlink()
        for t in sorted(picks):
            (dest / t.name).symlink_to(t)
        size = sum(t.stat().st_size for t in picks) / 2**20
        chosen[product] = sorted(t.name for t in picks)
        print(f"{product:13s} {len(picks):3d} tars ({len(examples)} layout examples), "
              f"{size:,.0f} MiB")
    (args.root / "sample.json").write_text(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    main()
