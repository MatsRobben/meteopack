"""meteopack {scan,list,fetch,parse,pack,derive,validate,catalog} <source|dataset> --window"""
from __future__ import annotations

import argparse

STAGES = ("scan", "list", "fetch", "parse", "pack", "derive", "validate", "catalog")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="meteopack", description=__doc__)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("target", help="source name (list/fetch/parse/pack) or dataset name (derive)")
    p.add_argument("--window", help="ISO interval, e.g. 2012-01-01/2026-01-01")
    p.add_argument("--config", help="path override for the product/derive config")
    p.add_argument("--workers", type=int, default=None, help="default: all available cores")
    p.add_argument(
        "--data-root",
        help="overrides $METEOPACK_DATA_ROOT, which overrides the config default",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise SystemExit(f"stage {args.stage!r} is not implemented yet (target={args.target!r})")


if __name__ == "__main__":
    main()
