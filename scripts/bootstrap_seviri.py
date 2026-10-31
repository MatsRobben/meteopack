"""One-off bootstrap: pull MSG15 RSS scenes into the raw layer.

The EUMETSAT source's own `fetch` reads an archive that is already on disk;
this fills that archive the first time. Kept separate from the source because
it needs credentials and is run by hand, not by the pipeline.

Resumable: a scene already present and non-empty is skipped, and every write
goes to a private temp file before an atomic rename, so an interrupted run
never leaves a half-scene that looks complete.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import os
import shutil
import sys
import threading
import uuid
from pathlib import Path

import eumdac

COLLECTION = "EO:EUM:DAT:MSG:MSG15-RSS"
_print_lock = threading.Lock()


def credentials() -> tuple[str, str]:
    path = Path.home() / ".eumdac" / "credentials"
    if not path.exists():
        raise SystemExit(f"no EUMETSAT credentials at {path}")
    key, secret = path.read_text().strip().split(",")[:2]
    return key.strip(), secret.strip()


def scene_time(product) -> dt.datetime:
    stamp = str(product).split("-NA-")[1][:14]
    return dt.datetime.strptime(stamp, "%Y%m%d%H%M%S")


def destination(root: Path, product) -> Path:
    when = scene_time(product)
    return root / f"{when:%Y/%m/%d}" / f"{product}.zip"


def fetch_one(product, root: Path) -> tuple[str, int]:
    dest = destination(root, product)
    if dest.exists() and dest.stat().st_size > 0:
        return "skip", dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex[:8]}.part")
    try:
        with product.open() as src, tmp.open("wb") as out:
            shutil.copyfileobj(src, out, length=1 << 20)
        size = tmp.stat().st_size
        os.replace(tmp, dest)
        return "ok", size
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root)
    store = eumdac.DataStore(eumdac.AccessToken(credentials()))
    products = list(store.get_collection(COLLECTION).search(
        dtstart=dt.datetime.fromisoformat(args.start),
        dtend=dt.datetime.fromisoformat(args.end)))
    print(f"{len(products)} scenes in {args.start}..{args.end}; {args.workers} workers", flush=True)

    done = skipped = failed = 0
    total = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, p, root): p for p in products}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            try:
                status, size = fut.result()
            except Exception as exc:                      # noqa: BLE001 - reported, not hidden
                failed += 1
                with _print_lock:
                    print(f"FAILED {futures[fut]}: {exc}", flush=True)
                continue
            total += size
            done += status == "ok"
            skipped += status == "skip"
            if i % 50 == 0:
                print(f"  {i}/{len(products)}  {total/2**30:.2f} GiB", flush=True)

    print(f"done={done} skipped={skipped} failed={failed} total={total/2**30:.2f} GiB")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
