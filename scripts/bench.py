"""Throughput benchmarks.

Usage: uv run python scripts/bench.py

* pickle analyser on synthetic streams (protocol 2 and 4) of ~1 MiB and ~8 MiB,
* full scans of the fixture corpus,
* full scans of the pinned real models when they have been fetched.

Each measurement is the median of 5 runs (after one warm-up) on the machine running
the script; results go to eval/results-bench.json together with platform details.
"""

from __future__ import annotations

import functools
import json
import pickle
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from picklescope import __version__, scan_file, scan_paths
from picklescope.pkl.analyzer import analyze_pickle

ROOT = Path(__file__).resolve().parent.parent
RUNS = 5


def _median_seconds(fn: Callable[[], object]) -> float:
    fn()
    samples = []
    for _ in range(RUNS):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _synthetic(n_items: int, protocol: int) -> bytes:
    rows = [{"id": i, "name": f"row-{i}", "values": [i * 0.5, -i, i % 7], "flag": i % 2 == 0}
            for i in range(n_items)]  # fmt: skip
    return pickle.dumps(rows, protocol=protocol)  # serialisation only


def main() -> int:
    results: dict[str, Any] = {
        "picklescope_version": __version__,
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
        "runs": RUNS,
        "pickle_analyser": [],
    }
    for n_items in (12_000, 100_000):
        for protocol in (2, 4):
            data = _synthetic(n_items, protocol)
            ops = analyze_pickle(data).op_count
            seconds = _median_seconds(functools.partial(analyze_pickle, data))
            row = {
                "protocol": protocol,
                "bytes": len(data),
                "opcodes": ops,
                "seconds": round(seconds, 4),
                "mib_per_s": round(len(data) / seconds / 1024**2, 2),
                "mops_per_s": round(ops / seconds / 1e6, 3),
            }
            results["pickle_analyser"].append(row)
            print(
                f"analyse p{protocol} {len(data) / 1024**2:6.2f} MiB {ops:>9,} ops: "
                f"{row['mib_per_s']:6.2f} MiB/s, {row['mops_per_s']:.3f} Mops/s"
            )

    fixtures = ROOT / "tests" / "fixtures"
    count = len(scan_paths([fixtures], hash_files=False).files)
    seconds = _median_seconds(lambda: scan_paths([fixtures], hash_files=False))
    results["fixture_corpus"] = {"files": count, "seconds": round(seconds, 4)}
    print(f"fixture corpus: {count} files in {seconds * 1000:.1f} ms")

    cache = ROOT / ".cache" / "models"
    manifest = json.loads((ROOT / "eval" / "manifest.json").read_text())["models"]
    real = []
    for entry in manifest:
        path = cache / entry["id"]
        if not path.exists():
            continue
        seconds = _median_seconds(functools.partial(scan_file, path, hash_file=False))
        real.append({"id": entry["id"], "bytes": entry["size"], "seconds": round(seconds, 4)})
        print(
            f"real model {entry['id']:<45} {entry['size'] / 1024**2:7.2f} MiB "
            f"{seconds * 1000:8.1f} ms"
        )
    results["real_models"] = real or "not fetched"
    out = ROOT / "eval" / "results-bench.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
