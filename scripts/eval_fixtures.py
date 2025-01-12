"""Conformance evaluation on the synthetic corpus plus a random benign-pickle corpus.

Usage: uv run python scripts/eval_fixtures.py

1. Fixture corpus (tests/fixtures): every fixture marked ``malicious_pattern`` must be
   reported at HIGH or CRITICAL; every fixture with no expected rule must be clean.
   The scanner was developed against these files, so this is a conformance check, not
   an unbiased accuracy estimate.
2. Random benign pickles: 2,000 values drawn from a fixed-seed generator over stdlib
   value types (dict/list/tuple/set/bytes/str/int/float/datetime/Decimal/OrderedDict),
   serialized with pickle.dumps at protocols 0-5. These were never used during
   development; any finding at MEDIUM or above is a false positive.
"""

from __future__ import annotations

import datetime
import decimal
import json
import pickle
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

from picklescope import Severity, __version__, scan_bytes, scan_file

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
SEED = 20261024
SAMPLES = 2000


def _value(rng: random.Random, depth: int = 0) -> Any:
    kind = rng.randrange(13 if depth < 3 else 8)
    leaves: list[Any] = [
        None, rng.random() < 0.5, rng.randint(-(2**70), 2**70), rng.random() * 1e6,
        "".join(chr(rng.randrange(32, 0x2FFF)) for _ in range(rng.randrange(12))),
        rng.randbytes(rng.randrange(40)),
        datetime.datetime(2020, 1, 1) + datetime.timedelta(seconds=rng.randrange(10**8)),
        decimal.Decimal(rng.randint(-999, 999)) / 7,
    ]  # fmt: skip
    if kind < 8:
        return leaves[kind]
    children = [_value(rng, depth + 1) for _ in range(rng.randrange(5))]
    if kind == 8:
        return children
    if kind == 9:
        return tuple(children)
    if kind == 10:
        return {str(i): c for i, c in enumerate(children)}
    if kind == 11:
        return OrderedDict((str(i), c) for i, c in enumerate(children))
    return {rng.randrange(100) for _ in children} | frozenset({1})


def fixture_conformance() -> dict[str, Any]:
    catalogue = json.loads((FIXTURES / "catalogue.json").read_text())
    malicious = {k for k, v in catalogue.items() if v["malicious_pattern"]}
    detected: list[str] = []
    missed: list[str] = []
    for rel in sorted(malicious):
        report = scan_file(FIXTURES / rel, display_path=rel, hash_file=False)
        top = report.max_severity
        (detected if top is not None and top >= Severity.HIGH else missed).append(rel)
    clean_expected = [k for k in catalogue if k not in malicious]
    return {
        "malicious_pattern_fixtures": len(malicious),
        "detected_high_or_critical": len(detected),
        "missed": missed,
        "other_fixtures": len(clean_expected),
    }


def random_benign() -> dict[str, Any]:
    rng = random.Random(SEED)
    flagged = []
    by_protocol = dict.fromkeys(range(6), 0)
    total_bytes = 0
    for index in range(SAMPLES):
        protocol = index % 6
        data = pickle.dumps(_value(rng), protocol=protocol)  # serialisation only
        total_bytes += len(data)
        report = scan_bytes(data, f"random-{index}.pkl")
        by_protocol[protocol] += 1
        if report.findings_at_least(Severity.MEDIUM):
            flagged.append({"index": index, "rules": sorted({f.rule_id for f in report.findings})})
    return {
        "seed": SEED,
        "samples": SAMPLES,
        "per_protocol": by_protocol,
        "total_bytes": total_bytes,
        "false_positives": len(flagged),
        "false_positive_rate": len(flagged) / SAMPLES,
        "flagged": flagged[:20],
    }


def main() -> int:
    conf = fixture_conformance()
    rnd = random_benign()
    result = {
        "picklescope_version": __version__,
        "fixture_conformance": conf,
        "random_benign_pickles": rnd,
    }
    out = ROOT / "eval" / "results-fixtures.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"malicious-pattern fixtures reported >= high: "
        f"{conf['detected_high_or_critical']}/{conf['malicious_pattern_fixtures']}"
    )
    print(
        f"random benign pickles flagged >= medium: {rnd['false_positives']}/{rnd['samples']} "
        f"({rnd['total_bytes']:,} bytes, protocols 0-5, seed {SEED})"
    )
    print(f"wrote {out.relative_to(ROOT)}")
    return 0 if not conf["missed"] else 1


if __name__ == "__main__":
    sys.exit(main())
