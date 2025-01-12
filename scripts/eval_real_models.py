"""False-positive evaluation on pinned real public models.

Usage: uv run python scripts/eval_real_models.py   (after scripts/fetch_models.py)

Every file in eval/manifest.json is a benign, widely downloaded model, so any finding
at MEDIUM or above counts as a false positive. Results are written to
eval/results-real-models.json and printed as a Markdown table.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from picklescope import Severity, __version__, scan_file

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "models"


def main() -> int:
    manifest = json.loads((ROOT / "eval" / "manifest.json").read_text())["models"]
    rows = []
    for entry in manifest:
        path = CACHE / entry["id"]
        if not path.exists():
            print(f"missing {path}; run scripts/fetch_models.py first", file=sys.stderr)
            return 2
        start = time.perf_counter()
        report = scan_file(path, display_path=entry["id"])
        elapsed = time.perf_counter() - start
        flagged = report.findings_at_least(Severity.MEDIUM)
        rows.append(
            {
                "id": entry["id"],
                "repo": entry["repo"],
                "revision": entry["revision"],
                "expected_format": entry["format"],
                "detected_format": report.format,
                "size": report.size,
                "sha256_verified": report.sha256 == entry["sha256"],
                "findings": [f.to_dict() for f in report.findings],
                "false_positive": bool(flagged),
                "imports": sorted(f"{i['module']}.{i['name']}" for i in report.imports),
                "seconds": round(elapsed, 4),
            }
        )
    fp = sum(r["false_positive"] for r in rows)
    any_finding = sum(bool(r["findings"]) for r in rows)
    result = {
        "picklescope_version": __version__,
        "files": len(rows),
        "files_with_medium_or_higher": fp,
        "false_positive_rate": fp / len(rows),
        "files_with_any_finding": any_finding,
        "rows": rows,
    }
    out = ROOT / "eval" / "results-real-models.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("| model file | format | size | findings | imports seen |")
    print("|---|---|---|---|---|")
    for r in rows:
        imports = ", ".join(r["imports"]) or "-"
        print(
            f"| {r['id']} | {r['detected_format']} | {r['size']:,} | {len(r['findings'])} "
            f"| {imports} |"
        )
    print(
        f"\nfalse-positive rate (files with >= medium findings): {fp}/{len(rows)} "
        f"= {result['false_positive_rate']:.1%}"
    )
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
