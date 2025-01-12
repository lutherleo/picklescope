"""Download the pinned public models listed in eval/manifest.json (checksum-verified).

Usage: uv run python scripts/fetch_models.py [--dest .cache/models]

Each file is fetched from a fixed Hugging Face commit, verified against its pinned
SHA-256 and size, and stored under ``<dest>/<id>``. Already-verified files are skipped.
Nothing downloaded here is ever loaded or deserialized; it is only scanned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(entry: dict[str, str | int], dest: Path) -> Path:
    target = dest / str(entry["id"])
    if target.exists() and _sha256(target) == entry["sha256"]:
        print(f"ok (cached)  {entry['id']}", flush=True)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = URL.format(**{k: entry[k] for k in ("repo", "revision", "path")})
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "picklescope-eval/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    digest = _sha256(partial)
    if digest != entry["sha256"] or partial.stat().st_size != entry["size"]:
        partial.unlink()
        raise SystemExit(f"checksum mismatch for {entry['id']}: got {digest}")
    partial.replace(target)
    print(f"downloaded   {entry['id']} ({entry['size']} bytes)", flush=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=ROOT / ".cache" / "models")
    parser.add_argument("--manifest", type=Path, default=ROOT / "eval" / "manifest.json")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    for entry in manifest["models"]:
        fetch(entry, args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
