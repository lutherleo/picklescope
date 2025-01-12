"""Offline end-to-end demo over the bundled fixtures (no network, nothing loaded).

Usage: uv run python scripts/demo.py   (or: make demo)

Runs the real CLI entry point for each step and prints its output, then writes a SARIF
report to out/demo.sarif.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path

from picklescope.cli import main as cli

ROOT = Path(__file__).resolve().parent.parent
FX = "tests/fixtures"


def run(*argv: str) -> int:
    print(f"$ picklescope {' '.join(argv)}", flush=True)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = cli(list(argv))
    if buffer.getvalue().strip():
        print(buffer.getvalue().rstrip())
    print(f"[exit {code}]\n", flush=True)
    return code


def main() -> int:
    os.chdir(ROOT)
    print("== 1. A benign PyTorch archive and the same archive with a hostile data.pkl ==\n")
    run(
        "scan",
        "--no-hash",
        "--show-imports",
        f"{FX}/torch/benign_state_dict.pt",
        f"{FX}/torch/print_in_data_pkl.pt",
    )

    print("== 2. Pickle evasion patterns (all built from harmless callables) ==\n")
    run(
        "scan",
        "--no-hash",
        f"{FX}/pickle/getattr_gadget.pkl",
        f"{FX}/pickle/stack_global_memo.pkl",
        f"{FX}/pickle/nested_pickle.pkl",
        f"{FX}/pickle/two_pickles.pkl",
        f"{FX}/pickle/dunder_traversal.pkl",
    )

    print("== 3. Why: annotated disassembly of the getattr gadget ==\n")
    run("disasm", f"{FX}/pickle/getattr_gadget.pkl")

    print("== 4. Other formats: Keras, GGUF, ONNX, safetensors ==\n")
    run(
        "scan",
        "--no-hash",
        f"{FX}/keras/lambda.keras",
        f"{FX}/keras/function_denied.keras",
        f"{FX}/gguf/chat_template_dunder.gguf",
        f"{FX}/onnx/traversal.onnx",
        f"{FX}/safetensors/zip_polyglot.safetensors",
        f"{FX}/safetensors/pickle_in_disguise.safetensors",
    )

    print("== 5. Policy decisions and a custom policy file ==\n")
    run(
        "policy",
        "check",
        "torch._utils:_rebuild_tensor_v2",
        "builtins:getattr",
        "sklearn.svm:SVC",
        "colorsys:rgb_to_hsv",
    )
    run("scan", "--no-hash", "--policy", f"{FX}/policy-test.toml", f"{FX}/pickle/colorsys_call.pkl")

    print("== 6. Whole corpus to SARIF 2.1.0 ==\n")
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    code = run("scan", "--no-hash", "-f", "sarif", "-o", "out/demo.sarif", FX)
    log = json.loads((out / "demo.sarif").read_text())
    results = log["runs"][0]["results"]
    levels: dict[str, int] = {}
    for result in results:
        levels[result["level"]] = levels.get(result["level"], 0) + 1
    print(
        f"out/demo.sarif: {len(log['runs'][0]['artifacts'])} artifacts, {len(results)} "
        f"results, by level {dict(sorted(levels.items()))}"
    )
    return 0 if code in (0, 1) else code


if __name__ == "__main__":
    sys.exit(main())
