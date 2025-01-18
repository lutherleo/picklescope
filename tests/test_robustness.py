"""Robustness: hostile bytes never crash a scanner, and nothing is ever deserialized."""

from __future__ import annotations

import ast
import random
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from picklescope import scan_bytes

from conftest import FIXTURES

SRC = Path(__file__).resolve().parent.parent / "src" / "picklescope"
NAMES = [
    "x.pkl",
    "x.pt",
    "x.safetensors",
    "x.gguf",
    "x.onnx",
    "x.npy",
    "x.keras",
    "x.h5",
    "config.json",
    "x.zip",
]
PREFIXES = [
    b"",
    b"\x80\x04",
    b"PK\x03\x04",
    b"GGUF\x03\x00\x00\x00",
    b"\x93NUMPY\x01\x00",
    b"\x89HDF\r\n\x1a\n",
    b"\x08\x08",
    b"\x10\x00\x00\x00\x00\x00\x00\x00{",
]


def _assert_no_internal_error(data: bytes, name: str) -> None:
    report = scan_bytes(data, name)
    internal = [f for f in report.findings if f.rule_id == "GEN901"]
    assert not internal, (name, internal)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(prefix=st.sampled_from(PREFIXES), body=st.binary(max_size=400), name=st.sampled_from(NAMES))
def test_random_bytes_never_crash(prefix: bytes, body: bytes, name: str) -> None:
    _assert_no_internal_error(prefix + body, name)


CORPUS = sorted(
    p
    for p in FIXTURES.rglob("*")
    if p.is_file()
    and p.suffix != ".toml"
    and p.name != "catalogue.json"
    and p.stat().st_size < 200_000
)


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: str(p.relative_to(FIXTURES)))
def test_mutated_fixtures_never_crash(path: Path) -> None:
    original = path.read_bytes()
    rng = random.Random(path.name)  # deterministic per fixture
    for _ in range(40):
        data = bytearray(original)
        for _ in range(rng.randint(1, 4)):
            choice = rng.random()
            pos = rng.randrange(len(data)) if data else 0
            if choice < 0.5 and data:
                data[pos] = rng.randrange(256)
            elif choice < 0.75:
                del data[pos : pos + rng.randint(1, 16)]
            else:
                data[pos:pos] = rng.randbytes(rng.randint(1, 8))
        _assert_no_internal_error(bytes(data), path.name)
        _assert_no_internal_error(bytes(data[: rng.randrange(len(data) + 1)]), path.name)


FORBIDDEN_MODULES = {"pickle", "_pickle", "cPickle", "marshal", "dill", "joblib", "torch",
                     "numpy", "yaml", "importlib", "shelve", "subprocess"}  # fmt: skip
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__"}


def test_source_never_imports_deserializers_or_evaluates_code() -> None:
    """Architectural guard: the scanner must not be able to deserialize anything."""
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                roots = {(node.module or "").split(".")[0]}
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in FORBIDDEN_CALLS, f"{path}: calls {node.func.id}"
                continue
            else:
                continue
            assert not roots & FORBIDDEN_MODULES, f"{path}: imports {roots & FORBIDDEN_MODULES}"
