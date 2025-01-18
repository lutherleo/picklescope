"""False-positive check on pinned real models (skipped unless `make fetch-models` ran)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from picklescope import Severity, scan_file
from picklescope.blob import open_buffer
from picklescope.formats.gguf import scan_gguf
from picklescope.formats.onnx import scan_onnx

from conftest import LOC, make_ctx

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "eval" / "manifest.json").read_text())["models"]
CACHE = ROOT / ".cache" / "models"


@pytest.mark.parametrize("entry", MANIFEST, ids=lambda e: e["id"])
def test_real_model_has_no_medium_or_higher_findings(entry: dict[str, str]) -> None:
    path = CACHE / entry["id"]
    if not path.exists():
        pytest.skip("real models not downloaded (make fetch-models)")
    report = scan_file(path, hash_file=False)
    assert report.skipped_reason is None
    assert not report.findings_at_least(Severity.MEDIUM), [f.message for f in report.findings]


def _cached(fmt: str) -> list[Path]:
    paths = [CACHE / e["id"] for e in MANIFEST if e["format"] == fmt]
    return [p for p in paths if p.exists()]


def test_real_gguf_parity_with_reference_reader() -> None:
    gguf_lib = pytest.importorskip("gguf")
    paths = _cached("gguf")
    if not paths:
        pytest.skip("real models not downloaded (make fetch-models)")
    for path in paths:
        reader = gguf_lib.GGUFReader(str(path))
        with open_buffer(path) as buf:
            summary = scan_gguf(buf, LOC, make_ctx())
        assert summary is not None
        assert [t["name"] for t in summary.tensors] == [t.name for t in reader.tensors]
        assert summary.data_offset == reader.data_offset


def test_real_onnx_parity_with_reference() -> None:
    onnx_lib = pytest.importorskip("onnx")
    paths = _cached("onnx")
    if not paths:
        pytest.skip("real models not downloaded (make fetch-models)")
    for path in paths:
        model = onnx_lib.load(str(path), load_external_data=False)
        summary = scan_onnx(path.read_bytes(), LOC, make_ctx())
        assert summary is not None
        assert summary.node_count == len(model.graph.node)
        attribute_tensors = sum(
            1 for node in model.graph.node for attr in node.attribute if attr.HasField("t")
        )
        assert summary.tensor_count == len(model.graph.initializer) + attribute_tensors
