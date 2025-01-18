"""Format scanners, with reference-implementation parity where a reference exists."""

from __future__ import annotations

import io
import json
import os
import struct
import zipfile
from pathlib import Path

import pytest

from picklescope import scan_bytes
from picklescope.findings import Severity
from picklescope.formats import archive, keras
from picklescope.formats import gguf as gguf_scan
from picklescope.formats import npy as npy_scan
from picklescope.formats import onnx as onnx_scan
from picklescope.formats import safetensors as st_scan
from picklescope.formats.detect import detect_format, extension_mismatch
from picklescope.formats.protobuf import ProtobufError, iter_fields, read_varint, to_signed64
from picklescope.limits import ScanLimits

from conftest import FIXTURES, LOC, make_ctx, rules_of, scan_fixture
from fixturegen import (
    GGUF_STRING,
    GGUFSpec,
    _zip,
    dense,
    gguf,
    gguf_basic_kv,
    gguf_valid,
    keras_archive,
    keras_lambda,
    keras_ref,
    keras_sequential,
    npy,
    object_array_pickle,
    onnx_model,
    onnx_tensor,
    pb_bytes,
    pb_int,
    print_pickle,
    safetensors,
    torch_data_pkl,
    torch_zip,
)

# --------------------------------------------------------------------------- safetensors

SAFETENSORS_FIXTURES = sorted(p.name for p in (FIXTURES / "safetensors").iterdir())
# Our extra parser-differential / metadata checks that the reference loader tolerates.
REFERENCE_TOLERATES = {"duplicate_key.safetensors"}


@pytest.mark.parametrize("name", SAFETENSORS_FIXTURES)
def test_safetensors_parity_with_reference(name: str) -> None:
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    data = (FIXTURES / "safetensors" / name).read_bytes()
    report = scan_fixture(f"safetensors/{name}")
    ours_ok = not any(f.rule_id.startswith("ST") for f in report.findings)
    try:
        safetensors_numpy.load(data)
        reference_ok = True
    except Exception:  # noqa: BLE001 - any rejection counts
        reference_ok = False
    if name in REFERENCE_TOLERATES:
        assert reference_ok and not ours_ok
    elif name == "zip_polyglot.safetensors":
        assert not reference_ok and not ours_ok
    elif name == "pickle_in_disguise.safetensors":
        assert not reference_ok
    else:
        assert ours_ok == reference_ok, name


def test_safetensors_sub_byte_and_metadata_types() -> None:
    ctx = make_ctx()
    header = {"a": {"dtype": "F4", "shape": [3], "data_offsets": [0, 2]}}
    st_scan.scan_safetensors(safetensors(header, b"\x00\x00"), LOC, ctx)
    assert rules_of(ctx) == {"ST304"}  # 12 bits is not a whole number of bytes
    ctx = make_ctx()
    bad_entry: dict[str, object] = {
        "a": {"dtype": "F4", "shape": [4], "data_offsets": [0, 2]},
        "b": 5,
    }
    st_scan.scan_safetensors(safetensors(bad_entry, b"\x00\x00"), LOC, ctx)
    assert rules_of(ctx) == {"ST303"}


@pytest.mark.parametrize(
    ("header", "rule"),
    [
        ({"a": {"dtype": "F32", "shape": [1], "data_offsets": [8, 4]}}, "ST304"),
        ({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0]}}, "ST304"),
        ({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4], "x": 1}}, "ST301"),
        ([1, 2], "ST301"),
    ],
)
def test_safetensors_entry_errors(header: object, rule: str) -> None:
    ctx = make_ctx()
    raw = __import__("json").dumps(header)
    st_scan.scan_safetensors(safetensors(raw, b"\x00" * 4), LOC, ctx)
    assert rule in rules_of(ctx)


def test_safetensors_short_and_non_utf8() -> None:
    ctx = make_ctx()
    st_scan.scan_safetensors(b"\x01", LOC, ctx)
    assert rules_of(ctx) == {"ST300"}
    ctx = make_ctx()
    st_scan.scan_safetensors(struct.pack("<Q", 4) + b"{\xff\xfe}", LOC, ctx)
    assert rules_of(ctx) == {"ST301"}
    ctx = make_ctx()
    st_scan.scan_safetensors(struct.pack("<Q", 2) + b"[]", LOC, ctx)
    assert rules_of(ctx) == {"ST301"}


# --------------------------------------------------------------------------- gguf


@pytest.mark.parametrize("name", ["valid_v3.gguf", "valid_big_endian.gguf"])
def test_gguf_parity_with_reference_reader(name: str) -> None:
    gguf_lib = pytest.importorskip("gguf")
    path = FIXTURES / "gguf" / name
    reader = gguf_lib.GGUFReader(str(path))
    with path.open("rb") as handle:
        data = handle.read()
    summary = gguf_scan.scan_gguf(data, LOC, make_ctx())
    assert summary is not None
    assert [t["name"] for t in summary.tensors] == [t.name for t in reader.tensors]
    assert summary.data_offset == reader.data_offset
    ref_keys = [k for k in reader.fields if not k.startswith("GGUF.")]
    assert list(summary.metadata) == ref_keys


def test_gguf_summary_fields() -> None:
    summary = gguf_scan.scan_gguf(gguf(gguf_valid()), LOC, make_ctx())
    assert summary is not None
    assert summary.version == 3 and not summary.big_endian
    assert summary.metadata["tokenizer.ggml.tokens"]["count"] == 4
    assert summary.metadata["tokenizer.ggml.tokens"]["sample"] == ["<s>", "</s>", "a"]
    assert summary.metadata["llama.context_length"] == 128


@pytest.mark.parametrize(
    ("spec", "rule"),
    [
        (GGUFSpec(kv=[("general.alignment", 4, 3)]), "GG409"),
        (GGUFSpec(kv=[("x", 9, (9, [(9, [(9, [])])]))]), None),
        (GGUFSpec(kv=[("t", GGUF_STRING, b"\xff\xfe")]), "GG414"),
        (GGUFSpec(kv=[("a", 9, (4, [1, 2]))], kv_count=1), None),
        (GGUFSpec(tensors=[("n" * 70, [1], 0, 0)], data=b"\x00" * 4), "GG406"),
        (GGUFSpec(tensors=[("big", [2**40, 2**40], 0, 0)]), "GG406"),
        (GGUFSpec(tensors=[("t", [4], 0, 0)], data=b""), "GG408"),
    ],
)
def test_gguf_edge_cases(spec: GGUFSpec, rule: str | None) -> None:
    ctx = make_ctx()
    gguf_scan.scan_gguf(gguf(spec), LOC, ctx)
    if rule is None:
        assert not ctx.findings
    else:
        assert rule in rules_of(ctx)


def test_gguf_array_limits() -> None:
    nested: tuple[int, list[object]] = (
        9,
        [(9, [(9, [(9, [(9, [(9, [(9, [(9, [(9, [])])])])])])])])],
    )
    ctx = make_ctx()
    gguf_scan.scan_gguf(gguf(GGUFSpec(kv=[("deep", 9, nested)])), LOC, ctx)
    assert "GG413" in rules_of(ctx)
    ctx = make_ctx(ScanLimits(max_gguf_array=2))
    gguf_scan.scan_gguf(gguf(GGUFSpec(kv=gguf_basic_kv())), LOC, ctx)
    assert "GG413" in rules_of(ctx)
    ctx = make_ctx()
    gguf_scan.scan_gguf(b"GGUF\x03\x00", LOC, ctx)
    assert rules_of(ctx) == {"GG400"}


@pytest.mark.parametrize(
    "template",
    [
        "{{ cycler.__init__.__globals__ }}",
        "{{ x|attr('y') }}",
        "{{ self._TemplateReference__context }}",
        "{{ os.popen('id') }}",
    ],
)
def test_chat_template_patterns(template: str) -> None:
    ctx = make_ctx()
    gguf_scan.scan_gguf(gguf(gguf_valid(kv=gguf_basic_kv(template))), LOC, ctx)
    assert rules_of(ctx) == {"GG411"}


# --------------------------------------------------------------------------- onnx


@pytest.mark.parametrize(
    "rel",
    [
        "onnx/valid.onnx",
        "onnx/external_ok/model.onnx",
        "onnx/custom_domain.onnx",
        "onnx/absolute_path.onnx",
    ],
)
def test_onnx_fixtures_parse_with_reference(rel: str) -> None:
    onnx_lib = pytest.importorskip("onnx")
    model = onnx_lib.load_model_from_string((FIXTURES / rel).read_bytes())
    summary = onnx_scan.scan_onnx((FIXTURES / rel).read_bytes(), LOC, make_ctx())
    assert summary is not None
    assert summary.ir_version == model.ir_version
    assert summary.node_count == len(model.graph.node)
    assert summary.tensor_count == len(model.graph.initializer)


def test_onnx_external_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "w.bin").write_bytes(b"\x00" * 8)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    os.symlink(outside, model_dir / "data")
    model = onnx_model(onnx_tensor("w", [2], external={"location": "data/w.bin"}))
    ctx = make_ctx()
    onnx_scan.scan_onnx(model, LOC, ctx, model_dir)
    assert rules_of(ctx) == {"OX502"}
    assert "symlink" in ctx.findings[0].message


def test_onnx_external_symlink_loop(tmp_path: Path) -> None:
    os.symlink("loop_b", tmp_path / "loop_a")
    os.symlink("loop_a", tmp_path / "loop_b")
    model = onnx_model(onnx_tensor("w", [2], external={"location": "loop_a/w.bin"}))
    ctx = make_ctx()
    onnx_scan.scan_onnx(model, LOC, ctx, tmp_path)
    assert rules_of(ctx) == {"OX502"}
    assert "cannot be resolved" in ctx.findings[0].message


@pytest.mark.parametrize(
    ("external", "rule"),
    [
        ({"location": "C:\\weights.bin"}, "OX501"),
        ({"location": "~/w.bin"}, "OX501"),
        ({"location": "a\x00b"}, "OX502"),
        ({"location": "sub\\..\\..\\w.bin"}, "OX502"),
        ({"offset": "0"}, "OX504"),
        ({"location": "w.bin", "offset": "-1"}, "OX504"),
        ({"location": "w.bin", "length": "4"}, "OX504"),
    ],
)
def test_onnx_external_lexical_checks(external: dict[str, str], rule: str) -> None:
    ctx = make_ctx()
    onnx_scan.scan_onnx(onnx_model(onnx_tensor("w", [2], external=external)), LOC, ctx)
    assert rule in rules_of(ctx)


def test_onnx_subgraphs_functions_and_depth() -> None:
    inner_tensor = onnx_tensor("c", [1], external={"location": "../x.bin"})
    attr = pb_bytes(1, "value") + pb_bytes(5, inner_tensor)
    node = pb_bytes(4, "Constant") + pb_bytes(5, attr)
    subgraph = pb_bytes(1, node)
    if_node = pb_bytes(4, "If") + pb_bytes(5, pb_bytes(1, "then_branch") + pb_bytes(6, subgraph))
    function = pb_bytes(1, "f") + pb_bytes(10, "local.fn") + pb_bytes(7, pb_bytes(4, "Relu"))
    graph = pb_bytes(1, if_node) + pb_bytes(1, pb_bytes(4, "f") + pb_bytes(7, "local.fn"))
    model = pb_int(1, 9) + pb_bytes(7, graph) + pb_bytes(25, function)
    ctx = make_ctx()
    summary = onnx_scan.scan_onnx(model, LOC, ctx)
    assert summary is not None
    assert rules_of(ctx) == {"OX502"}  # local.fn is a model function, not a custom op
    assert summary.function_domains == {"local.fn"}
    ctx = make_ctx(ScanLimits(max_protobuf_depth=2))
    onnx_scan.scan_onnx(model, LOC, ctx)
    assert "OX506" in rules_of(ctx)


def test_onnx_packed_dims_and_missing_ir_version() -> None:
    packed = pb_bytes(1, bytes([2, 3])) + pb_int(2, 1) + pb_bytes(8, "p")
    packed += pb_bytes(13, pb_bytes(1, "location") + pb_bytes(2, "w.bin"))
    packed += pb_bytes(13, pb_bytes(1, "length") + pb_bytes(2, "24"))
    graph = pb_bytes(5, packed)
    ctx = make_ctx()
    summary = onnx_scan.scan_onnx(pb_bytes(7, graph), LOC, ctx)
    assert summary is not None and summary.external[0].dims == [2, 3]
    assert rules_of(ctx) == {"OX500"}  # no ir_version; length 24 == 2*3*4 is consistent


def test_protobuf_decoder() -> None:
    assert read_varint(b"\xac\x02", 0, 2) == (300, 2)
    assert to_signed64(2**64 - 1) == -1
    raw = b"\x08\x01\x11" + b"\x00" * 8 + b"\x1d" + b"\x00" * 4
    fields = list(iter_fields(raw, 0, len(raw)))
    assert [(n, w) for n, w, _ in fields] == [(1, 0), (2, 1), (3, 5)]
    for bad in (b"\x80", b"\x00\x01", b"\x0b", b"\x09\x00", b"\x0d\x00", b"\xff" * 11):
        with pytest.raises(ProtobufError):
            list(iter_fields(bad, 0, len(bad)))


# --------------------------------------------------------------------------- archives


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("/etc/x", "absolute path"),
        ("C:/x", "drive-letter path"),
        ("a/../../x", "parent"),
        ("a\\..\\x", "parent"),
        ("a\x00b", "NUL"),
        ("archive/data.pkl", None),
    ],
)
def test_unsafe_member_names(name: str, reason: str | None) -> None:
    result = archive.unsafe_member_name(name)
    assert (result is None) if reason is None else (result is not None and reason in result)


def test_zip_classification() -> None:
    def infos(*names: str) -> list[zipfile.ZipInfo]:
        return [zipfile.ZipInfo(n) for n in names]

    assert archive.classify(infos("a/data.pkl", "a/data/0")).kind == "torch"
    assert archive.classify(infos("a/constants.pkl", "a/data.pkl", "a/code/x.py")).kind == (
        "torchscript"
    )
    assert archive.classify(infos("config.json", "metadata.json")).kind == "keras"
    assert archive.classify(infos("a.npy", "b.npy")).kind == "npz"
    assert archive.classify(infos("readme.txt")).kind == "zip"
    assert archive.classify(infos("x/constants.pkl")).kind == "zip"


def test_torch_storage_key_checks() -> None:
    data = torch_data_pkl("0", ["1"])
    ctx = make_ctx()
    archive.scan_zip(torch_zip(data, storages=("0", "1")), LOC, ctx)
    assert rules_of(ctx) == set()
    weird_pid = (
        __import__("fixturegen")
        .PickleAsm(2)
        .mark()
        .unicode("not-storage")
        .int1(1)
        .op(b"t")
        .binpersid()
        .op(b"0")
        .mark()
        .unicode("storage")
        .int1(3)
        .unicode("0")
        .op(b"t")
        .binpersid()
        .stop()
    )
    ctx = make_ctx()
    archive.scan_zip(torch_zip(weird_pid), LOC, ctx)
    messages = [f.message for f in ctx.findings]
    assert rules_of(ctx) == {"ZIP207"}
    assert any("not a torch storage" in m for m in messages)
    assert any("not an imported class" in m for m in messages)


def test_zip_member_read_failures_are_findings() -> None:
    good = _zip([("x.pkl", print_pickle())])
    # corrupt the member's data so its CRC check fails
    corrupted = good.replace(b"hello from", b"HELLO FROM")
    ctx = make_ctx()
    archive.scan_zip(corrupted, LOC, ctx)
    assert "ZIP205" in rules_of(ctx)


def test_zip_member_size_limit() -> None:
    blob = _zip([("big.pkl", print_pickle()), ("other.bin", b"\x80\x04" + b"\x00" * 64)])
    ctx = make_ctx(ScanLimits(max_member_size=16))
    archive.scan_zip(blob, LOC, ctx)
    assert "ZIP202" in rules_of(ctx)


def test_zip_sniffs_pickles_and_nested_formats_by_content() -> None:
    blob = _zip(
        [
            ("weights", print_pickle()),
            ("inner.zip", _zip([("x.pkl", print_pickle())])),
            ("model.gguf", gguf(gguf_valid())),
            ("notes.py", b"print('x')"),
        ]
    )
    ctx = make_ctx()
    archive.scan_zip(io.BytesIO(blob), LOC, ctx)
    members = {f.location.members for f in ctx.findings if f.rule_id == "PKL102"}
    assert ("weights",) in members and ("inner.zip", "x.pkl") in members
    assert "ZIP208" in rules_of(ctx)


def test_keras_archive_without_config_is_generic_zip() -> None:
    blob = _zip([("metadata.json", b"{}"), ("model.weights.h5", b"\x89HDF\r\n\x1a\n")])
    assert archive.scan_zip(blob, LOC, make_ctx()) == "zip"


def _json(value: object) -> bytes:
    return json.dumps(value).encode()


def test_keras_archive_with_only_config_json_is_scanned() -> None:
    # Keras 3 saving_lib deserialises config.json before it looks for any weights file.
    config = keras_sequential(dense(keras_ref("os", "function", "getcwd", "function")))
    ctx = make_ctx()
    assert archive.scan_zip(_zip([("config.json", _json(config))]), LOC, ctx) == "keras"
    assert "KR601" in rules_of(ctx)


def test_keras_archive_with_npz_weights_is_scanned() -> None:
    config = keras_sequential(keras_lambda())
    blob = _zip([("config.json", _json(config)), ("model.weights.npz", _zip([]))])
    assert archive.classify(zipfile.ZipFile(io.BytesIO(blob)).infolist()).kind == "keras"
    ctx = make_ctx()
    assert archive.scan_zip(blob, LOC, ctx) == "keras"
    assert "KR600" in rules_of(ctx)


def test_keras_extension_forces_keras_layout() -> None:
    blob = _zip([("config.json", b"{not json")])
    ctx = make_ctx()
    assert archive.scan_zip(blob, LOC, ctx, "model.keras") == "keras"
    assert rules_of(ctx) == {"KR604"}
    ctx = make_ctx()
    assert archive.scan_zip(blob, LOC, ctx, "bundle.zip") == "zip"
    assert rules_of(ctx) == set()


def test_non_keras_config_json_keeps_generic_layout() -> None:
    blob = _zip([("config.json", _json({"model_type": "bert", "hidden_size": 4}))])
    ctx = make_ctx()
    assert archive.scan_zip(blob, LOC, ctx) == "zip"
    assert rules_of(ctx) == set()
    nested = _zip([("sub/config.json", _json(keras_sequential(keras_lambda())))])
    assert archive.scan_zip(nested, LOC, make_ctx()) == "zip"  # loader reads the root only


def test_keras_layout_by_content_through_scan_bytes() -> None:
    config = keras_sequential(dense(keras_ref("os", "function", "getcwd", "function")))
    report = scan_bytes(_zip([("config.json", _json(config))]), "bundle.zip")
    assert report.format == "zip:keras"
    assert "KR601" in rules_of(report)


@pytest.mark.parametrize("payload_first", [True, False])
def test_every_copy_of_a_duplicate_member_is_scanned(payload_first: bool) -> None:
    copies = [print_pickle(), torch_data_pkl()]
    if not payload_first:
        copies.reverse()
    blob = torch_zip(copies[0], extra=[("archive/data.pkl", copies[1])])
    ctx = make_ctx()
    assert archive.scan_zip(blob, LOC, ctx) == "torch"
    assert {"ZIP201", "PKL102"} <= rules_of(ctx)
    payload_copy = "#1" if payload_first else "#2"
    flagged = {f.location.members for f in ctx.findings if f.rule_id == "PKL102"}
    assert flagged == {(f"archive/data.pkl{payload_copy}",)}
    assert max(f.severity for f in ctx.findings) >= Severity.HIGH


def test_duplicate_npz_entries_are_all_scanned() -> None:
    plain = npy("<f4", (1,), struct.pack("<f", 1.0))
    hostile = npy("|O", (1,), print_pickle())
    blob = _zip([("a.npy", hostile), ("a.npy", plain)])
    ctx = make_ctx()
    assert archive.scan_zip(blob, LOC, ctx) == "npz"
    assert {"ZIP201", "NP700", "PKL102"} <= rules_of(ctx)


# --------------------------------------------------------------------------- keras / npy


def test_keras_registered_custom_object_without_module() -> None:
    config = keras_sequential({"class_name": "X", "config": {}, "registered_name": "pkg>X"})
    ctx = make_ctx()
    archive.scan_zip(keras_archive(config), LOC, ctx)
    assert rules_of(ctx) == {"KR603"}


def test_keras_function_reference_without_name() -> None:
    ctx = make_ctx()
    keras.walk_config(keras_ref("keras.activations", "function", 3), LOC, ctx, "test")
    assert rules_of(ctx) == {"KR604"}


def test_keras_walk_handles_deep_nesting_without_recursion() -> None:
    node: object = {"class_name": "Lambda", "config": {}}
    for _ in range(5000):
        node = [node]
    ctx = make_ctx()
    assert keras.walk_config(node, LOC, ctx, "test") == 1


def test_keras_h5_heuristic_skips_non_configs() -> None:
    data = (
        b'\x89HDF\r\n\x1a\n{"class_name": broken {"class_name": "X"} '
        b'{"class_name": "Y", "config": {}}'
    )
    ctx = make_ctx()
    assert keras.scan_h5(data, LOC, ctx) == 1


def test_npy_versions_and_structured_dtypes() -> None:
    ctx = make_ctx()
    header = "{'descr': [('Obj', '<f4'), ('b', '|O')], 'fortran_order': False, 'shape': (1,), }"
    npy_scan.scan_npy(npy("", (), object_array_pickle(["a"]), header=header), LOC, ctx)
    assert rules_of(ctx) == {"NP700"}
    ctx = make_ctx()
    header = "{'descr': [('Opt', '<f4')], 'fortran_order': False, 'shape': (1,), }"
    npy_scan.scan_npy(npy("", (), b"\x00" * 4, header=header), LOC, ctx)
    assert rules_of(ctx) == set()
    raw = b"{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }\n"
    v2 = b"\x93NUMPY\x02\x00" + struct.pack("<I", len(raw)) + raw + b"\x00" * 4
    ctx = make_ctx()
    assert npy_scan.scan_npy(v2, LOC, ctx) is not None and not ctx.findings


@pytest.mark.parametrize(
    "data",
    [
        b"\x93NUMPY",
        b"\x93NUMPY\x09\x00\x00\x00\x00\x00",
        b"\x93NUMPY\x02\x00\x00\x00",
        b"\x93NUMPY\x01\x00\xff\xff{}",
        npy("<f4", (), b"", header="{'descr': '<f4'}"),
        npy("<f4", (), b"", header="{'descr': '<f4', 'fortran_order': False, 'shape': (-1,)}"),
    ],
)
def test_npy_malformed(data: bytes) -> None:
    ctx = make_ctx()
    npy_scan.scan_npy(data, LOC, ctx)
    assert "NP701" in rules_of(ctx)


# --------------------------------------------------------------------------- detection


@pytest.mark.parametrize(
    ("data", "name", "fmt"),
    [
        (b"", "x.bin", "empty"),
        (b"PK\x05\x06" + b"\x00" * 18, "x", "zip"),
        (b"\x08\x07", "m.onnx", "onnx"),
        (b"\x08\x07", "m.bin", "unknown"),
        (b"(lp0\n.", "x.pkl", "pickle"),
        (b"(lp0\n", "x.pkl", "unknown"),
        (b"hello", "x.safetensors", "safetensors"),
        (b'{"module": "keras", "class_name": "X", "config": {}}', "c.json", "keras-config"),
        (b'{"architectures": ["Bert"]}', "config.json", "unknown"),
        (b"{bad json", "c.json", "unknown"),
    ],
)
def test_detect_format(data: bytes, name: str, fmt: str) -> None:
    assert detect_format(data, name) == fmt


def test_extension_mismatch_rules() -> None:
    assert extension_mismatch("m.safetensors", "pickle") is not None
    assert extension_mismatch("m.bin", "unknown") is None
    assert extension_mismatch("m.keras", "unknown") is not None
    assert extension_mismatch("m.pt", "zip") is None
    assert extension_mismatch("m.txt", "pickle") is None
    assert extension_mismatch("m.npy", "empty") is None


def test_severity_parse() -> None:
    assert Severity.parse(" High ") is Severity.HIGH
    with pytest.raises(ValueError, match="unknown severity"):
        Severity.parse("urgent")


def test_torch_pid_shapes_and_directory_members() -> None:
    from fixturegen import PickleAsm

    data = (
        PickleAsm(2)
        .unicode("plain")
        .binpersid()
        .op(b"0")
        .mark()
        .unicode("storage")
        .global_("torch", "FloatStorage")
        .int1(0)
        .op(b"t")
        .binpersid()
        .stop()
    )
    ctx = make_ctx()
    archive.scan_zip(torch_zip(data, extra=[("archive/extra/", b"")]), LOC, ctx)
    assert [f.rule_id for f in ctx.findings] == ["ZIP207", "ZIP207"]


def test_nested_zip_respects_nesting_limit() -> None:
    blob = _zip([("a.zip", _zip([("b.zip", _zip([("x.pkl", print_pickle())]))]))])
    ctx = make_ctx(ScanLimits(max_nesting=1))
    archive.scan_zip(blob, LOC, ctx)
    assert rules_of(ctx) == {"PKL116"}


def test_limits_validation() -> None:
    with pytest.raises(ValueError, match="max_pickle_ops"):
        ScanLimits(max_pickle_ops=0)
    with pytest.raises(ValueError, match="compression"):
        ScanLimits(max_compression_ratio=1)
