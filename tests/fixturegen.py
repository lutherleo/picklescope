"""Deterministic generator for picklescope's test fixtures.

Run ``uv run python tests/fixturegen.py`` (or ``make fixtures``) to rewrite
``tests/fixtures/``. Every file is assembled byte by byte from the writers below; no
fixture is produced by pickling a live object, and no fixture is ever loaded.

Safety rule (enforced by ``tests/test_fixtures.py``): every callable that a fixture
imports or references is harmless (``print``, ``len``, ``str.upper``, ``colorsys``,
``textwrap``, ``os.getcwd`` ...). Fixtures that must trip the deny-list do so with
harmless deny-listed names (``builtins.getattr``, ``os.getcwd``) or with a test policy
that deny-lists a harmless function (``colorsys.rgb_to_hsv``).
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import warnings
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"
ZIP_DATE = (2020, 1, 1, 0, 0, 0)

# Every (module, name) a fixture may import or reference. Checked by the test suite.
HARMLESS_REFERENCES: frozenset[tuple[str, str]] = frozenset(
    {
        ("builtins", "print"),
        ("builtins", "len"),
        ("builtins", "str"),
        ("builtins", "getattr"),
        ("__builtin__", "getattr"),
        ("builtins", "print len"),
        ("collections", "OrderedDict"),
        ("colorsys", "rgb_to_hsv"),
        ("colorsys", "rgb_to_hsv.__code__"),
        ("textwrap", "dedent"),
        ("textwrap", "TextWrapper"),
        ("os", "getcwd"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch", "FloatStorage"),
        ("keras", "Sequential"),
        ("keras.layers", "Dense"),
        ("keras.layers", "InputLayer"),
        ("keras.layers", "Lambda"),
        ("keras.activations", "relu"),
        ("fixture_project.layers", "FancyDense"),
    }
)

# --------------------------------------------------------------------------- pickle


class PickleAsm:
    """Tiny pickle assembler: append opcodes with encoded arguments."""

    def __init__(self, proto: int | None = None) -> None:
        self.out = bytearray()
        if proto is not None:
            self.op(b"\x80", struct.pack("<B", proto))

    def op(self, code: bytes, payload: bytes = b"") -> PickleAsm:
        self.out += code + payload
        return self

    # constants
    def none(self) -> PickleAsm:
        return self.op(b"N")

    def int1(self, value: int) -> PickleAsm:
        return self.op(b"K", struct.pack("<B", value))

    def false(self) -> PickleAsm:
        return self.op(b"\x89")

    def float(self, value: float) -> PickleAsm:
        return self.op(b"G", struct.pack(">d", value))

    def unicode(self, text: str) -> PickleAsm:
        data = text.encode("utf-8")
        if len(data) < 256:
            return self.op(b"\x8c", struct.pack("<B", len(data)) + data)
        return self.op(b"X", struct.pack("<I", len(data)) + data)

    def binunicode(self, text: str) -> PickleAsm:
        data = text.encode("utf-8")
        return self.op(b"X", struct.pack("<I", len(data)) + data)

    def short_bytes(self, data: bytes) -> PickleAsm:
        return self.op(b"C", struct.pack("<B", len(data)) + data)

    def binbytes(self, data: bytes) -> PickleAsm:
        return self.op(b"B", struct.pack("<I", len(data)) + data)

    def long1(self, value: int) -> PickleAsm:
        length = (value.bit_length() + 8) // 8
        return self.op(
            b"\x8a", struct.pack("<B", length) + value.to_bytes(length, "little", signed=True)
        )

    # structure
    def mark(self) -> PickleAsm:
        return self.op(b"(")

    def tuple(self) -> PickleAsm:
        return self.op(b"t")

    def tuple1(self) -> PickleAsm:
        return self.op(b"\x85")

    def tuple2(self) -> PickleAsm:
        return self.op(b"\x86")

    def empty_tuple(self) -> PickleAsm:
        return self.op(b")")

    def empty_dict(self) -> PickleAsm:
        return self.op(b"}")

    def empty_list(self) -> PickleAsm:
        return self.op(b"]")

    def setitem(self) -> PickleAsm:
        return self.op(b"s")

    def setitems(self) -> PickleAsm:
        return self.op(b"u")

    def appends(self) -> PickleAsm:
        return self.op(b"e")

    def binput(self, index: int) -> PickleAsm:
        return self.op(b"q", struct.pack("<B", index))

    def binget(self, index: int) -> PickleAsm:
        return self.op(b"h", struct.pack("<B", index))

    def memoize(self) -> PickleAsm:
        return self.op(b"\x94")

    # imports and calls
    def global_(self, module: str, name: str) -> PickleAsm:
        return self.op(b"c", f"{module}\n{name}\n".encode())

    def stack_global(self, module: str, name: str) -> PickleAsm:
        return self.unicode(module).unicode(name).op(b"\x93")

    def inst(self, module: str, name: str) -> PickleAsm:
        return self.op(b"i", f"{module}\n{name}\n".encode())

    def reduce(self) -> PickleAsm:
        return self.op(b"R")

    def build(self) -> PickleAsm:
        return self.op(b"b")

    def obj(self) -> PickleAsm:
        return self.op(b"o")

    def newobj_ex(self) -> PickleAsm:
        return self.op(b"\x92")

    def binpersid(self) -> PickleAsm:
        return self.op(b"Q")

    def ext1(self, code: int) -> PickleAsm:
        return self.op(b"\x82", struct.pack("<B", code))

    def stop(self) -> bytes:
        self.op(b".")
        return bytes(self.out)


def call_print(asm: PickleAsm, message: str, proto4: bool = True) -> PickleAsm:
    """builtins.print(message)"""
    if proto4:
        asm.stack_global("builtins", "print")
    else:
        asm.global_("builtins", "print")
    return asm.unicode(message).tuple1().reduce()


def print_pickle(message: str = "hello from a picklescope fixture") -> bytes:
    return call_print(PickleAsm(4), message).stop()


def benign_pickle() -> bytes:
    """OrderedDict([('answer', 42), ('ratio', 0.5)]) under protocol 2."""
    asm = PickleAsm(2).global_("collections", "OrderedDict").empty_tuple().reduce()
    asm.mark().unicode("answer").int1(42).unicode("ratio").float(0.5).setitems()
    return asm.stop()


def torch_data_pkl(storage_key: str = "0", extra_keys: Iterable[str] = ()) -> bytes:
    """A state dict in the shape torch.save writes: {'weight': tensor(2x2)}."""
    asm = PickleAsm(2).global_("collections", "OrderedDict").binput(0).empty_tuple().reduce()
    asm.mark()
    for index, key in enumerate([storage_key, *extra_keys]):
        asm.unicode(f"weight{index}")
        asm.global_("torch._utils", "_rebuild_tensor_v2")
        asm.mark()
        asm.mark().unicode("storage").global_("torch", "FloatStorage").unicode(key)
        asm.unicode("cpu").int1(4).tuple().binpersid()
        asm.int1(0).int1(2).int1(2).tuple2().int1(2).int1(1).tuple2().false()
        asm.binget(0).empty_tuple().reduce().tuple().reduce()
    asm.setitems()
    return asm.stop()


def legacy_torch_file() -> bytes:
    """Pre-1.6 torch.save layout: five pickles followed by raw storage bytes."""
    magic = PickleAsm(2).long1(0x1950A86A20F9469CFC6C).stop()
    version = PickleAsm(2).op(b"M", struct.pack("<H", 1001)).stop()
    sys_info = PickleAsm(2).empty_dict().unicode("little_endian").op(b"\x88").setitem().stop()
    payload = torch_data_pkl()
    keys = PickleAsm(2).empty_list().mark().unicode("0").appends().stop()
    storage = struct.pack("<q", 4) + struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    return magic + version + sys_info + payload + keys + storage


# --------------------------------------------------------------------------- zip


def _zip(entries: Iterable[tuple[str, bytes] | tuple[str, bytes, dict[str, Any]]]) -> bytes:
    buffer = io.BytesIO()
    with warnings.catch_warnings(), zipfile.ZipFile(buffer, "w") as zf:
        warnings.simplefilter("ignore", UserWarning)  # duplicate names are deliberate
        for entry in entries:
            name, data = entry[0], entry[1]
            opts: dict[str, Any] = entry[2] if len(entry) == 3 else {}
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.create_system = 3
            info.external_attr = opts.get("mode", 0o100644) << 16
            info.compress_type = opts.get("compress", zipfile.ZIP_STORED)
            zf.writestr(info, data)
    return buffer.getvalue()


def torch_zip(data_pkl: bytes, storages: Iterable[str] = ("0",),
              extra: Iterable[tuple[str, bytes]] = ()) -> bytes:  # fmt: skip
    entries: list[tuple[str, bytes]] = [("archive/data.pkl", data_pkl)]
    entries += [(f"archive/data/{key}", struct.pack("<4f", 1, 2, 3, 4)) for key in storages]
    entries += [("archive/version", b"3\n"), ("archive/byteorder", b"little")]
    entries += list(extra)
    return _zip(entries)


def set_encrypted_flag(archive: bytes) -> bytes:
    """Set general-purpose bit 0 (encrypted) in every local and central header."""
    data = bytearray(archive)
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        pos = data.find(signature)
        while pos >= 0:
            data[pos + flag_offset] |= 0x01
            pos = data.find(signature, pos + 4)
    return bytes(data)


# --------------------------------------------------------------------------- npy


def npy(descr: str, shape: tuple[int, ...], payload: bytes, header: str | None = None) -> bytes:
    text = header or f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape!r}, }}"
    raw = text.encode("latin-1")
    pad = 64 - (10 + len(raw) + 1) % 64
    raw += b" " * pad + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(raw)) + raw + payload


def object_array_pickle(items: list[str]) -> bytes:
    asm = PickleAsm(3).empty_list().mark()
    for item in items:
        asm.unicode(item)
    return asm.appends().stop()


# --------------------------------------------------------------------------- safetensors


def safetensors(header: dict[str, Any] | str | bytes, data: bytes,
                length: int | None = None) -> bytes:  # fmt: skip
    if isinstance(header, dict):
        raw = json.dumps(header, separators=(",", ":")).encode()
    elif isinstance(header, str):
        raw = header.encode()
    else:
        raw = header
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw) if length is None else length) + raw + data


def _f32(*values: float) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


# --------------------------------------------------------------------------- gguf

GGUF_STRING, GGUF_ARRAY, GGUF_U32, GGUF_F32, GGUF_U64 = 8, 9, 4, 6, 10


@dataclass
class GGUFSpec:
    kv: list[tuple[str, int, Any]] = field(default_factory=list)
    tensors: list[tuple[str, list[int], int, int]] = field(default_factory=list)
    data: bytes = b""
    version: int = 3
    alignment: int = 32
    big_endian: bool = False
    tensor_count: int | None = None
    kv_count: int | None = None


def _gguf_str(text: str | bytes, e: str, wide: bool) -> bytes:
    raw = text.encode() if isinstance(text, str) else text
    return struct.pack(e + ("Q" if wide else "I"), len(raw)) + raw


def _gguf_value(vtype: int, value: Any, e: str, wide: bool) -> bytes:
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q",
              11: "q", 12: "d"}  # fmt: skip
    if vtype in scalar:
        return struct.pack(e + scalar[vtype], value)
    if vtype == GGUF_STRING:
        return _gguf_str(value, e, wide)
    if vtype == GGUF_ARRAY:
        etype, items = value
        head = struct.pack(e + "I", etype) + struct.pack(e + ("Q" if wide else "I"), len(items))
        return head + b"".join(_gguf_value(etype, item, e, wide) for item in items)
    return b""  # deliberately invalid types carry no payload


def gguf(spec: GGUFSpec) -> bytes:
    e = ">" if spec.big_endian else "<"
    wide = spec.version >= 2
    count = "Q" if wide else "I"
    out = bytearray(b"GGUF" + struct.pack(e + "I", spec.version))
    out += struct.pack(
        e + count, len(spec.tensors) if spec.tensor_count is None else spec.tensor_count
    )
    out += struct.pack(e + count, len(spec.kv) if spec.kv_count is None else spec.kv_count)
    for key, vtype, value in spec.kv:
        out += _gguf_str(key, e, wide) + struct.pack(e + "I", vtype)
        out += _gguf_value(vtype, value, e, wide)
    for name, dims, ggml_type, offset in spec.tensors:
        out += _gguf_str(name, e, wide) + struct.pack(e + "I", len(dims))
        out += b"".join(struct.pack(e + count, d) for d in dims)
        out += struct.pack(e + "I", ggml_type) + struct.pack(e + "Q", offset)
    out += b"\x00" * (-len(out) % spec.alignment)
    return bytes(out) + spec.data


def gguf_basic_kv(template: str | None = None) -> list[tuple[str, int, Any]]:
    kv: list[tuple[str, int, Any]] = [
        ("general.architecture", GGUF_STRING, "llama"),
        ("general.name", GGUF_STRING, "picklescope-fixture"),
        ("llama.context_length", GGUF_U32, 128),
        ("tokenizer.ggml.tokens", GGUF_ARRAY, (GGUF_STRING, ["<s>", "</s>", "a", "b"])),
        ("tokenizer.ggml.scores", GGUF_ARRAY, (GGUF_F32, [0.0, 0.0, -1.0, -2.0])),
    ]
    if template is not None:
        kv.append(("tokenizer.chat_template", GGUF_STRING, template))
    return kv


BENIGN_TEMPLATE = (
    "{% for message in messages %}<|{{ message['role'] }}|>{{ message['content'] }}"
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def gguf_valid(**overrides: Any) -> GGUFSpec:
    """Two tensors: F32[4] at 0 and Q8_0[32] (34 bytes) at 32."""
    data = _f32(1, 2, 3, 4) + b"\x00" * 16 + b"\x01" * 34
    spec = GGUFSpec(
        kv=gguf_basic_kv(BENIGN_TEMPLATE),
        tensors=[("token_embd.weight", [4], 0, 0), ("output.weight", [32], 8, 32)],
        data=data,
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


# --------------------------------------------------------------------------- onnx


def _varint(value: int) -> bytes:
    out = bytearray()
    value &= (1 << 64) - 1
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def pb_int(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value)


def pb_bytes(number: int, payload: bytes | str) -> bytes:
    raw = payload.encode() if isinstance(payload, str) else payload
    return _varint(number << 3 | 2) + _varint(len(raw)) + raw


def onnx_tensor(name: str, dims: list[int], raw: bytes | None = None,
                external: dict[str, str] | None = None) -> bytes:  # fmt: skip
    out = b"".join(pb_int(1, d) for d in dims) + pb_int(2, 1) + pb_bytes(8, name)
    if raw is not None:
        out += pb_bytes(9, raw)
    if external is not None:
        for key, value in external.items():
            out += pb_bytes(13, pb_bytes(1, key) + pb_bytes(2, value))
        out += pb_int(14, 1)
    return out


def onnx_model(initializer: bytes, op_type: str = "Add", domain: str = "") -> bytes:
    node = pb_bytes(1, "x") + pb_bytes(1, "w") + pb_bytes(2, "y") + pb_bytes(3, "node0")
    node += pb_bytes(4, op_type)
    if domain:
        node += pb_bytes(7, domain)
    graph = pb_bytes(1, node) + pb_bytes(2, "fixture-graph") + pb_bytes(5, initializer)
    graph += pb_bytes(11, pb_bytes(1, "x")) + pb_bytes(12, pb_bytes(1, "y"))
    opsets = pb_bytes(8, pb_bytes(1, "") + pb_int(2, 17))
    if domain:
        opsets += pb_bytes(8, pb_bytes(1, domain) + pb_int(2, 1))
    return pb_int(1, 8) + pb_bytes(2, "picklescope-fixtures") + opsets + pb_bytes(7, graph)


# --------------------------------------------------------------------------- keras


def keras_ref(
    module: str, class_name: str, config: Any, registered: str | None = None
) -> dict[str, Any]:
    return {"module": module, "class_name": class_name, "config": config,
            "registered_name": registered}  # fmt: skip


def keras_sequential(*layers: dict[str, Any]) -> dict[str, Any]:
    base = [keras_ref("keras.layers", "InputLayer", {"batch_shape": [None, 4], "name": "in"})]
    return keras_ref("keras", "Sequential", {"name": "fixture", "layers": base + list(layers)})


def dense(activation: Any = "relu", name: str = "dense") -> dict[str, Any]:
    return keras_ref("keras.layers", "Dense", {"name": name, "units": 2, "activation": activation})


LAMBDA_CODE = "bGFtYmRhIHg6IHgK"  # base64 of the text "lambda x: x" - placeholder, not bytecode


def keras_lambda() -> dict[str, Any]:
    function = {"class_name": "__lambda__",
                "config": {"code": LAMBDA_CODE, "defaults": None, "closure": None}}  # fmt: skip
    return keras_ref("keras.layers", "Lambda", {"name": "identity_lambda", "function": function})


def keras_archive(config: dict[str, Any] | bytes) -> bytes:
    raw = config if isinstance(config, bytes) else json.dumps(config, indent=1).encode()
    metadata = json.dumps({"keras_version": "3.6.0", "date_saved": "2020-01-01@00:00:00"})
    weights = b"\x89HDF\r\n\x1a\n" + b"\x00" * 56
    return _zip(
        [("metadata.json", metadata.encode()), ("config.json", raw), ("model.weights.h5", weights)]
    )


def legacy_h5(config: dict[str, Any]) -> bytes:
    """Synthetic stand-in for a Keras 2 HDF5 file: HDF5 signature, padding, and the
    model_config attribute string. Not a valid HDF5 tree; exercises the heuristic."""
    blob = json.dumps(config).encode()
    return b"\x89HDF\r\n\x1a\n" + b"\x00" * 120 + blob + b"\x00" * 64


# --------------------------------------------------------------------------- catalogue

Builder = Callable[[], bytes]


@dataclass(frozen=True)
class Fixture:
    path: str
    build: Builder
    description: str
    malicious_pattern: bool = False


def _fixtures() -> list[Fixture]:
    f = Fixture
    p4 = PickleAsm
    return [
        # ---- plain pickles
        f("pickle/benign_ordereddict_p2.pkl", benign_pickle, "allow-listed OrderedDict"),
        f(
            "pickle/benign_text_p0.pkl",
            lambda: b"(dp0\nVkey\np1\nI42\nsVname\np2\nVfixture\np3\ns.",
            "protocol-0 dict of constants",
        ),
        f("pickle/call_print.pkl", print_pickle, "invokes builtins.print", True),
        f(
            "pickle/import_only_dedent.pkl",
            lambda: p4(2).global_("textwrap", "dedent").stop(),
            "imports textwrap.dedent without calling it",
        ),
        f(
            "pickle/getattr_gadget.pkl",
            lambda: (
                p4(2)
                .global_("builtins", "getattr")
                .global_("builtins", "str")
                .unicode("upper")
                .tuple2()
                .reduce()
                .unicode("abc")
                .tuple1()
                .reduce()
                .stop()
            ),
            "getattr(str, 'upper')('abc'): deny-listed gadget plus indirect call",
            True,
        ),
        f(
            "pickle/stack_global_memo.pkl",
            lambda: (
                p4(4)
                .unicode("builtins")
                .memoize()
                .unicode("print")
                .memoize()
                .op(b"0")
                .op(b"0")
                .binget(0)
                .binget(1)
                .op(b"\x93")
                .unicode("memo route")
                .tuple1()
                .reduce()
                .stop()
            ),
            "STACK_GLOBAL operands fetched from the memo",
            True,
        ),
        f(
            "pickle/dynamic_stack_global.pkl",
            lambda: (
                p4(4)
                .unicode("builtins")
                .global_("builtins", "str")
                .unicode("len")
                .tuple1()
                .reduce()
                .op(b"\x93")
                .stop()
            ),
            "STACK_GLOBAL name computed by a call",
            True,
        ),
        f(
            "pickle/dunder_traversal.pkl",
            lambda: p4(4).stack_global("colorsys", "rgb_to_hsv.__code__").stop(),
            "dotted name walking through __code__",
            True,
        ),
        f(
            "pickle/ext_opcode.pkl",
            lambda: p4(2).ext1(1).stop(),
            "EXT1 extension-registry import",
            True,
        ),
        f(
            "pickle/global_mutation.pkl",
            lambda: (
                p4(2)
                .global_("collections", "OrderedDict")
                .empty_dict()
                .unicode("fixture_marker")
                .int1(1)
                .setitem()
                .build()
                .stop()
            ),
            "BUILD applied to an imported class",
            True,
        ),
        f(
            "pickle/persistent_id.pkl",
            lambda: p4(2).unicode("fixture").int1(1).tuple2().binpersid().stop(),
            "persistent id outside a torch archive",
        ),
        f(
            "pickle/nested_pickle.pkl",
            lambda: p4(4).binbytes(print_pickle("nested hello")).stop(),
            "bytes constant that is itself a print pickle",
            True,
        ),
        f(
            "pickle/two_pickles.pkl",
            lambda: benign_pickle() + print_pickle(),
            "benign pickle followed by a second pickle",
            True,
        ),
        f(
            "pickle/trailing_data.pkl",
            lambda: benign_pickle() + b"trailing bytes\n",
            "non-pickle bytes after STOP",
        ),
        f("pickle/truncated.pkl", lambda: print_pickle()[:-6], "stream cut short"),
        f("pickle/unknown_opcode.pkl", lambda: p4(2).op(b"\xff").stop(), "opcode 0xff", True),
        f("pickle/protocol_9.pkl", lambda: p4(9).none().stop(), "PROTO 9", True),
        f(
            "pickle/malformed_name.pkl",
            lambda: p4(2).global_("builtins", "print len").stop(),
            "global name containing a space",
            True,
        ),
        f(
            "pickle/inst_p0.pkl",
            lambda: b"(F0.1\nF0.2\nF0.3\nicolorsys\nrgb_to_hsv\np0\n.",
            "protocol-0 INST calling colorsys.rgb_to_hsv",
            True,
        ),
        f(
            "pickle/obj_p1.pkl",
            lambda: p4().mark().global_("textwrap", "TextWrapper").obj().stop(),
            "protocol-1 OBJ instantiating textwrap.TextWrapper",
            True,
        ),
        f(
            "pickle/newobj_ex_p4.pkl",
            lambda: (
                p4(4)
                .stack_global("textwrap", "TextWrapper")
                .empty_tuple()
                .empty_dict()
                .newobj_ex()
                .stop()
            ),
            "NEWOBJ_EX instantiating textwrap.TextWrapper",
            True,
        ),
        f(
            "pickle/py2_builtin_getattr.pkl",
            lambda: p4(2).global_("__builtin__", "getattr").stop(),
            "Python-2 spelling of builtins.getattr",
            True,
        ),
        f(
            "pickle/colorsys_call.pkl",
            lambda: (
                p4(2)
                .global_("colorsys", "rgb_to_hsv")
                .float(0.5)
                .float(0.5)
                .float(0.5)
                .op(b"\x87")
                .reduce()
                .stop()
            ),
            "calls colorsys.rgb_to_hsv (deny-listed by tests/fixtures/policy-test.toml)",
            True,
        ),
        f("pickle/legacy_torch.pt", legacy_torch_file, "legacy (non-zip) torch.save layout"),
        # ---- torch zip archives
        f(
            "torch/benign_state_dict.pt",
            lambda: torch_zip(torch_data_pkl()),
            "torch.save-shaped archive",
        ),
        f(
            "torch/print_in_data_pkl.pt",
            lambda: torch_zip(print_pickle()),
            "archive whose data.pkl calls print",
            True,
        ),
        f(
            "torch/shadowed_data_pkl.pt",
            lambda: torch_zip(print_pickle(), extra=[("archive/data.pkl", torch_data_pkl())]),
            "two archive/data.pkl entries; the first calls print, the second is benign",
            True,
        ),
        f(
            "torch/missing_storage.pt",
            lambda: torch_zip(torch_data_pkl("7", ["../escape"]), storages=()),
            "persistent ids referencing absent and path-like records",
        ),
        f(
            "torch/torchscript.pt",
            lambda: torch_zip(
                torch_data_pkl(),
                extra=[
                    ("archive/code/__torch__.py", b"def forward(self, x):\n    return x\n"),
                    ("archive/constants.pkl", PickleAsm(2).empty_tuple().stop()),
                ],
            ),
            "TorchScript archive with code/",
        ),
        # ---- generic zips
        f(
            "zip/hostile_layout.zip",
            lambda: _zip(
                [
                    ("../escape.txt", b"x"),
                    ("dup.txt", b"a"),
                    ("dup.txt", b"b"),
                    ("link", b"target", {"mode": 0o120777}),
                    ("payload.so", b"\x7fELF"),
                ]
            ),
            "zip-slip, duplicate, symlink and native-library members",
            True,
        ),
        f(
            "zip/bomb.zip",
            lambda: _zip(
                [("zeros.bin", b"\x00" * (20 * 1024 * 1024), {"compress": zipfile.ZIP_DEFLATED})]
            ),
            "20 MiB of zeros deflated to ~20 KiB",
        ),
        f(
            "zip/encrypted.zip",
            lambda: set_encrypted_flag(_zip([("secret.pkl", benign_pickle())])),
            "member flagged as encrypted",
        ),
        f(
            "zip/nested_npy.zip",
            lambda: _zip([("inner/objects.npy", npy("|O", (1,), print_pickle()))]),
            "object .npy holding a print pickle inside a generic zip",
            True,
        ),
        f("zip/not_really.zip", lambda: b"PK\x03\x04" + b"\x00" * 40, "corrupt zip"),
        # ---- numpy
        f("npy/float32.npy", lambda: npy("<f4", (4,), _f32(1, 2, 3, 4)), "plain float array"),
        f(
            "npy/object_strings.npy",
            lambda: npy("|O", (2,), object_array_pickle(["alpha", "beta"])),
            "object array of strings",
        ),
        f(
            "npy/object_print.npy",
            lambda: npy("|O", (1,), print_pickle()),
            "object array whose pickle calls print",
            True,
        ),
        f(
            "npy/bad_header.npy",
            lambda: npy("", (), b"", header="__import__('os')"),
            "header that is not a literal",
        ),
        f(
            "npy/arrays.npz",
            lambda: _zip(
                [
                    ("a.npy", npy("<f4", (4,), _f32(1, 2, 3, 4))),
                    ("b.npy", npy("<f4", (2,), _f32(5, 6))),
                ]
            ),
            "npz with two plain arrays",
        ),
        # ---- safetensors
        f(
            "safetensors/valid.safetensors",
            lambda: safetensors(
                {
                    "__metadata__": {"format": "pt"},
                    "a": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
                    "b": {"dtype": "I8", "shape": [4], "data_offsets": [16, 20]},
                },
                _f32(1, 2, 3, 4) + b"\x01\x02\x03\x04",
            ),
            "two tensors tiling the buffer",
        ),
        f(
            "safetensors/header_too_large.safetensors",
            lambda: struct.pack("<Q", 200 * 1024**2) + b"{}",
            "length prefix of 200 MB",
        ),
        f(
            "safetensors/header_past_eof.safetensors",
            lambda: safetensors(
                {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, _f32(1), length=4096
            ),
            "length prefix beyond the file",
        ),
        f(
            "safetensors/not_json.safetensors",
            lambda: safetensors("{not json", b""),
            "header is not JSON",
        ),
        f(
            "safetensors/bad_dtype.safetensors",
            lambda: safetensors(
                {"a": {"dtype": "F31", "shape": [1], "data_offsets": [0, 4]}}, _f32(1)
            ),
            "unknown dtype",
        ),
        f(
            "safetensors/bad_shape.safetensors",
            lambda: safetensors(
                {"a": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}}, _f32(1)
            ),
            "negative dimension",
        ),
        f(
            "safetensors/size_mismatch.safetensors",
            lambda: safetensors(
                {"a": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}, _f32(1, 2)
            ),
            "range does not match shape x dtype",
        ),
        f(
            "safetensors/overlap.safetensors",
            lambda: safetensors(
                {
                    "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                    "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
                },
                _f32(1, 2, 3),
            ),
            "two tensors share bytes",
        ),
        f(
            "safetensors/gap.safetensors",
            lambda: safetensors(
                {"a": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}, _f32(1, 2)
            ),
            "unreferenced bytes before the first tensor",
        ),
        f(
            "safetensors/duplicate_key.safetensors",
            lambda: safetensors(
                '{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
                '"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}',
                _f32(1),
            ),
            "tensor name repeated in the JSON header",
        ),
        f(
            "safetensors/bad_metadata.safetensors",
            lambda: safetensors(
                {
                    "__metadata__": {"epoch": 3},
                    "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                },
                _f32(1),
            ),
            "non-string metadata value",
        ),
        f(
            "safetensors/appended_payload.safetensors",
            lambda: safetensors(
                {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
                _f32(1) + b"extra bytes after the last tensor",
            ),
            "bytes appended after the tensors",
        ),
        f(
            "safetensors/pickle_in_disguise.safetensors",
            print_pickle,
            "a print pickle with a .safetensors name",
            True,
        ),
        f(
            "safetensors/zip_polyglot.safetensors",
            lambda: (
                safetensors({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, _f32(1))
                + _zip([("data.pkl", print_pickle())])
            ),
            "safetensors file with a zip archive appended",
            True,
        ),
        # ---- gguf
        f("gguf/valid_v3.gguf", lambda: gguf(gguf_valid()), "F32 and Q8_0 tensors"),
        f(
            "gguf/valid_v1.gguf",
            lambda: gguf(gguf_valid(version=1)),
            "same content in the v1 layout",
        ),
        f(
            "gguf/valid_big_endian.gguf",
            lambda: gguf(gguf_valid(big_endian=True)),
            "big-endian v3 header",
        ),
        f("gguf/bad_magic.gguf", lambda: b"GGUX" + gguf(gguf_valid())[4:], "wrong magic"),
        f("gguf/bad_version.gguf", lambda: gguf(gguf_valid(version=9)), "version 9"),
        f(
            "gguf/huge_counts.gguf",
            lambda: gguf(GGUFSpec(tensor_count=2**40, kv_count=2**40)),
            "counts that cannot fit in the file",
            True,
        ),
        f(
            "gguf/string_oob.gguf",
            lambda: b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, 0xFFFF) + b"general.name",
            "key length beyond the file",
            True,
        ),
        f(
            "gguf/bad_value_type.gguf",
            lambda: gguf(GGUFSpec(kv=[("general.name", 99, None)])),
            "value type 99",
        ),
        f(
            "gguf/duplicate_key.gguf",
            lambda: gguf(gguf_valid(kv=[*gguf_basic_kv(), ("general.name", GGUF_STRING, "dup")])),
            "general.name twice",
        ),
        f(
            "gguf/too_many_dims.gguf",
            lambda: gguf(gguf_valid(tensors=[("t", [1, 1, 1, 1, 4], 0, 0)])),
            "five dimensions",
            True,
        ),
        f(
            "gguf/unknown_type.gguf",
            lambda: gguf(gguf_valid(tensors=[("t", [4], 99, 0)])),
            "ggml type 99",
        ),
        f(
            "gguf/tensor_oob.gguf",
            lambda: gguf(gguf_valid(tensors=[("t", [4096], 0, 0)])),
            "tensor larger than the file",
            True,
        ),
        f(
            "gguf/misaligned.gguf",
            lambda: gguf(gguf_valid(tensors=[("t", [2], 0, 4)])),
            "offset 4 with alignment 32",
        ),
        f(
            "gguf/overlap.gguf",
            lambda: gguf(gguf_valid(tensors=[("a", [4], 0, 0), ("b", [4], 0, 0)])),
            "two tensors at the same offset",
        ),
        f(
            "gguf/duplicate_tensor.gguf",
            lambda: gguf(gguf_valid(tensors=[("t", [4], 0, 0), ("t", [4], 0, 32)])),
            "tensor name repeated",
        ),
        f(
            "gguf/block_mismatch.gguf",
            lambda: gguf(gguf_valid(tensors=[("q", [30], 8, 0)])),
            "Q8_0 row of 30 elements",
        ),
        f(
            "gguf/chat_template_dunder.gguf",
            lambda: gguf(gguf_valid(kv=gguf_basic_kv("{{ ''.__class__.__name__ }}"))),
            "chat template reaching __class__ (prints a class name)",
            True,
        ),
        # ---- onnx
        f(
            "onnx/valid.onnx",
            lambda: onnx_model(onnx_tensor("w", [2], raw=_f32(1, 2))),
            "inline initializer",
        ),
        f(
            "onnx/external_ok/model.onnx",
            lambda: onnx_model(
                onnx_tensor(
                    "w", [2], external={"location": "weights.bin", "offset": "0", "length": "8"}
                )
            ),
            "external data next to the model",
        ),
        f("onnx/external_ok/weights.bin", lambda: _f32(1, 2), "external tensor bytes"),
        f(
            "onnx/absolute_path.onnx",
            lambda: onnx_model(onnx_tensor("w", [2], external={"location": "/nonexistent/w.bin"})),
            "absolute external location",
            True,
        ),
        f(
            "onnx/traversal.onnx",
            lambda: onnx_model(onnx_tensor("w", [2], external={"location": "../outside.bin"})),
            "external location with '..'",
            True,
        ),
        f(
            "onnx/missing_external.onnx",
            lambda: onnx_model(onnx_tensor("w", [2], external={"location": "absent.bin"})),
            "external file that does not exist",
        ),
        f(
            "onnx/bad_range/model.onnx",
            lambda: onnx_model(
                onnx_tensor(
                    "w", [2], external={"location": "weights.bin", "offset": "64", "length": "8"}
                )
            ),
            "offset beyond the external file",
        ),
        f("onnx/bad_range/weights.bin", lambda: _f32(1, 2), "8-byte external file"),
        f(
            "onnx/custom_domain.onnx",
            lambda: onnx_model(
                onnx_tensor("w", [2], raw=_f32(1, 2)), "FixtureOp", "org.example.fixture"
            ),
            "operator from a custom domain",
        ),
        f("onnx/garbage.onnx", lambda: b"\x08\x08\x12\xff\xff\xff\xff\x0f", "truncated protobuf"),
        # ---- keras
        f(
            "keras/benign.keras",
            lambda: keras_archive(keras_sequential(dense())),
            "Sequential with Dense",
        ),
        f(
            "keras/lambda.keras",
            lambda: keras_archive(keras_sequential(keras_lambda())),
            "Keras 3 Lambda layer",
            True,
        ),
        f(
            "keras/function_print.keras",
            lambda: keras_archive(
                keras_sequential(dense(keras_ref("builtins", "function", "print", "function")))
            ),
            "activation referencing builtins.print",
            True,
        ),
        f(
            "keras/function_denied.keras",
            lambda: keras_archive(
                keras_sequential(dense(keras_ref("os", "function", "getcwd", "function")))
            ),
            "activation referencing deny-listed os.getcwd",
            True,
        ),
        f(
            "keras/custom_object.keras",
            lambda: keras_archive(
                keras_sequential(
                    keras_ref(
                        "fixture_project.layers", "FancyDense", {"units": 2}, "fixture>FancyDense"
                    )
                )
            ),
            "custom layer class from a third-party module",
        ),
        f(
            "keras/allowed_function.keras",
            lambda: keras_archive(
                keras_sequential(dense(keras_ref("keras.activations", "function", "relu", "relu")))
            ),
            "activation referencing keras.activations.relu",
        ),
        f("keras/bad_config.keras", lambda: keras_archive(b"{not json"), "invalid config.json"),
        f(
            "keras/config_only.keras",
            lambda: _zip(
                [
                    (
                        "config.json",
                        json.dumps(
                            keras_sequential(
                                dense(keras_ref("os", "function", "getcwd", "function"))
                            )
                        ).encode(),
                    )
                ]
            ),
            "archive holding only config.json, activation referencing deny-listed os.getcwd",
            True,
        ),
        f(
            "keras/lambda_npz_weights.keras",
            lambda: _zip(
                [
                    ("config.json", json.dumps(keras_sequential(keras_lambda())).encode()),
                    ("model.weights.npz", _zip([])),
                ]
            ),
            "Keras 3 Lambda layer with an .npz weights store and no metadata.json",
            True,
        ),
        f(
            "keras/legacy_lambda.h5",
            lambda: legacy_h5(
                {
                    "class_name": "Sequential",
                    "config": {
                        "layers": [
                            {
                                "class_name": "Lambda",
                                "config": {
                                    "name": "lambda",
                                    "function_type": "lambda",
                                    "function": [LAMBDA_CODE, None, None],
                                },
                            }
                        ]
                    },
                }
            ),
            "synthetic HDF5 with a Keras 2 Lambda config",
            True,
        ),
        f(
            "keras/preset/config.json",
            lambda: json.dumps(keras_sequential(keras_lambda()), indent=1).encode(),
            "KerasHub-style preset config.json with a Lambda",
            True,
        ),
    ]


FIXTURES_LIST: list[Fixture] = _fixtures()

POLICY_TEST = """\
# Test policy: deny-lists a harmless function so the CRITICAL path can be exercised
# end to end without any dangerous callable in the fixtures.
name = "fixture-test"

[[deny]]
pattern = "colorsys:rgb_to_hsv"
reason = "harmless function deny-listed for tests"
category = "test"
"""


def write_all(out: Path) -> list[Path]:
    """Write every fixture (and the test policy) under ``out``; return the paths."""
    written = []
    for fixture in FIXTURES_LIST:
        target = out / fixture.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(fixture.build())
        written.append(target)
    policy = out / "policy-test.toml"
    policy.write_text(POLICY_TEST, encoding="utf-8")
    written.append(policy)
    catalogue = {
        fx.path: {"description": fx.description, "malicious_pattern": fx.malicious_pattern}
        for fx in FIXTURES_LIST
    }
    index = out / "catalogue.json"
    index.write_text(json.dumps(catalogue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written.append(index)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="regenerate picklescope test fixtures")
    parser.add_argument("--out", type=Path, default=FIXTURES)
    args = parser.parse_args(argv)
    paths = write_all(args.out)
    print(f"wrote {len(paths)} files under {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
