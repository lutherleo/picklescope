"""GGUF header, metadata and tensor-table sanity checks.

Layout (v2/v3; v1 uses 32-bit counts and lengths)::

    "GGUF" | u32 version | u64 n_tensors | u64 n_kv
    n_kv x  (string key, u32 type, value)
    n_tensors x (string name, u32 n_dims, u64 dims[n_dims], u32 ggml_type, u64 offset)
    padding to general.alignment (default 32) | tensor data

Every length and count is checked against the bytes that remain before it is used, so
hostile counts cannot drive large allocations. Version 3 files may be big-endian; the
byte order is detected from the version field like the reference ``gguf`` reader.
"""

from __future__ import annotations

import itertools
import math
import re
import struct
from dataclasses import dataclass, field
from typing import Any

from picklescope.blob import Buffer
from picklescope.context import ScanContext
from picklescope.findings import Location

MAGIC = b"GGUF"
GGML_MAX_DIMS = 4
GGML_MAX_NAME = 64
DEFAULT_ALIGNMENT = 32
MAX_NESTING = 8

# ggml type id -> (name, elements per block, bytes per block). From ggml.h / ggml.c
# type traits; ids 4, 5 and 31-33 were removed upstream and are reported as unknown.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
}  # fmt: skip

# Metadata value types: id -> (struct code or None for variable-size, name).
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL = range(8)
_STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 8, 9, 10, 11, 12
_SCALAR_FORMATS: dict[int, str] = {
    _UINT8: "B", _INT8: "b", _UINT16: "H", _INT16: "h", _UINT32: "I", _INT32: "i",
    _FLOAT32: "f", _BOOL: "?", _UINT64: "Q", _INT64: "q", _FLOAT64: "d",
}  # fmt: skip

_TEMPLATE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"__[A-Za-z][A-Za-z0-9_]*__"), "dunder attribute access"),
    (re.compile(r"\|\s*attr\s*\("), "attr() filter (attribute access by string)"),
    (re.compile(r"_TemplateReference__context"), "template context escape"),
    (
        re.compile(r"\b(?:popen|subprocess|system|eval|exec|import_module)\s*\("),
        "process or code execution call",
    ),
)


class _TruncatedError(Exception):
    def __init__(self, rule: str, offset: int, message: str) -> None:
        super().__init__(message)
        self.rule = rule
        self.offset = offset
        self.message = message


@dataclass(slots=True)
class GGUFSummary:
    version: int
    big_endian: bool
    tensor_count: int
    kv_count: int
    alignment: int = DEFAULT_ALIGNMENT
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: list[dict[str, Any]] = field(default_factory=list)
    data_offset: int | None = None


class _Reader:
    def __init__(
        self, buf: Buffer, endian: str, version: int, ctx: ScanContext, loc: Location
    ) -> None:
        self.buf = buf
        self.pos = 0
        self.endian = endian
        self.wide = version >= 2
        self.ctx = ctx
        self.loc = loc

    def remaining(self) -> int:
        return len(self.buf) - self.pos

    def scalar(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        if self.remaining() < size:
            raise _TruncatedError("GG400", self.pos, f"truncated reading {fmt!r} field")
        (value,) = struct.unpack_from(self.endian + fmt, self.buf, self.pos)
        self.pos += size
        return value

    def length(self) -> int:
        return int(self.scalar("Q" if self.wide else "I"))

    def string(self, what: str) -> str:
        start = self.pos
        n = self.length()
        if n > self.remaining():
            raise _TruncatedError(
                "GG403", start, f"{what} length {n} exceeds the {self.remaining()} bytes left"
            )
        raw = self.buf[self.pos : self.pos + n]
        self.pos += n
        try:
            return bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            self.ctx.emit("GG414", self.loc.at(start), f"{what} is not valid UTF-8")
            return bytes(raw).decode("utf-8", "replace")

    def value(self, vtype: int, depth: int, key: str) -> Any:
        if vtype in _SCALAR_FORMATS:
            return self.scalar(_SCALAR_FORMATS[vtype])
        if vtype == _STRING:
            return self.string(f"value of {key!r}")
        if vtype == _ARRAY:
            return self.array(depth, key)
        raise _TruncatedError("GG404", self.pos - 4, f"key {key!r} has unknown value type {vtype}")

    def array(self, depth: int, key: str) -> Any:
        start = self.pos
        if depth >= MAX_NESTING:
            raise _TruncatedError(
                "GG413", start, f"array nesting deeper than {MAX_NESTING} in {key!r}"
            )
        etype = int(self.scalar("I"))
        count = self.length()
        min_size = (
            struct.calcsize(_SCALAR_FORMATS[etype])
            if etype in _SCALAR_FORMATS
            else (8 if self.wide else 4)
        )
        if count * min_size > self.remaining():
            raise _TruncatedError(
                "GG403",
                start,
                f"array {key!r} declares {count} elements, more than the file can hold",
            )
        if count > self.ctx.limits.max_gguf_array:
            raise _TruncatedError(
                "GG413",
                start,
                f"array {key!r} has {count} elements (limit {self.ctx.limits.max_gguf_array})",
            )
        if etype in _SCALAR_FORMATS:
            self.pos += count * min_size
            return {"array_type": etype, "count": count}
        items = [self.value(etype, depth + 1, key) for _ in range(count)]
        if etype == _STRING:
            return {"array_type": etype, "count": count, "sample": items[:3]}
        return {"array_type": etype, "count": count}


def _byte_order(buf: Buffer) -> tuple[str, int]:
    (le_version,) = struct.unpack_from("<I", buf, 4)
    if le_version & 0xFFFF == 0:
        (be_version,) = struct.unpack_from(">I", buf, 4)
        return ">", be_version
    return "<", le_version


def _check_template(key: str, value: Any, loc: Location, ctx: ScanContext) -> None:
    if not isinstance(value, str):
        return
    hits = sorted({label for pattern, label in _TEMPLATE_PATTERNS if pattern.search(value)})
    if hits:
        ctx.emit(
            "GG411",
            loc,
            f"{key} contains {', '.join(hits)}",
            key=key,
            patterns=hits,
            template_length=len(value),
        )


def _tensor_nbytes(dims: list[int], ggml_type: int) -> int | None:
    traits = GGML_TYPES.get(ggml_type)
    if traits is None:
        return None
    _name, block, type_size = traits
    rows = math.prod(dims[1:]) if len(dims) > 1 else 1
    return (dims[0] // block) * type_size * rows if dims else type_size


def _read_tensor_infos(
    reader: _Reader, summary: GGUFSummary, loc: Location, ctx: ScanContext
) -> None:
    names: set[str] = set()
    for _ in range(summary.tensor_count):
        start = reader.pos
        name = reader.string("tensor name")
        if len(name.encode()) >= GGML_MAX_NAME:
            ctx.emit(
                "GG406",
                loc.at(start),
                f"tensor name {name[:40]!r}... is "
                f"{len(name.encode())} bytes (ggml limit {GGML_MAX_NAME - 1})",
            )
        if name in names:
            ctx.emit("GG412", loc.at(start), f"tensor {name!r} is defined twice", tensor=name)
        names.add(name)
        n_dims = int(reader.scalar("I"))
        if n_dims > GGML_MAX_DIMS:
            raise _TruncatedError(
                "GG406",
                start,
                f"tensor {name!r} has {n_dims} dimensions "
                f"(max {GGML_MAX_DIMS}); tensor table not parsed further",
            )
        dims = [int(reader.scalar("Q" if reader.wide else "I")) for _ in range(n_dims)]
        ggml_type = int(reader.scalar("I"))
        offset = int(reader.scalar("Q"))
        if any(d >= 2**63 for d in dims) or math.prod(dims) >= 2**63:
            ctx.emit(
                "GG406", loc.at(start), f"tensor {name!r} dims {dims} overflow int64", tensor=name
            )
        summary.tensors.append(
            {"name": name, "dims": dims, "type": ggml_type, "offset": offset, "at": start}
        )


def _check_tensors(summary: GGUFSummary, size: int, loc: Location, ctx: ScanContext) -> None:
    assert summary.data_offset is not None
    spans: list[tuple[int, int, str]] = []
    for tensor in summary.tensors:
        name, dims, ggml_type, offset = (tensor[k] for k in ("name", "dims", "type", "offset"))
        here = loc.at(tensor["at"])
        traits = GGML_TYPES.get(ggml_type)
        if traits is None:
            ctx.emit(
                "GG407", here, f"tensor {name!r} has unknown ggml type {ggml_type}", tensor=name
            )
            continue
        type_name, block, _type_size = traits
        if dims and dims[0] % block:
            ctx.emit(
                "GG415",
                here,
                f"tensor {name!r}: row length {dims[0]} is not a "
                f"multiple of the {type_name} block size {block}",
                tensor=name,
            )
        if offset % summary.alignment:
            ctx.emit(
                "GG409",
                here,
                f"tensor {name!r} offset {offset} is not aligned to {summary.alignment}",
                tensor=name,
            )
        nbytes = _tensor_nbytes(dims, ggml_type) or 0
        begin = summary.data_offset + offset
        end = begin + nbytes
        if end > size:
            ctx.emit(
                "GG408",
                here,
                f"tensor {name!r} occupies bytes [{begin}, {end}) beyond the {size}-byte file",
                tensor=name,
            )
        spans.append((begin, end, name))
    spans.sort()
    for (_b1, e1, n1), (b2, _e2, n2) in itertools.pairwise(spans):
        if b2 < e1:
            ctx.emit("GG410", loc, f"tensors {n1!r} and {n2!r} overlap", tensors=[n1, n2])


def scan_gguf(buf: Buffer, loc: Location, ctx: ScanContext) -> GGUFSummary | None:
    """Validate a GGUF file; returns a summary of what could be parsed."""
    size = len(buf)
    if size < 8 or buf[:4] != MAGIC:
        ctx.emit("GG400", loc, "missing GGUF magic or file shorter than the header")
        return None
    endian, version = _byte_order(buf)
    if version not in (1, 2, 3):
        ctx.emit("GG401", loc, f"unsupported GGUF version {version}", version=version)
        return None
    reader = _Reader(buf, endian, version, ctx, loc)
    reader.pos = 8
    summary: GGUFSummary | None = None
    try:
        tensor_count = reader.length()
        kv_count = reader.length()
        summary = GGUFSummary(version, endian == ">", tensor_count, kv_count)
        min_kv, min_tensor = (13, 24) if reader.wide else (9, 16)
        if kv_count * min_kv + tensor_count * min_tensor > reader.remaining():
            raise _TruncatedError(
                "GG402",
                8,
                f"header declares {tensor_count} tensors and {kv_count} metadata entries, "
                f"which cannot fit in {size} bytes",
            )
        seen: set[str] = set()
        for _ in range(kv_count):
            start = reader.pos
            key = reader.string("metadata key")
            vtype = int(reader.scalar("I"))
            value = reader.value(vtype, 0, key)
            if key in seen:
                ctx.emit("GG405", loc.at(start), f"metadata key {key!r} repeats", key=key)
            seen.add(key)
            if key == "tokenizer.chat_template" or key.startswith("tokenizer.chat_template."):
                _check_template(key, value, loc.at(start), ctx)
            if isinstance(value, str) and len(value) > 256:
                value = value[:256] + "..."
            summary.metadata[key] = value
        alignment = summary.metadata.get("general.alignment", DEFAULT_ALIGNMENT)
        if not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
            ctx.emit("GG409", loc, f"general.alignment {alignment!r} is not a power of two")
            alignment = DEFAULT_ALIGNMENT
        summary.alignment = alignment
        _read_tensor_infos(reader, summary, loc, ctx)
    except _TruncatedError as exc:
        ctx.emit(exc.rule, loc.at(exc.offset), exc.message)
        return summary
    summary.data_offset = -(-reader.pos // summary.alignment) * summary.alignment
    if summary.tensors and summary.data_offset > size:
        ctx.emit("GG408", loc, "tensor data section starts beyond the end of the file")
        return summary
    _check_tensors(summary, size, loc, ctx)
    return summary
