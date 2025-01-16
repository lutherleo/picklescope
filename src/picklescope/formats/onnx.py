"""ONNX model checks: external-data paths, custom operator domains, nesting depth.

The model is decoded directly from the protobuf wire format (field numbers from
``onnx/onnx.proto3``); the ``onnx`` package is not required. Every tensor reachable
from the model is visited: graph initializers, sparse initializers, tensor-valued node
attributes (``Constant``), and tensors inside sub-graphs (``If``/``Loop``/``Scan``),
local functions and training-info graphs.

External data is the risky part of ONNX: a tensor's ``location`` is a path the runtime
opens relative to the model directory. The scanner never opens such a path; for files
on disk it only resolves it lexically and with ``realpath`` to decide containment, and
calls ``stat`` only on paths already proven to lie inside the model directory.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from picklescope.blob import Buffer
from picklescope.context import ScanContext
from picklescope.findings import Location
from picklescope.formats.protobuf import (
    LENGTH_DELIMITED,
    VARINT,
    ProtobufError,
    iter_fields,
    read_varint,
    text,
    to_signed64,
)

log = logging.getLogger(__name__)

STANDARD_DOMAINS = frozenset(
    {
        "",
        "ai.onnx",
        "ai.onnx.ml",
        "ai.onnx.training",
        "ai.onnx.preview.training",
        "com.microsoft",
        "com.microsoft.nchwc",
        "com.microsoft.experimental",
        "com.ms.internal.nhwc",
    }
)

# TensorProto.DataType -> bits per element (STRING=8 is variable and omitted).
DATA_TYPE_BITS: dict[int, int] = {
    1: 32, 2: 8, 3: 8, 4: 16, 5: 16, 6: 32, 7: 64, 9: 8, 10: 16, 11: 64, 12: 32,
    13: 64, 14: 64, 15: 128, 16: 16, 17: 8, 18: 8, 19: 8, 20: 8, 21: 4, 22: 4, 23: 4,
    24: 8,
}  # fmt: skip

_DRIVE = re.compile(r"^[A-Za-z]:")

# Message kinds in the walk.
_MODEL, _GRAPH, _NODE, _ATTR, _TENSOR, _SPARSE, _FUNCTION, _TRAINING = range(8)


@dataclass(slots=True)
class ExternalRef:
    tensor: str
    location: str | None
    offset: str | None
    length: str | None
    dims: list[int]
    data_type: int
    at: int


@dataclass(slots=True)
class ONNXSummary:
    ir_version: int | None = None
    producer: str | None = None
    opsets: dict[str, int] = field(default_factory=dict)
    node_count: int = 0
    tensor_count: int = 0
    external: list[ExternalRef] = field(default_factory=list)
    custom_ops: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    function_domains: set[str] = field(default_factory=set)


def _parse_tensor(buf: Buffer, span: tuple[int, int], summary: ONNXSummary) -> None:
    name = ""
    dims: list[int] = []
    data_type = 0
    location_flag = 0
    entries: dict[str, str] = {}
    for number, wire, value in iter_fields(buf, *span):
        if number == 1 and wire == VARINT:
            assert isinstance(value, int)
            dims.append(to_signed64(value))
        elif number == 1 and wire == LENGTH_DELIMITED:  # packed dims
            assert isinstance(value, tuple)
            pos, end = value
            while pos < end:
                dim, pos = read_varint(buf, pos, end)
                dims.append(to_signed64(dim))
        elif number == 2 and wire == VARINT:
            assert isinstance(value, int)
            data_type = value
        elif number == 8 and wire == LENGTH_DELIMITED:
            assert isinstance(value, tuple)
            name = text(buf, value)
        elif number == 13 and wire == LENGTH_DELIMITED:
            assert isinstance(value, tuple)
            key = val = ""
            for n2, w2, v2 in iter_fields(buf, *value):
                if w2 == LENGTH_DELIMITED and isinstance(v2, tuple):
                    if n2 == 1:
                        key = text(buf, v2)
                    elif n2 == 2:
                        val = text(buf, v2)
            entries[key] = val
        elif number == 14 and wire == VARINT:
            assert isinstance(value, int)
            location_flag = value
    summary.tensor_count += 1
    if entries or location_flag == 1:
        summary.external.append(
            ExternalRef(
                tensor=name or "<unnamed>",
                location=entries.get("location"),
                offset=entries.get("offset"),
                length=entries.get("length"),
                dims=dims,
                data_type=data_type,
                at=span[0],
            )
        )


_CHILDREN: dict[int, dict[int, int]] = {
    # parent kind -> {field number: child kind} for length-delimited children
    _MODEL: {7: _GRAPH, 20: _TRAINING, 25: _FUNCTION},
    _TRAINING: {1: _GRAPH, 2: _GRAPH},
    _GRAPH: {1: _NODE, 5: _TENSOR, 15: _SPARSE},
    _FUNCTION: {7: _NODE},
    _NODE: {5: _ATTR},
    _ATTR: {5: _TENSOR, 6: _GRAPH, 10: _TENSOR, 11: _GRAPH, 22: _SPARSE, 23: _SPARSE},
    _SPARSE: {1: _TENSOR, 2: _TENSOR},
}


def _parse_model(buf: Buffer, loc: Location, ctx: ScanContext) -> ONNXSummary:
    summary = ONNXSummary()
    stack: list[tuple[int, tuple[int, int], int]] = [(_MODEL, (0, len(buf)), 0)]
    depth_reported = False
    while stack:
        kind, span, depth = stack.pop()
        if depth > ctx.limits.max_protobuf_depth:
            if not depth_reported:
                ctx.emit(
                    "OX506",
                    loc.at(span[0]),
                    f"message nesting exceeds "
                    f"{ctx.limits.max_protobuf_depth} levels; deeper content skipped",
                )
                depth_reported = True
            continue
        if kind == _TENSOR:
            _parse_tensor(buf, span, summary)
            continue
        op_type = domain = ""
        for number, wire, value in iter_fields(buf, *span):
            child = _CHILDREN.get(kind, {}).get(number)
            if child is not None and wire == LENGTH_DELIMITED:
                assert isinstance(value, tuple)
                stack.append((child, value, depth + 1))
            elif kind == _MODEL and number == 1 and wire == VARINT:
                assert isinstance(value, int)
                summary.ir_version = value
            elif kind == _MODEL and number == 2 and wire == LENGTH_DELIMITED:
                assert isinstance(value, tuple)
                summary.producer = text(buf, value)
            elif kind == _MODEL and number == 8 and wire == LENGTH_DELIMITED:
                assert isinstance(value, tuple)
                opset_domain, version = "", 0
                for n2, w2, v2 in iter_fields(buf, *value):
                    if n2 == 1 and w2 == LENGTH_DELIMITED and isinstance(v2, tuple):
                        opset_domain = text(buf, v2)
                    elif n2 == 2 and w2 == VARINT and isinstance(v2, int):
                        version = v2
                summary.opsets[opset_domain] = version
            elif kind == _NODE and wire == LENGTH_DELIMITED and number in (4, 7):
                assert isinstance(value, tuple)
                if number == 4:
                    op_type = text(buf, value)
                else:
                    domain = text(buf, value)
            elif kind == _FUNCTION and number == 10 and wire == LENGTH_DELIMITED:
                assert isinstance(value, tuple)
                summary.function_domains.add(text(buf, value))
        if kind == _NODE:
            summary.node_count += 1
            if domain not in STANDARD_DOMAINS:
                summary.custom_ops[domain].add(op_type)
    return summary


def unsafe_location(location: str) -> tuple[str, str] | None:
    """Lexical check of an external-data location: (rule id, reason) or None."""
    if "\x00" in location:
        return "OX502", "contains a NUL byte"
    if location.startswith(("/", "\\")) or _DRIVE.match(location):
        return "OX501", "is an absolute path"
    if location.startswith("~"):
        return "OX501", "starts with '~' (home-directory expansion)"
    if ".." in re.split(r"[\\/]", location):
        return "OX502", "contains a '..' segment"
    return None


def _expected_bytes(ref: ExternalRef) -> int | None:
    bits = DATA_TYPE_BITS.get(ref.data_type)
    if bits is None or any(d < 0 for d in ref.dims):
        return None
    total = math.prod(ref.dims) * bits
    return -(-total // 8)


def _parse_uint(text_value: str | None) -> int | None:
    if text_value is None:
        return None
    return int(text_value) if text_value.isdigit() else -1


def _check_external(
    ref: ExternalRef, base_dir: Path | None, loc: Location, ctx: ScanContext
) -> None:
    here = loc.at(ref.at)
    tensor = ref.tensor
    if not ref.location:
        ctx.emit("OX504", here, f"tensor {tensor!r} is external but has no location", tensor=tensor)
        return
    problem = unsafe_location(ref.location)
    if problem is not None:
        rule, reason = problem
        ctx.emit(
            rule,
            here,
            f"tensor {tensor!r} external location {ref.location!r} {reason}",
            tensor=tensor,
            external_location=ref.location,
        )
        return
    offset, length = _parse_uint(ref.offset), _parse_uint(ref.length)
    if offset == -1 or length == -1:
        ctx.emit(
            "OX504",
            here,
            f"tensor {tensor!r}: offset/length are not non-negative "
            f"integers ({ref.offset!r}, {ref.length!r})",
            tensor=tensor,
        )
        return
    expected = _expected_bytes(ref)
    if length is not None and expected is not None and length != expected:
        ctx.emit(
            "OX504",
            here,
            f"tensor {tensor!r}: length {length} differs from the {expected} bytes its shape needs",
            tensor=tensor,
        )
    if base_dir is None:
        return
    root = Path(base_dir).resolve()
    try:
        target = (root / ref.location).resolve()
    except (OSError, RuntimeError) as exc:  # RuntimeError: symlink loop (Python 3.12)
        ctx.emit(
            "OX502",
            here,
            f"tensor {tensor!r} external location {ref.location!r} cannot be resolved: {exc}",
            tensor=tensor,
            external_location=ref.location,
        )
        return
    if not target.is_relative_to(root):
        ctx.emit(
            "OX502",
            here,
            f"tensor {tensor!r} external location {ref.location!r} "
            "resolves outside the model directory (symlink)",
            tensor=tensor,
            external_location=ref.location,
        )
        return
    if not target.is_file():
        ctx.emit(
            "OX503",
            here,
            f"tensor {tensor!r} external file {ref.location!r} is missing",
            tensor=tensor,
            external_location=ref.location,
        )
        return
    file_size = target.stat().st_size
    need = (offset or 0) + (length if length is not None else (expected or 0))
    if need > file_size:
        ctx.emit(
            "OX504",
            here,
            f"tensor {tensor!r} reads bytes up to {need} of "
            f"{ref.location!r}, which has {file_size}",
            tensor=tensor,
        )


def scan_onnx(
    buf: Buffer, loc: Location, ctx: ScanContext, base_dir: Path | None = None
) -> ONNXSummary | None:
    """Scan an ONNX model. ``base_dir`` enables on-disk external-data checks."""
    try:
        summary = _parse_model(buf, loc, ctx)
    except ProtobufError as exc:
        ctx.emit("OX500", loc.at(exc.offset), f"cannot decode ONNX protobuf: {exc}")
        return None
    if summary.ir_version is None:
        ctx.emit("OX500", loc, "ModelProto has no ir_version; not a valid ONNX model")
    for ref in summary.external:
        _check_external(ref, base_dir, loc, ctx)
    for domain, ops in sorted(summary.custom_ops.items()):
        if domain in summary.function_domains:
            continue  # implemented by a model-local function, not native code
        ctx.emit(
            "OX505",
            loc,
            f"operators {sorted(ops)} come from custom domain {domain!r}",
            domain=domain,
            op_types=sorted(ops),
        )
    log.debug(
        "%s: onnx ir=%s nodes=%d tensors=%d external=%d",
        loc.display(),
        summary.ir_version,
        summary.node_count,
        summary.tensor_count,
        len(summary.external),
    )
    return summary


def looks_like_onnx(buf: Buffer) -> bool:
    """Sniff: first field is ir_version (field 1, varint) - weak, used with extensions."""
    return len(buf) >= 2 and buf[0] == 0x08
