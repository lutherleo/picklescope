"""safetensors header validation.

Layout: ``u64 little-endian N`` | ``N bytes of UTF-8 JSON`` | ``data buffer``. The JSON
maps tensor names to ``{"dtype", "shape", "data_offsets": [begin, end]}`` (offsets
relative to the data buffer) plus an optional ``"__metadata__"`` string map.

The checks mirror the reference implementation's ``Metadata::validate`` (header size
cap, ``{`` start, exact tiling of the data buffer, size = elements x dtype width) and
add parser-differential checks it does not perform (duplicate JSON keys). The format
cannot carry code; findings here indicate tampering, truncation or polyglot files.
"""

from __future__ import annotations

import json
import math
import struct
from typing import Any

from picklescope.blob import Buffer
from picklescope.context import ScanContext
from picklescope.findings import Location

DTYPE_BITS: dict[str, int] = {
    "BOOL": 8, "U8": 8, "I8": 8, "F8_E5M2": 8, "F8_E4M3": 8, "F8_E8M0": 8,
    "I16": 16, "U16": 16, "F16": 16, "BF16": 16,
    "I32": 32, "U32": 32, "F32": 32,
    "I64": 64, "U64": 64, "F64": 64, "C64": 64,
    "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6,
}  # fmt: skip
METADATA_KEY = "__metadata__"


class _DuplicateKeys(dict[str, Any]):
    """JSON object hook result that remembers duplicate keys."""

    duplicates: list[str]


def _pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = _DuplicateKeys()
    result.duplicates = []
    for key, value in pairs:
        if key in result:
            result.duplicates.append(key)
        result[key] = value
    return result


def _collect_duplicates(obj: Any) -> list[str]:
    found: list[str] = []
    stack = [obj]
    while stack:
        node = stack.pop()
        if isinstance(node, _DuplicateKeys):
            found.extend(node.duplicates)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return found


def _is_uint(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _check_tensor(name: str, info: Any, loc: Location, ctx: ScanContext) -> tuple[int, int] | None:
    """Validate one tensor entry; return its (begin, end) range when usable."""
    if not isinstance(info, dict):
        ctx.emit("ST303", loc, f"tensor {name!r}: entry is not an object")
        return None
    dtype = info.get("dtype")
    shape = info.get("shape")
    offsets = info.get("data_offsets")
    extra = set(info) - {"dtype", "shape", "data_offsets"}
    if extra:
        ctx.emit("ST301", loc, f"tensor {name!r}: unexpected key(s) {sorted(extra)}")
    bits = DTYPE_BITS.get(dtype) if isinstance(dtype, str) else None
    if bits is None:
        ctx.emit("ST302", loc, f"tensor {name!r}: unknown dtype {dtype!r}", tensor=name)
    if not isinstance(shape, list) or not all(_is_uint(d) for d in shape):
        ctx.emit("ST303", loc, f"tensor {name!r}: invalid shape {shape!r}", tensor=name)
        shape = None
    if not isinstance(offsets, list) or len(offsets) != 2 or not all(_is_uint(o) for o in offsets):
        ctx.emit("ST304", loc, f"tensor {name!r}: invalid data_offsets {offsets!r}", tensor=name)
        return None
    begin, end = offsets
    if end < begin:
        ctx.emit(
            "ST304", loc, f"tensor {name!r}: data_offsets end {end} < begin {begin}", tensor=name
        )
        return None
    if bits is not None and shape is not None:
        elements = math.prod(shape)
        total_bits = elements * bits
        expected = total_bits // 8
        if total_bits % 8 or expected != end - begin:
            ctx.emit(
                "ST304",
                loc,
                f"tensor {name!r}: {dtype}{shape} needs {total_bits / 8:g} bytes but its "
                f"range holds {end - begin}",
                tensor=name,
            )
    return begin, end


def _check_tiling(
    ranges: list[tuple[int, int, str]], data_len: int, loc: Location, ctx: ScanContext
) -> None:
    ranges.sort()
    cursor = 0
    for begin, end, name in ranges:
        if begin < cursor:
            ctx.emit(
                "ST305",
                loc,
                f"tensor {name!r} [{begin}, {end}) overlaps the previous tensor (ends at {cursor})",
                tensor=name,
            )
        elif begin > cursor:
            ctx.emit(
                "ST305",
                loc,
                f"{begin - cursor} unreferenced byte(s) before tensor {name!r}",
                tensor=name,
            )
        cursor = max(cursor, end)
    if cursor != data_len:
        what = "beyond the end of" if cursor > data_len else "short of"
        ctx.emit(
            "ST308",
            loc,
            f"tensors end at byte {cursor} of the data section, {what} its {data_len} bytes",
            tensors_end=cursor,
            data_length=data_len,
        )


def scan_safetensors(buf: Buffer, loc: Location, ctx: ScanContext) -> dict[str, Any] | None:
    """Validate a safetensors file. Returns the parsed header when it is valid JSON."""
    size = len(buf)
    if size < 8:
        ctx.emit("ST300", loc, f"file is {size} bytes, too short for the length prefix")
        return None
    (header_len,) = struct.unpack_from("<Q", buf, 0)
    if header_len == 0 or header_len > ctx.limits.max_header_bytes:
        ctx.emit(
            "ST300",
            loc,
            f"header length {header_len} is outside (0, {ctx.limits.max_header_bytes}]",
            header_length=header_len,
        )
        return None
    if 8 + header_len > size:
        ctx.emit(
            "ST300",
            loc,
            f"header length {header_len} runs past the end of the {size}-byte file",
            header_length=header_len,
        )
        return None
    raw = buf[8 : 8 + header_len]
    if raw[:1] != b"{":
        ctx.emit("ST301", loc, "header does not start with '{'")
        return None
    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs_hook)
    except UnicodeDecodeError as exc:
        ctx.emit("ST301", loc, f"header is not UTF-8: {exc}")
        return None
    except (json.JSONDecodeError, RecursionError) as exc:
        ctx.emit("ST301", loc, f"header is not valid JSON: {exc}")
        return None
    if not isinstance(header, dict):
        ctx.emit("ST301", loc, "header JSON is not an object")
        return None
    for key in sorted(set(_collect_duplicates(header))):
        ctx.emit("ST306", loc, f"header key {key!r} appears more than once", key=key)

    metadata = header.get(METADATA_KEY)
    if metadata is not None and not (
        isinstance(metadata, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())
    ):
        ctx.emit("ST307", loc, "__metadata__ must map strings to strings")

    ranges: list[tuple[int, int, str]] = []
    for name, info in header.items():
        if name == METADATA_KEY:
            continue
        span = _check_tensor(name, info, loc, ctx)
        if span is not None:
            ranges.append((span[0], span[1], name))
    _check_tiling(ranges, size - 8 - header_len, loc, ctx)
    return dict(header)


def looks_like_safetensors(buf: Buffer) -> bool:
    """Sniff: plausible length prefix followed by ``{``."""
    if len(buf) < 10:
        return False
    (header_len,) = struct.unpack_from("<Q", buf, 0)
    return 2 <= header_len <= len(buf) - 8 and buf[8:9] == b"{"
