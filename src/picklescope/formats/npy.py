"""NumPy ``.npy`` files: header validation and object-array pickle extraction.

Layout: ``\\x93NUMPY`` | u8 major | u8 minor | header length (u16 for v1, u32 for v2/3)
| header (a Python dict literal) | array data. Arrays with an object dtype store their
data as a pickle, which ``numpy.load(..., allow_pickle=True)`` executes; that pickle is
scanned like any other.

The header is parsed with :func:`ast.literal_eval` (literals only, no code), exactly
as NumPy's own ``read_array_header`` does, after a length cap.
"""

from __future__ import annotations

import ast
import struct
from typing import Any

from picklescope.blob import Buffer
from picklescope.context import ScanContext
from picklescope.findings import Location
from picklescope.pkl.scan import scan_pickle

MAGIC = b"\x93NUMPY"
MAX_HEADER = 64 * 1024


def _descr_has_object(descr: Any) -> bool:
    if isinstance(descr, str):
        return "O" in descr.lstrip("<>|=")[:1]
    if isinstance(descr, list):  # structured dtype: [(name, dtype[, shape]), ...]
        return any(
            isinstance(field, tuple) and len(field) >= 2 and _descr_has_object(field[1])
            for field in descr
        )
    return False


def scan_npy(buf: Buffer, loc: Location, ctx: ScanContext) -> dict[str, Any] | None:
    """Validate a ``.npy`` header and scan object-array pickles. Returns the header."""
    if len(buf) < 10 or buf[:6] != MAGIC:
        ctx.emit("NP701", loc, "missing \\x93NUMPY magic")
        return None
    major = buf[6]
    if major == 1:
        (header_len,) = struct.unpack_from("<H", buf, 8)
        start = 10
    elif major in (2, 3):
        if len(buf) < 12:
            ctx.emit("NP701", loc, "truncated v2/v3 header length")
            return None
        (header_len,) = struct.unpack_from("<I", buf, 8)
        start = 12
    else:
        ctx.emit("NP701", loc, f"unsupported .npy format version {major}")
        return None
    if header_len > MAX_HEADER or start + header_len > len(buf):
        ctx.emit("NP701", loc, f"header length {header_len} is too large or truncated")
        return None
    raw = bytes(buf[start : start + header_len])
    try:
        header = ast.literal_eval(raw.decode("utf-8" if major == 3 else "latin-1"))
    except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError) as exc:
        ctx.emit("NP701", loc, f"header is not a Python literal: {exc}")
        return None
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        ctx.emit("NP701", loc, "header must be a dict with exactly descr, fortran_order, shape")
        return None
    shape = header["shape"]
    if not isinstance(shape, tuple) or not all(isinstance(d, int) and d >= 0 for d in shape):
        ctx.emit("NP701", loc, f"invalid shape {shape!r}")
    if _descr_has_object(header["descr"]):
        data_start = start + header_len
        ctx.emit(
            "NP700",
            loc,
            f"object dtype {header['descr']!r}: array data is a pickle "
            "(numpy.load requires allow_pickle=True)",
            descr=str(header["descr"]),
        )
        payload = bytes(buf[data_start:])
        scan_pickle(payload, loc.child(f"pickle@{data_start}"), ctx, "npy")
    return header
