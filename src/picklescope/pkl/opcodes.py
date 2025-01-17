"""Pickle opcode table (protocols 0-5) and argument decoders.

This is an independent implementation of the opcode catalogue documented in CPython's
``pickletools``. Argument decoders operate on an in-memory buffer and never call into
the unpickler: decoding a GLOBAL argument yields the *names*, it never imports them.
The decoded values intentionally mirror ``pickletools.genops`` so the two can be
differentially tested.
"""

from __future__ import annotations

import codecs
import enum
import struct
from collections.abc import Callable
from dataclasses import dataclass

from picklescope.blob import Buffer


class PickleParseError(Exception):
    """Raised when the byte stream cannot be decoded at ``offset``."""

    def __init__(self, kind: str, offset: int, message: str) -> None:
        super().__init__(f"{message} (at byte {offset})")
        self.kind = kind
        self.offset = offset
        self.message = message


class ArgKind(enum.Enum):
    NONE = "none"
    UINT1 = "uint1"
    UINT2 = "uint2"
    UINT4 = "uint4"
    UINT8 = "uint8"
    INT4 = "int4"
    LONG1 = "long1"
    LONG4 = "long4"
    STRING1 = "string1"
    STRING4 = "string4"
    BYTES1 = "bytes1"
    BYTES4 = "bytes4"
    BYTES8 = "bytes8"
    BYTEARRAY8 = "bytearray8"
    UNICODE1 = "unicodestring1"
    UNICODE4 = "unicodestring4"
    UNICODE8 = "unicodestring8"
    UNICODENL = "unicodestringnl"
    STRINGNL = "stringnl"
    STRINGNL_NOESCAPE = "stringnl_noescape"
    STRINGNL_NOESCAPE_PAIR = "stringnl_noescape_pair"
    DECIMALNL_SHORT = "decimalnl_short"
    DECIMALNL_LONG = "decimalnl_long"
    FLOATNL = "floatnl"
    FLOAT8 = "float8"


@dataclass(frozen=True, slots=True)
class OpcodeInfo:
    code: int
    name: str
    arg: ArgKind
    proto: int


def _table() -> dict[int, OpcodeInfo]:
    a = ArgKind
    rows: list[tuple[bytes, str, ArgKind, int]] = [
        (b"I", "INT", a.DECIMALNL_SHORT, 0),
        (b"J", "BININT", a.INT4, 1),
        (b"K", "BININT1", a.UINT1, 1),
        (b"M", "BININT2", a.UINT2, 1),
        (b"L", "LONG", a.DECIMALNL_LONG, 0),
        (b"\x8a", "LONG1", a.LONG1, 2),
        (b"\x8b", "LONG4", a.LONG4, 2),
        (b"S", "STRING", a.STRINGNL, 0),
        (b"T", "BINSTRING", a.STRING4, 1),
        (b"U", "SHORT_BINSTRING", a.STRING1, 1),
        (b"B", "BINBYTES", a.BYTES4, 3),
        (b"C", "SHORT_BINBYTES", a.BYTES1, 3),
        (b"\x8e", "BINBYTES8", a.BYTES8, 4),
        (b"\x96", "BYTEARRAY8", a.BYTEARRAY8, 5),
        (b"\x97", "NEXT_BUFFER", a.NONE, 5),
        (b"\x98", "READONLY_BUFFER", a.NONE, 5),
        (b"N", "NONE", a.NONE, 0),
        (b"\x88", "NEWTRUE", a.NONE, 2),
        (b"\x89", "NEWFALSE", a.NONE, 2),
        (b"V", "UNICODE", a.UNICODENL, 0),
        (b"\x8c", "SHORT_BINUNICODE", a.UNICODE1, 4),
        (b"X", "BINUNICODE", a.UNICODE4, 1),
        (b"\x8d", "BINUNICODE8", a.UNICODE8, 4),
        (b"F", "FLOAT", a.FLOATNL, 0),
        (b"G", "BINFLOAT", a.FLOAT8, 1),
        (b"]", "EMPTY_LIST", a.NONE, 1),
        (b"a", "APPEND", a.NONE, 0),
        (b"e", "APPENDS", a.NONE, 1),
        (b"l", "LIST", a.NONE, 0),
        (b")", "EMPTY_TUPLE", a.NONE, 1),
        (b"t", "TUPLE", a.NONE, 0),
        (b"\x85", "TUPLE1", a.NONE, 2),
        (b"\x86", "TUPLE2", a.NONE, 2),
        (b"\x87", "TUPLE3", a.NONE, 2),
        (b"}", "EMPTY_DICT", a.NONE, 1),
        (b"d", "DICT", a.NONE, 0),
        (b"s", "SETITEM", a.NONE, 0),
        (b"u", "SETITEMS", a.NONE, 1),
        (b"\x8f", "EMPTY_SET", a.NONE, 4),
        (b"\x90", "ADDITEMS", a.NONE, 4),
        (b"\x91", "FROZENSET", a.NONE, 4),
        (b"0", "POP", a.NONE, 0),
        (b"2", "DUP", a.NONE, 0),
        (b"(", "MARK", a.NONE, 0),
        (b"1", "POP_MARK", a.NONE, 1),
        (b"g", "GET", a.DECIMALNL_SHORT, 0),
        (b"h", "BINGET", a.UINT1, 1),
        (b"j", "LONG_BINGET", a.UINT4, 1),
        (b"p", "PUT", a.DECIMALNL_SHORT, 0),
        (b"q", "BINPUT", a.UINT1, 1),
        (b"r", "LONG_BINPUT", a.UINT4, 1),
        (b"\x94", "MEMOIZE", a.NONE, 4),
        (b"\x82", "EXT1", a.UINT1, 2),
        (b"\x83", "EXT2", a.UINT2, 2),
        (b"\x84", "EXT4", a.INT4, 2),
        (b"c", "GLOBAL", a.STRINGNL_NOESCAPE_PAIR, 0),
        (b"\x93", "STACK_GLOBAL", a.NONE, 4),
        (b"R", "REDUCE", a.NONE, 0),
        (b"b", "BUILD", a.NONE, 0),
        (b"i", "INST", a.STRINGNL_NOESCAPE_PAIR, 0),
        (b"o", "OBJ", a.NONE, 1),
        (b"\x81", "NEWOBJ", a.NONE, 2),
        (b"\x92", "NEWOBJ_EX", a.NONE, 4),
        (b"\x80", "PROTO", a.UINT1, 2),
        (b".", "STOP", a.NONE, 0),
        (b"\x95", "FRAME", a.UINT8, 4),
        (b"P", "PERSID", a.STRINGNL_NOESCAPE, 0),
        (b"Q", "BINPERSID", a.NONE, 1),
    ]
    return {code[0]: OpcodeInfo(code[0], name, arg, proto) for code, name, arg, proto in rows}


OPCODES: dict[int, OpcodeInfo] = _table()
OPCODES_BY_NAME: dict[str, OpcodeInfo] = {info.name: info for info in OPCODES.values()}
HIGHEST_PROTOCOL = 5

ArgValue = int | float | bool | str | bytes | bytearray | None
Reader = Callable[[Buffer, int, int], tuple[ArgValue, int]]
"""A reader takes (buffer, position of the argument, opcode offset) and returns
(decoded value, position after the argument)."""


def _need(buf: Buffer, pos: int, n: int, op_offset: int, what: str) -> None:
    if n < 0:
        raise PickleParseError("bad_argument", op_offset, f"negative length {n} for {what}")
    if pos + n > len(buf):
        raise PickleParseError(
            "truncated",
            op_offset,
            f"{what} needs {n} bytes but only {max(0, len(buf) - pos)} remain",
        )


def _fixed(fmt: str, size: int, what: str) -> Reader:
    unpack = struct.Struct(fmt).unpack_from

    def read(buf: Buffer, pos: int, op_offset: int) -> tuple[ArgValue, int]:
        _need(buf, pos, size, op_offset, what)
        return unpack(buf, pos)[0], pos + size

    return read


_read_uint1 = _fixed("<B", 1, "uint1")
_read_uint2 = _fixed("<H", 2, "uint2")
_read_uint4 = _fixed("<I", 4, "uint4")
_read_uint8 = _fixed("<Q", 8, "uint8")
_read_int4 = _fixed("<i", 4, "int4")
_read_float8 = _fixed(">d", 8, "float8")


def _read_sized(
    width_reader: Reader, what: str, convert: Callable[[bytes, int], ArgValue]
) -> Reader:
    def read(buf: Buffer, pos: int, op_offset: int) -> tuple[ArgValue, int]:
        n, pos = width_reader(buf, pos, op_offset)
        assert isinstance(n, int)
        _need(buf, pos, n, op_offset, what)
        return convert(buf[pos : pos + n], op_offset), pos + n

    return read


def _latin1(data: bytes, _offset: int) -> ArgValue:
    return data.decode("latin-1")


def _raw(data: bytes, _offset: int) -> ArgValue:
    return bytes(data)


def _bytearray(data: bytes, _offset: int) -> ArgValue:
    return bytearray(data)


def _utf8(data: bytes, offset: int) -> ArgValue:
    try:
        return data.decode("utf-8", "surrogatepass")
    except UnicodeDecodeError as exc:
        raise PickleParseError("bad_argument", offset, f"invalid UTF-8 string: {exc}") from None


def _long(data: bytes, _offset: int) -> ArgValue:
    return int.from_bytes(data, "little", signed=True)


def _readline(buf: Buffer, pos: int, op_offset: int, what: str) -> tuple[bytes, int]:
    end = buf.find(b"\n", pos)
    if end < 0:
        raise PickleParseError("truncated", op_offset, f"no newline found when reading {what}")
    return buf[pos:end], end + 1


def _read_stringnl(buf: Buffer, pos: int, op_offset: int) -> tuple[ArgValue, int]:
    data, pos = _readline(buf, pos, op_offset, "stringnl")
    for quote in (b'"', b"'"):
        if data.startswith(quote):
            if len(data) < 2 or not data.endswith(quote):
                raise PickleParseError("bad_argument", op_offset, "unbalanced string quotes")
            data = data[1:-1]
            break
    else:
        raise PickleParseError("bad_argument", op_offset, "STRING argument is not quoted")
    try:
        decoded: bytes = codecs.escape_decode(data)[0]
    except ValueError as exc:
        raise PickleParseError("bad_argument", op_offset, f"bad string escape: {exc}") from None
    return decoded.decode("latin-1"), pos


def _read_noescape(buf: Buffer, pos: int, op_offset: int) -> tuple[ArgValue, int]:
    data, pos = _readline(buf, pos, op_offset, "stringnl_noescape")
    return _utf8(data, op_offset), pos


def _read_noescape_pair(buf: Buffer, pos: int, op_offset: int) -> tuple[ArgValue, int]:
    first, pos = _read_noescape(buf, pos, op_offset)
    second, pos = _read_noescape(buf, pos, op_offset)
    assert isinstance(first, str)
    assert isinstance(second, str)
    return f"{first} {second}", pos


def _read_unicodenl(buf: Buffer, pos: int, op_offset: int) -> tuple[ArgValue, int]:
    data, pos = _readline(buf, pos, op_offset, "unicodestringnl")
    try:
        return data.decode("raw-unicode-escape"), pos
    except UnicodeDecodeError as exc:
        raise PickleParseError("bad_argument", op_offset, f"bad unicode escape: {exc}") from None


def _parse_int(text: bytes, op_offset: int) -> int:
    try:
        return int(text)
    except ValueError:
        raise PickleParseError(
            "bad_argument", op_offset, f"invalid integer {text[:40]!r}"
        ) from None


def _read_decimal_short(buf: Buffer, pos: int, op_offset: int) -> tuple[ArgValue, int]:
    data, pos = _readline(buf, pos, op_offset, "decimalnl_short")
    if data == b"00":
        return False, pos
    if data == b"01":
        return True, pos
    return _parse_int(data, op_offset), pos


def _read_decimal_long(buf: Buffer, pos: int, op_offset: int) -> tuple[ArgValue, int]:
    data, pos = _readline(buf, pos, op_offset, "decimalnl_long")
    if data.endswith(b"L"):
        data = data[:-1]
    return _parse_int(data, op_offset), pos


def _read_floatnl(buf: Buffer, pos: int, op_offset: int) -> tuple[ArgValue, int]:
    data, pos = _readline(buf, pos, op_offset, "floatnl")
    try:
        return float(data), pos
    except ValueError:
        raise PickleParseError("bad_argument", op_offset, f"invalid float {data[:40]!r}") from None


def _read_none(_buf: Buffer, pos: int, _op_offset: int) -> tuple[ArgValue, int]:
    return None, pos


READERS: dict[ArgKind, Reader] = {
    ArgKind.NONE: _read_none,
    ArgKind.UINT1: _read_uint1,
    ArgKind.UINT2: _read_uint2,
    ArgKind.UINT4: _read_uint4,
    ArgKind.UINT8: _read_uint8,
    ArgKind.INT4: _read_int4,
    ArgKind.LONG1: _read_sized(_read_uint1, "long1", _long),
    ArgKind.LONG4: _read_sized(_read_int4, "long4", _long),
    ArgKind.STRING1: _read_sized(_read_uint1, "string1", _latin1),
    ArgKind.STRING4: _read_sized(_read_int4, "string4", _latin1),
    ArgKind.BYTES1: _read_sized(_read_uint1, "bytes1", _raw),
    ArgKind.BYTES4: _read_sized(_read_uint4, "bytes4", _raw),
    ArgKind.BYTES8: _read_sized(_read_uint8, "bytes8", _raw),
    ArgKind.BYTEARRAY8: _read_sized(_read_uint8, "bytearray8", _bytearray),
    ArgKind.UNICODE1: _read_sized(_read_uint1, "unicodestring1", _utf8),
    ArgKind.UNICODE4: _read_sized(_read_uint4, "unicodestring4", _utf8),
    ArgKind.UNICODE8: _read_sized(_read_uint8, "unicodestring8", _utf8),
    ArgKind.UNICODENL: _read_unicodenl,
    ArgKind.STRINGNL: _read_stringnl,
    ArgKind.STRINGNL_NOESCAPE: _read_noescape,
    ArgKind.STRINGNL_NOESCAPE_PAIR: _read_noescape_pair,
    ArgKind.DECIMALNL_SHORT: _read_decimal_short,
    ArgKind.DECIMALNL_LONG: _read_decimal_long,
    ArgKind.FLOATNL: _read_floatnl,
    ArgKind.FLOAT8: _read_float8,
}
