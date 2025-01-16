"""Minimal protobuf wire-format decoder (no generated code, no dependencies).

Only what the ONNX scanner needs: iterate the fields of a message as
``(field_number, wire_type, value)`` where length-delimited values are returned as
``(start, end)`` byte ranges into the original buffer, so nested messages are decoded
lazily without copying.
"""

from __future__ import annotations

from collections.abc import Iterator

from picklescope.blob import Buffer

VARINT, FIXED64, LENGTH_DELIMITED, START_GROUP, END_GROUP, FIXED32 = 0, 1, 2, 3, 4, 5


class ProtobufError(ValueError):
    def __init__(self, offset: int, message: str) -> None:
        super().__init__(f"{message} (at byte {offset})")
        self.offset = offset


def read_varint(buf: Buffer, pos: int, end: int) -> tuple[int, int]:
    """Decode a base-128 varint at ``pos``; returns (value, next position)."""
    result = 0
    shift = 0
    start = pos
    while True:
        if pos >= end:
            raise ProtobufError(start, "truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift >= 70:
            raise ProtobufError(start, "varint longer than 10 bytes")


Field = tuple[int, int, int | tuple[int, int]]


def iter_fields(buf: Buffer, start: int, end: int) -> Iterator[Field]:
    """Yield the fields of the message occupying ``buf[start:end]``.

    Varints and fixed-width values are returned as ints; length-delimited values as a
    ``(start, end)`` range. Groups (deprecated) are rejected.
    """
    pos = start
    while pos < end:
        key, pos = read_varint(buf, pos, end)
        number, wire = key >> 3, key & 7
        if number == 0:
            raise ProtobufError(pos, "field number 0 is invalid")
        if wire == VARINT:
            value, pos = read_varint(buf, pos, end)
            yield number, wire, value
        elif wire == FIXED64:
            if pos + 8 > end:
                raise ProtobufError(pos, "truncated fixed64")
            yield number, wire, int.from_bytes(buf[pos : pos + 8], "little")
            pos += 8
        elif wire == FIXED32:
            if pos + 4 > end:
                raise ProtobufError(pos, "truncated fixed32")
            yield number, wire, int.from_bytes(buf[pos : pos + 4], "little")
            pos += 4
        elif wire == LENGTH_DELIMITED:
            length, pos = read_varint(buf, pos, end)
            if pos + length > end:
                raise ProtobufError(pos, f"length {length} runs past the enclosing message")
            yield number, wire, (pos, pos + length)
            pos += length
        else:
            raise ProtobufError(pos, f"unsupported wire type {wire}")


def text(buf: Buffer, span: tuple[int, int]) -> str:
    return bytes(buf[span[0] : span[1]]).decode("utf-8", "replace")


def to_signed64(value: int) -> int:
    """Interpret an unsigned varint as a two's-complement int64."""
    return value - (1 << 64) if value >= 1 << 63 else value
