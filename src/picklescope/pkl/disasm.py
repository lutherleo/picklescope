"""Static pickle disassembler.

``iter_ops`` walks one pickle stream opcode by opcode, decoding arguments with the
readers from :mod:`picklescope.pkl.opcodes`. It stops after ``STOP`` and raises
:class:`~picklescope.pkl.opcodes.PickleParseError` on truncated input or unknown
opcodes. ``format_ops`` renders a listing similar to ``pickletools.dis``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from picklescope.blob import Buffer
from picklescope.pkl.opcodes import OPCODES, READERS, ArgValue, OpcodeInfo, PickleParseError


@dataclass(frozen=True, slots=True)
class Op:
    """One decoded opcode."""

    offset: int
    info: OpcodeInfo
    arg: ArgValue
    end: int

    @property
    def name(self) -> str:
        return self.info.name


def iter_ops(buf: Buffer, start: int = 0) -> Iterator[Op]:
    """Yield the opcodes of the pickle beginning at ``start``, through ``STOP``.

    Raises ``PickleParseError`` with ``kind`` one of ``truncated``, ``unknown_opcode``
    or ``bad_argument``. Already-yielded opcodes remain valid.
    """
    pos = start
    size = len(buf)
    while True:
        if pos >= size:
            raise PickleParseError("truncated", pos, "pickle ended without a STOP opcode")
        code = buf[pos]
        info = OPCODES.get(code)
        if info is None:
            raise PickleParseError("unknown_opcode", pos, f"unknown opcode byte 0x{code:02x}")
        arg, end = READERS[info.arg](buf, pos + 1, pos)
        yield Op(pos, info, arg, end)
        if code == 0x2E:  # STOP
            return
        pos = end


def looks_like_pickle(buf: Buffer, start: int = 0) -> bool:
    """Cheap test for "a pickle starts here" used for nested-payload detection.

    Only protocol 2+ prologues (``PROTO n``) are recognised; text protocols are too
    ambiguous to sniff inside arbitrary bytes.
    """
    return (
        len(buf) - start >= 3
        and buf[start] == 0x80
        and 2 <= buf[start + 1] <= 5
        and buf[start + 2] in OPCODES
    )


def _render_arg(arg: ArgValue, width: int = 60) -> str:
    if arg is None:
        return ""
    text = repr(arg)
    if len(text) > width:
        text = text[: width - 3] + "..."
    return text


def format_ops(ops: list[Op], annotations: dict[int, str] | None = None) -> str:
    """Render opcodes as ``offset: NAME arg  # note`` lines."""
    notes = annotations or {}
    lines = []
    for op in ops:
        line = f"{op.offset:8d}: {op.name:<16} {_render_arg(op.arg)}".rstrip()
        note = notes.get(op.offset)
        if note:
            line = f"{line:<90} # {note}"
        lines.append(line)
    return "\n".join(lines)
