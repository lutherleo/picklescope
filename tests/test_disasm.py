"""Opcode table and disassembler, differentially tested against pickletools."""

from __future__ import annotations

import math
import pickle
import pickletools
from collections import OrderedDict
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from picklescope.pkl.disasm import format_ops, iter_ops, looks_like_pickle
from picklescope.pkl.opcodes import OPCODES, OPCODES_BY_NAME, ArgKind, PickleParseError


def test_opcode_table_matches_pickletools() -> None:
    ours = {(info.code, info.name, info.arg.value, info.proto) for info in OPCODES.values()}
    ref = {
        (ord(o.code) if isinstance(o.code, str) else o.code[0], o.name,
         o.arg.name if o.arg is not None else ArgKind.NONE.value, o.proto)
        for o in pickletools.opcodes
    }  # fmt: skip
    assert ours == ref
    assert len(OPCODES_BY_NAME) == len(pickletools.opcodes) == 68


def _norm(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def _compare(data: bytes) -> None:
    ours = [(op.offset, op.name, _norm(op.arg)) for op in iter_ops(data)]
    ref = [(pos, op.name, _norm(arg)) for op, arg, pos in pickletools.genops(data)]
    assert ours == ref


scalars = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=True)
    | st.text(max_size=40)
    | st.binary(max_size=300)
)
values = st.recursive(
    scalars,
    lambda children: (
        st.lists(children, max_size=5)
        | st.tuples(children, children)
        | st.tuples(children)
        | st.dictionaries(st.text(max_size=8), children, max_size=4)
        | st.frozensets(st.integers(), max_size=4)
        | st.sets(st.text(max_size=4), max_size=4)
    ),
    max_leaves=25,
)


@settings(max_examples=150, deadline=None)
@given(value=values, protocol=st.integers(min_value=0, max_value=5))
def test_disassembly_matches_pickletools(value: Any, protocol: int) -> None:
    # pickle.dumps only *serialises* plain data here; nothing is ever loaded.
    _compare(pickle.dumps(value, protocol=protocol))


@pytest.mark.parametrize("protocol", range(6))
def test_large_values_and_containers(protocol: int) -> None:
    value = {
        "big": 2**200, "neg": -(2**70), "long_text": "x" * 300, "bytes": b"\x00" * 70000,
        "od": OrderedDict(a=1), "nested": [(1, 2, 3), {"k": frozenset({1})}],
        "bytearray": bytearray(b"abc"), "float": 1.5,
    }  # fmt: skip
    _compare(pickle.dumps(value, protocol=protocol))


@pytest.mark.parametrize(
    ("data", "kind"),
    [
        (b"", "truncated"),
        (b"\x80\x04", "truncated"),
        (b"\x80\x04\xff", "unknown_opcode"),
        (b"X\x05\x00\x00\x00ab", "truncated"),
        (b"T\xff\xff\xff\xff", "bad_argument"),
        (b"S'abc\n.", "bad_argument"),
        (b"Sabc\n.", "bad_argument"),
        (b"S'\\x'\n.", "bad_argument"),
        (b"Iabc\n.", "bad_argument"),
        (b"Fnope\n.", "bad_argument"),
        (b"cmodule-without-newline", "truncated"),
        (b"\x8c\x02\xff\xfe.", "bad_argument"),
        (b"V\\u12\n.", "bad_argument"),
    ],
)
def test_parse_errors_are_structured(data: bytes, kind: str) -> None:
    with pytest.raises(PickleParseError) as info:
        list(iter_ops(data))
    assert info.value.kind == kind
    assert info.value.offset >= 0


def test_text_protocol_arguments() -> None:
    data = b"I01\nI00\nI-7\nL12L\nF1.5\nS'a\\nb'\nVsnow\\u2603\nP pid\ncmod\nname\n."
    args = [op.arg for op in iter_ops(data)]
    assert args == [True, False, -7, 12, 1.5, "a\nb", "snow\u2603", " pid", "mod name", None]


def test_looks_like_pickle() -> None:
    assert looks_like_pickle(b"\x80\x04\x95")
    assert not looks_like_pickle(b"\x80\x06N.")
    assert not looks_like_pickle(b"\x80\x04")
    assert not looks_like_pickle(b"PK\x03\x04")


def test_format_ops_renders_annotations() -> None:
    data = pickle.dumps(["x" * 100], protocol=4)
    ops = list(iter_ops(data))
    text = format_ops(ops, {ops[0].offset: "hello"})
    lines = text.splitlines()
    assert lines[0].rstrip().endswith("# hello")
    assert "..." in text  # long argument truncated
    assert len(lines) == len(ops)
