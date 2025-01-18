"""Abstract interpreter semantics."""

from __future__ import annotations

import pickle

import pytest

from picklescope.limits import ScanLimits
from picklescope.pkl.analyzer import (
    BufferValue,
    CallValue,
    Const,
    ContainerValue,
    ExtValue,
    GlobalValue,
    PersistentValue,
    TupleValue,
    analyze_pickle,
    describe,
    describe_args,
)

from fixturegen import PickleAsm, print_pickle


def test_records_import_and_call_with_arguments() -> None:
    analysis = analyze_pickle(print_pickle("hi"))
    assert analysis.complete
    assert [(e.module, e.name, e.opcode) for e in analysis.imports] == [
        ("builtins", "print", "STACK_GLOBAL")
    ]
    (call,) = analysis.calls
    assert isinstance(call.callee, GlobalValue)
    assert describe(CallValue(call.callee, call.args, 0, "REDUCE")) == "builtins.print('hi')"
    assert analysis.protocol == 4


def test_benign_pickle_has_no_calls_to_unknowns() -> None:
    data = pickle.dumps({"a": [1, 2], "b": (b"x", None)}, protocol=5)
    analysis = analyze_pickle(data)
    assert analysis.complete and not analysis.imports and not analysis.calls
    assert analysis.end == len(data)


def test_memo_routes_stack_global_operands() -> None:
    data = (
        PickleAsm(4)
        .unicode("builtins")
        .memoize()
        .unicode("len")
        .memoize()
        .op(b"0")
        .op(b"0")
        .binget(0)
        .binget(1)
        .op(b"\x93")
        .stop()
    )
    analysis = analyze_pickle(data)
    assert [(e.module, e.name) for e in analysis.imports] == [("builtins", "len")]


def test_text_protocol_put_get() -> None:
    data = b"Vbuiltins\np0\n0Vlen\np1\n0g0\ng1\n\x93."
    analysis = analyze_pickle(data)
    assert [(e.module, e.name) for e in analysis.imports] == [("builtins", "len")]


def test_dynamic_stack_global() -> None:
    data = PickleAsm(4).unicode("builtins").int1(3).op(b"\x93").stop()
    analysis = analyze_pickle(data)
    assert not analysis.imports
    (event,) = analysis.dynamic_imports
    assert isinstance(event.name, Const) and event.name.value == 3


def test_inst_and_obj_calls() -> None:
    data = b"(I1\nicolorsys\nrgb_to_hsv\n(ctextwrap\nTextWrapper\no."
    analysis = analyze_pickle(data)
    assert [c.opcode for c in analysis.calls] == ["INST", "OBJ"]
    assert isinstance(analysis.calls[0].args, TupleValue)


def test_newobj_and_newobj_ex() -> None:
    data = (
        PickleAsm(4)
        .stack_global("textwrap", "TextWrapper")
        .empty_tuple()
        .op(b"\x81")
        .op(b"0")
        .stack_global("textwrap", "TextWrapper")
        .empty_tuple()
        .empty_dict()
        .newobj_ex()
        .stop()
    )
    analysis = analyze_pickle(data)
    assert [c.opcode for c in analysis.calls] == ["NEWOBJ", "NEWOBJ_EX"]


def test_mutations_on_globals_only() -> None:
    data = (
        PickleAsm(2)
        .global_("collections", "OrderedDict")
        .empty_list()
        .op(b"a")
        .op(b"0")
        .empty_list()
        .int1(1)
        .op(b"a")
        .stop()
    )
    analysis = analyze_pickle(data)
    assert [m.opcode for m in analysis.mutations] == ["APPEND"]


def test_persistent_ids_and_ext() -> None:
    data = b"Pfoo\n\x80\x02\x82\x07\x86."
    analysis = analyze_pickle(data)
    assert len(analysis.persistent_ids) == 1
    assert isinstance(analysis.persistent_ids[0].pid, Const)
    assert [e.code for e in analysis.extensions] == [7]


def test_embedded_pickle_detection_in_bytes_and_latin1_strings() -> None:
    inner = print_pickle()
    as_bytes = PickleAsm(4).binbytes(inner).stop()
    assert analyze_pickle(as_bytes).embedded[0].data == inner
    # protocol 2 stores bytes as _codecs.encode(<latin-1 text>, 'latin1'); the text
    # constant is re-encoded with latin-1 to recover the embedded pickle
    as_text = PickleAsm(2).binunicode(inner.decode("latin-1")).stop()
    assert analyze_pickle(as_text).embedded[0].data == inner
    not_latin1 = PickleAsm(2).binunicode("\x80\x04\u2603").stop()
    assert analyze_pickle(not_latin1).embedded == []
    short = b"U" + bytes([len(inner)]) + inner + b"."
    assert analyze_pickle(short).embedded[0].data == inner


@pytest.mark.parametrize(
    ("data", "kind"),
    [
        (b"0.", "stack"),  # POP on empty stack
        (b"R.", "stack"),  # REDUCE underflow
        (b"(K\x01t1.", "stack"),  # POP_MARK without mark
        (b"h\x05.", "memo"),  # BINGET of an unset slot
        (b"p-1\n.", "memo"),  # PUT with a negative key
        (b"(K\x01K\x02K\x03d.", "stack"),  # DICT with odd items
        (b"(o.", "stack"),  # OBJ without class
        (b"\x80\x07N.", "protocol"),
        (b"(N", "truncated"),
    ],
)
def test_structural_problems(data: bytes, kind: str) -> None:
    analysis = analyze_pickle(data)
    assert analysis.problem is not None
    assert analysis.problem.kind == kind


def test_mark_fence_blocks_popping_across_marks() -> None:
    # REDUCE right after MARK must not consume values below the mark
    analysis = analyze_pickle(b"K\x01K\x02(R.")
    assert analysis.problem is not None and analysis.problem.kind == "stack"


def test_pop_discards_mark_when_frame_empty() -> None:
    analysis = analyze_pickle(b"K\x01(0.")
    assert analysis.complete
    assert isinstance(analysis.result, Const) and analysis.result.value == 1


@pytest.mark.parametrize(
    ("limits", "data"),
    [
        (ScanLimits(max_pickle_ops=3), b"N0N0N0N."),
        (ScanLimits(max_stack_depth=2), b"NNN..."),
        (ScanLimits(max_memo_entries=1), b"N\x94\x94N\x94."),
    ],
)
def test_resource_limits(limits: ScanLimits, data: bytes) -> None:
    analysis = analyze_pickle(data, limits=limits)
    assert analysis.problem is not None and analysis.problem.kind == "limit"


def test_keep_ops_and_buffers() -> None:
    data = PickleAsm(5).op(b"\x97").op(b"\x98").stop()
    analysis = analyze_pickle(data, keep_ops=True)
    assert analysis.ops is not None
    assert [op.name for op in analysis.ops] == ["PROTO", "NEXT_BUFFER", "READONLY_BUFFER", "STOP"]
    assert isinstance(analysis.result, BufferValue)


def test_describe_renders_every_value_kind() -> None:
    g = GlobalValue("m", "f", 0)
    assert describe(g) == "m.f"
    assert describe(GlobalValue(None, None, 0)) == "<dynamic import>"
    assert describe(TupleValue((Const(1),))) == "(1,)"
    assert describe(TupleValue(tuple(Const(i) for i in range(8)))).endswith("...)")
    assert describe(ContainerValue("dict")) == "<dict>"
    assert describe(PersistentValue(Const("k"), 0)) == "<persistent 'k'>"
    assert describe(ExtValue(3, 0)) == "<extension code 3>"
    assert describe(BufferValue(0)) == "<buffer>"
    assert describe(None) == "?"
    assert describe(Const("x" * 100)).endswith("...")
    deep: CallValue | GlobalValue = g
    for _ in range(6):
        deep = CallValue(deep, TupleValue(()), 0, "REDUCE")
    assert "..." in describe(deep)
    assert describe_args(Const(1)) == "*1"
    assert describe_args(TupleValue(tuple(Const(i) for i in range(8)))).endswith("...")
