"""Abstract interpretation of pickle bytecode.

The analyzer replays a pickle's stack machine over *abstract* values. Constants are
kept (so ``STACK_GLOBAL`` operands and call arguments can be recovered), imports become
:class:`GlobalValue` placeholders, and calls become :class:`CallValue` nodes. Nothing
is imported, constructed or called: the output is a list of events describing what the
real unpickler *would* import and invoke.

The stack/mark discipline follows CPython's C unpickler (``Modules/_pickle.c``): marks
are fences, popping across a fence is an error, ``POP`` on an empty frame discards the
mark. This keeps the analysis aligned with what a real load would do on the same bytes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from picklescope.blob import Buffer
from picklescope.limits import ScanLimits
from picklescope.pkl.disasm import Op, iter_ops, looks_like_pickle
from picklescope.pkl.opcodes import HIGHEST_PROTOCOL, PickleParseError

# --------------------------------------------------------------------------- values


class Value:
    """Base class for abstract stack values."""

    __slots__ = ()


@dataclass(frozen=True, slots=True, eq=False)
class Const(Value):
    value: object


@dataclass(frozen=True, slots=True, eq=False)
class GlobalValue(Value):
    """Result of GLOBAL/STACK_GLOBAL/INST. ``module``/``name`` are None when dynamic."""

    module: str | None
    name: str | None
    offset: int


@dataclass(frozen=True, slots=True, eq=False)
class CallValue(Value):
    callee: Value
    args: Value
    offset: int
    opcode: str


@dataclass(frozen=True, slots=True, eq=False)
class TupleValue(Value):
    items: tuple[Value, ...]


@dataclass(frozen=True, slots=True, eq=False)
class ContainerValue(Value):
    kind: str  # list | dict | set | frozenset


@dataclass(frozen=True, slots=True, eq=False)
class PersistentValue(Value):
    pid: Value
    offset: int


@dataclass(frozen=True, slots=True, eq=False)
class ExtValue(Value):
    code: int
    offset: int


@dataclass(frozen=True, slots=True, eq=False)
class BufferValue(Value):
    offset: int


# --------------------------------------------------------------------------- events


@dataclass(frozen=True, slots=True)
class ImportEvent:
    module: str
    name: str
    offset: int
    opcode: str


@dataclass(frozen=True, slots=True)
class CallEvent:
    callee: Value
    args: Value
    offset: int
    opcode: str


@dataclass(frozen=True, slots=True)
class DynamicImportEvent:
    offset: int
    module: Value
    name: Value


@dataclass(frozen=True, slots=True)
class MutationEvent:
    offset: int
    opcode: str
    target: GlobalValue


@dataclass(frozen=True, slots=True)
class PersistentEvent:
    offset: int
    pid: Value


@dataclass(frozen=True, slots=True)
class EmbeddedBlob:
    offset: int
    data: bytes


@dataclass(frozen=True, slots=True)
class AnalysisProblem:
    """A structural problem that stopped (or limited) analysis."""

    kind: str  # truncated | unknown_opcode | bad_argument | stack | memo | protocol | limit
    offset: int
    message: str


@dataclass(slots=True)
class PickleAnalysis:
    start: int
    end: int = 0
    protocol: int | None = None
    op_count: int = 0
    imports: list[ImportEvent] = field(default_factory=list)
    calls: list[CallEvent] = field(default_factory=list)
    dynamic_imports: list[DynamicImportEvent] = field(default_factory=list)
    extensions: list[ExtValue] = field(default_factory=list)
    mutations: list[MutationEvent] = field(default_factory=list)
    persistent_ids: list[PersistentEvent] = field(default_factory=list)
    embedded: list[EmbeddedBlob] = field(default_factory=list)
    problem: AnalysisProblem | None = None
    result: Value | None = None
    ops: list[Op] | None = None

    @property
    def complete(self) -> bool:
        return self.problem is None


class _StackError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


# --------------------------------------------------------------------------- rendering


def describe(value: Value | None, depth: int = 0) -> str:
    """Short human-readable rendering of an abstract value."""
    if value is None:
        return "?"
    if depth > 3:
        return "..."
    if isinstance(value, Const):
        text = repr(value.value)
        return text if len(text) <= 48 else text[:45] + "..."
    if isinstance(value, GlobalValue):
        if value.module is None or value.name is None:
            return "<dynamic import>"
        return f"{value.module}.{value.name}"
    if isinstance(value, CallValue):
        return f"{describe(value.callee, depth + 1)}({describe_args(value.args, depth + 1)})"
    if isinstance(value, TupleValue):
        inner = ", ".join(describe(v, depth + 1) for v in value.items[:6])
        if len(value.items) > 6:
            inner += ", ..."
        return f"({inner}{',' if len(value.items) == 1 else ''})"
    if isinstance(value, ContainerValue):
        return f"<{value.kind}>"
    if isinstance(value, PersistentValue):
        return f"<persistent {describe(value.pid, depth + 1)}>"
    if isinstance(value, ExtValue):
        return f"<extension code {value.code}>"
    return "<buffer>"


def describe_args(args: Value, depth: int = 0) -> str:
    """Render a call's argument tuple without the surrounding parentheses."""
    if isinstance(args, TupleValue):
        return ", ".join(describe(v, depth) for v in args.items[:6]) + (
            ", ..." if len(args.items) > 6 else ""
        )
    return "*" + describe(args, depth)


# --------------------------------------------------------------------------- analyzer


class _Machine:
    def __init__(self, buf: Buffer, start: int, limits: ScanLimits, keep_ops: bool) -> None:
        self.buf = buf
        self.limits = limits
        self.stack: list[Value] = []
        self.marks: list[int] = []
        self.memo: dict[int, Value] = {}
        self.result = PickleAnalysis(start=start, ops=[] if keep_ops else None)
        self.handlers: dict[str, Callable[[Op], None]] = self._build_handlers()

    # -- stack primitives ----------------------------------------------------------

    def _fence(self) -> int:
        return self.marks[-1] if self.marks else 0

    def push(self, value: Value) -> None:
        self.stack.append(value)
        if len(self.stack) > self.limits.max_stack_depth:
            raise _StackError("limit", "abstract stack exceeded max_stack_depth")

    def pop(self) -> Value:
        if len(self.stack) <= self._fence():
            raise _StackError("stack", "stack underflow")
        return self.stack.pop()

    def top(self) -> Value:
        if len(self.stack) <= self._fence():
            raise _StackError("stack", "stack underflow")
        return self.stack[-1]

    def pop_mark(self) -> list[Value]:
        if not self.marks:
            raise _StackError("stack", "MARK expected but none is open")
        fence = self.marks.pop()
        items = self.stack[fence:]
        del self.stack[fence:]
        return items

    # -- event helpers ---------------------------------------------------------------

    def _const(self, op: Op) -> None:
        arg = op.arg
        if isinstance(arg, (bytes, bytearray)):
            data = bytes(arg)
            if looks_like_pickle(data):
                self.result.embedded.append(EmbeddedBlob(op.offset, data))
        elif isinstance(arg, str) and arg[:1] == "\x80" and len(arg) >= 3:
            try:
                data = arg.encode("latin-1")
            except UnicodeEncodeError:
                data = b""
            if looks_like_pickle(data):
                self.result.embedded.append(EmbeddedBlob(op.offset, data))
        self.push(Const(arg))

    def _import(self, module: str, name: str, op: Op) -> GlobalValue:
        self.result.imports.append(ImportEvent(module, name, op.offset, op.name))
        return GlobalValue(module, name, op.offset)

    def _call(self, callee: Value, args: Value, op: Op) -> None:
        self.result.calls.append(CallEvent(callee, args, op.offset, op.name))
        self.push(CallValue(callee, args, op.offset, op.name))

    def _mutate(self, target: Value, op: Op) -> None:
        if isinstance(target, GlobalValue):
            self.result.mutations.append(MutationEvent(op.offset, op.name, target))

    def _text_pair(self, op: Op) -> tuple[str, str]:
        raw = self.buf[op.offset + 1 : op.end]
        parts = raw.split(b"\n")
        return parts[0].decode("utf-8", "replace"), parts[1].decode("utf-8", "replace")

    def _memo_get(self, key: object, op: Op) -> None:
        if not isinstance(key, int) or isinstance(key, bool):
            key = int(bool(key)) if isinstance(key, bool) else -1
        if key not in self.memo:
            raise _StackError("memo", f"memo key {key} read before it was written")
        self.push(self.memo[key])

    def _memo_put(self, key: object) -> None:
        if isinstance(key, bool):
            key = int(key)
        if not isinstance(key, int) or key < 0:
            raise _StackError("memo", f"invalid memo key {key!r}")
        if key not in self.memo and len(self.memo) >= self.limits.max_memo_entries:
            raise _StackError("limit", "memo exceeded max_memo_entries")
        self.memo[key] = self.top()

    # -- handlers ----------------------------------------------------------------------

    def _build_handlers(self) -> dict[str, Callable[[Op], None]]:
        h: dict[str, Callable[[Op], None]] = {}
        for name in (
            "INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4", "STRING",
            "BINSTRING", "SHORT_BINSTRING", "BINBYTES", "SHORT_BINBYTES", "BINBYTES8",
            "BYTEARRAY8", "UNICODE", "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8",
            "FLOAT", "BINFLOAT",
        ):  # fmt: skip
            h[name] = self._const
        h["NONE"] = lambda op: self.push(Const(None))
        h["NEWTRUE"] = lambda op: self.push(Const(True))
        h["NEWFALSE"] = lambda op: self.push(Const(False))
        h["NEXT_BUFFER"] = lambda op: self.push(BufferValue(op.offset))
        h["READONLY_BUFFER"] = lambda op: self.push(self.pop())
        h["EMPTY_LIST"] = lambda op: self.push(ContainerValue("list"))
        h["EMPTY_DICT"] = lambda op: self.push(ContainerValue("dict"))
        h["EMPTY_SET"] = lambda op: self.push(ContainerValue("set"))
        h["EMPTY_TUPLE"] = lambda op: self.push(TupleValue(()))
        h["APPEND"] = self._append
        h["APPENDS"] = self._appends
        h["SETITEM"] = self._setitem
        h["SETITEMS"] = self._appends
        h["ADDITEMS"] = self._appends
        h["LIST"] = lambda op: self._collect(op, "list")
        h["DICT"] = self._dict
        h["FROZENSET"] = lambda op: self._collect(op, "frozenset")
        h["TUPLE"] = lambda op: self.push(TupleValue(tuple(self.pop_mark())))
        h["TUPLE1"] = lambda op: self._tuple_n(1)
        h["TUPLE2"] = lambda op: self._tuple_n(2)
        h["TUPLE3"] = lambda op: self._tuple_n(3)
        h["POP"] = self._pop
        h["DUP"] = lambda op: self.push(self.top())
        h["MARK"] = lambda op: self.marks.append(len(self.stack))
        h["POP_MARK"] = lambda op: self._discard_mark()
        h["GET"] = h["BINGET"] = h["LONG_BINGET"] = lambda op: self._memo_get(op.arg, op)
        h["PUT"] = h["BINPUT"] = h["LONG_BINPUT"] = lambda op: self._memo_put(op.arg)
        h["MEMOIZE"] = lambda op: self._memo_put(len(self.memo))
        h["EXT1"] = h["EXT2"] = h["EXT4"] = self._ext
        h["GLOBAL"] = self._global
        h["STACK_GLOBAL"] = self._stack_global
        h["REDUCE"] = self._reduce
        h["BUILD"] = self._build
        h["INST"] = self._inst
        h["OBJ"] = self._obj
        h["NEWOBJ"] = self._newobj
        h["NEWOBJ_EX"] = self._newobj_ex
        h["PROTO"] = self._proto
        h["FRAME"] = lambda op: None
        h["PERSID"] = lambda op: self._persistent(Const(op.arg), op)
        h["BINPERSID"] = lambda op: self._persistent(self.pop(), op)
        h["STOP"] = self._stop
        return h

    def _collect(self, op: Op, kind: str) -> None:
        self.pop_mark()
        self.push(ContainerValue(kind))

    def _discard_mark(self) -> None:
        self.pop_mark()

    def _tuple_n(self, n: int) -> None:
        items = [self.pop() for _ in range(n)]
        self.push(TupleValue(tuple(reversed(items))))

    def _pop(self, op: Op) -> None:
        if len(self.stack) > self._fence():
            self.stack.pop()
        elif self.marks:
            self.marks.pop()
        else:
            raise _StackError("stack", "POP on empty stack")

    def _append(self, op: Op) -> None:
        self.pop()
        self._mutate(self.top(), op)

    def _appends(self, op: Op) -> None:
        self.pop_mark()
        self._mutate(self.top(), op)

    def _setitem(self, op: Op) -> None:
        self.pop()
        self.pop()
        self._mutate(self.top(), op)

    def _dict(self, op: Op) -> None:
        items = self.pop_mark()
        if len(items) % 2:
            raise _StackError("stack", "DICT with an odd number of items")
        self.push(ContainerValue("dict"))

    def _ext(self, op: Op) -> None:
        assert isinstance(op.arg, int)
        ext = ExtValue(op.arg, op.offset)
        self.result.extensions.append(ext)
        self.push(ext)

    def _global(self, op: Op) -> None:
        module, name = self._text_pair(op)
        self.push(self._import(module, name, op))

    def _stack_global(self, op: Op) -> None:
        name = self.pop()
        module = self.pop()
        if (
            isinstance(module, Const)
            and isinstance(name, Const)
            and isinstance(module.value, str)
            and isinstance(name.value, str)
        ):
            self.push(self._import(module.value, name.value, op))
        else:
            self.result.dynamic_imports.append(DynamicImportEvent(op.offset, module, name))
            self.push(GlobalValue(None, None, op.offset))

    def _reduce(self, op: Op) -> None:
        args = self.pop()
        callee = self.pop()
        self._call(callee, args, op)

    def _build(self, op: Op) -> None:
        self.pop()
        self._mutate(self.top(), op)

    def _inst(self, op: Op) -> None:
        module, name = self._text_pair(op)
        args = self.pop_mark()
        cls = self._import(module, name, op)
        self._call(cls, TupleValue(tuple(args)), op)

    def _obj(self, op: Op) -> None:
        items = self.pop_mark()
        if not items:
            raise _StackError("stack", "OBJ without a class on the stack")
        self._call(items[0], TupleValue(tuple(items[1:])), op)

    def _newobj(self, op: Op) -> None:
        args = self.pop()
        cls = self.pop()
        self._call(cls, args, op)

    def _newobj_ex(self, op: Op) -> None:
        self.pop()  # kwargs
        args = self.pop()
        cls = self.pop()
        self._call(cls, args, op)

    def _proto(self, op: Op) -> None:
        assert isinstance(op.arg, int)
        if self.result.protocol is None:
            self.result.protocol = op.arg
        if op.arg > HIGHEST_PROTOCOL:
            raise _StackError("protocol", f"unsupported pickle protocol {op.arg}")

    def _persistent(self, pid: Value, op: Op) -> None:
        self.result.persistent_ids.append(PersistentEvent(op.offset, pid))
        self.push(PersistentValue(pid, op.offset))

    def _stop(self, op: Op) -> None:
        self.result.result = self.pop()

    # -- driver ------------------------------------------------------------------------

    def run(self) -> PickleAnalysis:
        res = self.result
        offset = res.start
        try:
            for op in iter_ops(self.buf, res.start):
                offset = op.offset
                res.op_count += 1
                if res.op_count > self.limits.max_pickle_ops:
                    raise _StackError("limit", "opcode count exceeded max_pickle_ops")
                if res.ops is not None:
                    res.ops.append(op)
                self.handlers[op.name](op)
                res.end = op.end
        except PickleParseError as exc:
            res.problem = AnalysisProblem(exc.kind, exc.offset, exc.message)
            res.end = len(self.buf)
        except _StackError as exc:
            res.problem = AnalysisProblem(exc.kind, offset, exc.message)
            res.end = len(self.buf)
        return res


def analyze_pickle(
    buf: Buffer, start: int = 0, limits: ScanLimits | None = None, *, keep_ops: bool = False
) -> PickleAnalysis:
    """Analyse the single pickle that begins at ``start`` in ``buf``.

    The returned :class:`PickleAnalysis` records imports, calls and anomalies. If the
    stream is malformed, ``problem`` describes the first error and ``end`` is set to
    ``len(buf)``; events gathered before the error are retained.
    """
    return _Machine(buf, start, limits or ScanLimits(), keep_ops).run()
