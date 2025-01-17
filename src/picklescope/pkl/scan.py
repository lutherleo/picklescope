"""Turn pickle analyses into findings by applying the import policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from picklescope.blob import Buffer
from picklescope.context import ScanContext
from picklescope.findings import Location, Severity
from picklescope.pkl.analyzer import (
    CallValue,
    Const,
    GlobalValue,
    PersistentEvent,
    PersistentValue,
    PickleAnalysis,
    analyze_pickle,
    describe,
    describe_args,
)
from picklescope.pkl.disasm import looks_like_pickle
from picklescope.policy import Decision, has_dunder_traversal, is_valid_reference

log = logging.getLogger(__name__)

TORCH_LEGACY_MAGIC = 0x1950A86A20F9469CFC6C
"""First pickle of a legacy (pre-1.6, non-zip) torch.save file."""
TORCH_LEGACY_PICKLES = 5
"""magic, protocol version, sys_info, payload, storage keys - then raw storage bytes."""

PERSISTENT_CONTEXTS = frozenset({"torch", "torch-legacy"})


@dataclass(slots=True)
class PickleScanResult:
    analyses: list[PickleAnalysis] = field(default_factory=list)
    persistent_ids: list[PersistentEvent] = field(default_factory=list)
    legacy_torch: bool = False


def _problem_rule(kind: str) -> str:
    return {
        "unknown_opcode": "PKL113",
        "protocol": "PKL114",
        "limit": "PKL116",
    }.get(kind, "PKL112")


def _report_imports(analysis: PickleAnalysis, loc: Location, ctx: ScanContext) -> None:
    invoked: dict[int, str] = {}
    for call in analysis.calls:
        if isinstance(call.callee, GlobalValue) and call.callee.module is not None:
            invoked.setdefault(
                call.callee.offset, f"{describe(call.callee)}({describe_args(call.args)})"
            )

    grouped: dict[tuple[str, str], list[int]] = {}
    for event in analysis.imports:
        grouped.setdefault((event.module, event.name), []).append(event.offset)

    for (module, name), offsets in grouped.items():
        here = loc.at(offsets[0])
        qualified = f"{module}.{name}"
        calls = [invoked[o] for o in offsets if o in invoked]
        verdict = ctx.policy.classify(module, name)
        ctx.record_import(verdict, loc, len(offsets))
        extra: dict[str, Any] = {"module": module, "name": name, "occurrences": len(offsets)}
        if calls:
            extra["example_call"] = calls[0]
        if not is_valid_reference(module, name):
            ctx.emit("PKL115", here, f"malformed global name {module!r} / {name!r}", **extra)
        if has_dunder_traversal(name):
            ctx.emit("PKL104", here, f"import of {qualified} walks dunder attributes", **extra)
        if verdict.decision is Decision.DENY:
            entry = verdict.entry
            assert entry is not None
            action = "invokes" if calls else "imports"
            ctx.emit(
                "PKL100",
                here,
                f"{action} deny-listed {qualified} ({entry.category}: {entry.reason})",
                category=entry.category,
                matched=str(entry.pattern),
                policy_source=entry.source,
                **extra,
            )
        elif verdict.decision is Decision.UNKNOWN:
            if calls:
                ctx.emit(
                    "PKL102",
                    here,
                    f"invokes {calls[0]} which is not on the allow-list",
                    **extra,
                )
            else:
                ctx.emit(
                    "PKL101", here, f"imports {qualified} which is not on the allow-list", **extra
                )


def _report_analysis(
    analysis: PickleAnalysis,
    loc: Location,
    ctx: ScanContext,
    context: str,
    result: PickleScanResult,
) -> None:
    _report_imports(analysis, loc, ctx)

    for event in analysis.dynamic_imports:
        ctx.emit(
            "PKL103",
            loc.at(event.offset),
            f"STACK_GLOBAL with non-constant operands ({describe(event.module)}, "
            f"{describe(event.name)})",
        )
    for ext in analysis.extensions:
        ctx.emit(
            "PKL105",
            loc.at(ext.offset),
            f"extension-registry code {ext.code} imports an object chosen by copyreg",
            code=ext.code,
        )
    for call in analysis.calls:
        callee = call.callee
        if isinstance(callee, (CallValue, PersistentValue)):
            ctx.emit(
                "PKL106",
                loc.at(call.offset),
                f"{call.opcode} calls a runtime-computed object: {describe(callee)}(...)",
            )
    for mutation in analysis.mutations:
        ctx.emit(
            "PKL107",
            loc.at(mutation.offset),
            f"{mutation.opcode} modifies imported object {describe(mutation.target)}",
        )

    result.persistent_ids.extend(analysis.persistent_ids)
    if analysis.persistent_ids and context not in PERSISTENT_CONTEXTS:
        first = analysis.persistent_ids[0]
        ctx.emit(
            "PKL108",
            loc.at(first.offset),
            f"{len(analysis.persistent_ids)} persistent id(s) resolved by application code, "
            f"first: {describe(first.pid)}",
            count=len(analysis.persistent_ids),
        )

    for blob in analysis.embedded:
        child = loc.child(f"embedded@{blob.offset}")
        ctx.emit(
            "PKL109",
            loc.at(blob.offset),
            f"bytes constant of {len(blob.data)} bytes is itself a pickle; "
            f"scanned as {child.display()}",
        )
        if ctx.can_nest():
            scan_pickle(blob.data, child, ctx.nested(), "plain")
        else:
            ctx.emit("PKL116", child, "embedded pickle not scanned: nesting limit reached")

    if analysis.problem is not None:
        problem = analysis.problem
        ctx.emit(
            _problem_rule(problem.kind),
            loc.at(problem.offset),
            f"{problem.message}; analysed {analysis.op_count} opcode(s) before stopping",
            kind=problem.kind,
        )


def _is_torch_magic(analysis: PickleAnalysis) -> bool:
    value = analysis.result
    return isinstance(value, Const) and value.value == TORCH_LEGACY_MAGIC


def scan_pickle(data: Buffer, loc: Location, ctx: ScanContext, context: str) -> PickleScanResult:
    """Scan every pickle in ``data`` (a stream may hold several) and emit findings.

    ``context`` names the container: ``plain``, ``torch`` (zip archive member) or
    ``npy``; it decides whether persistent ids are expected.
    """
    result = PickleScanResult()
    pos = 0
    size = len(data)
    reported_multiple = False
    while pos < size:
        analysis = analyze_pickle(data, pos, ctx.limits)
        index = len(result.analyses)
        if index > 0 and analysis.problem is not None and not looks_like_pickle(data, pos):
            ctx.emit(
                "PKL111",
                loc.at(pos),
                f"{size - pos} byte(s) of non-pickle data follow the last STOP",
                trailing_bytes=size - pos,
            )
            break
        if index > 0 and not result.legacy_torch and not reported_multiple:
            ctx.emit(
                "PKL110",
                loc.at(pos),
                "stream continues with another pickle after STOP; repeated load() calls execute it",
            )
            reported_multiple = True
        result.analyses.append(analysis)
        effective = "torch-legacy" if result.legacy_torch else context
        _report_analysis(analysis, loc, ctx, effective, result)
        if analysis.problem is not None:
            break
        if index == 0 and _is_torch_magic(analysis):
            result.legacy_torch = True
            log.debug("%s: legacy torch.save layout detected", loc.display())
        pos = analysis.end
        if result.legacy_torch and len(result.analyses) == TORCH_LEGACY_PICKLES:
            break
        if len(result.analyses) >= ctx.limits.max_pickles_per_stream and pos < size:
            ctx.emit(
                "PKL116",
                loc.at(pos),
                f"stopped after {len(result.analyses)} consecutive pickles",
            )
            break
    if not result.analyses:
        ctx.emit("PKL112", loc, "empty pickle stream", severity=Severity.LOW)
    return result
