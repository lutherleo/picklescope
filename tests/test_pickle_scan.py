"""Policy-driven pickle findings."""

from __future__ import annotations

from picklescope.findings import Severity
from picklescope.limits import ScanLimits
from picklescope.pkl.scan import scan_pickle

from conftest import LOC, make_ctx, rules_of
from fixturegen import PickleAsm, benign_pickle, legacy_torch_file, print_pickle


def test_findings_carry_offsets_and_details() -> None:
    ctx = make_ctx()
    scan_pickle(print_pickle("hello"), LOC, ctx, "plain")
    (finding,) = ctx.findings
    assert finding.rule_id == "PKL102"
    assert finding.severity is Severity.HIGH
    assert finding.location.offset == 19  # the STACK_GLOBAL opcode
    assert finding.details["example_call"] == "builtins.print('hello')"
    assert finding.details["occurrences"] == 1
    assert "builtins.print('hello')" in finding.message


def test_repeated_import_is_grouped() -> None:
    asm = PickleAsm(2)
    for _ in range(3):
        asm.global_("textwrap", "dedent").op(b"0")
    ctx = make_ctx()
    scan_pickle(asm.none().stop(), LOC, ctx, "plain")
    (finding,) = ctx.findings
    assert finding.details["occurrences"] == 3
    (record,) = ctx.imports.values()
    assert record.count == 3 and record.decision == "unknown"


def test_deny_listed_import_is_critical_even_without_call() -> None:
    ctx = make_ctx()
    scan_pickle(PickleAsm(2).global_("builtins", "getattr").stop(), LOC, ctx, "plain")
    (finding,) = ctx.findings
    assert finding.severity is Severity.CRITICAL
    assert finding.details["category"] == "attribute-gadget"
    assert finding.message.startswith("imports deny-listed")


def test_persistent_ids_allowed_in_torch_context() -> None:
    data = PickleAsm(2).unicode("k").binpersid().stop()
    plain = make_ctx()
    scan_pickle(data, LOC, plain, "plain")
    torch = make_ctx()
    result = scan_pickle(data, LOC, torch, "torch")
    assert rules_of(plain) == {"PKL108"}
    assert rules_of(torch) == set()
    assert len(result.persistent_ids) == 1


def test_legacy_torch_layout_is_recognised() -> None:
    ctx = make_ctx()
    result = scan_pickle(legacy_torch_file(), LOC, ctx, "plain")
    assert result.legacy_torch
    assert len(result.analyses) == 5
    assert rules_of(ctx) == set()


def test_multiple_pickle_cap() -> None:
    ctx = make_ctx(ScanLimits(max_pickles_per_stream=3))
    scan_pickle(benign_pickle() * 5, LOC, ctx, "plain")
    assert rules_of(ctx) == {"PKL110", "PKL116"}


def test_broken_second_pickle_is_reported_as_malformed() -> None:
    ctx = make_ctx()
    scan_pickle(benign_pickle() + print_pickle()[:-3], LOC, ctx, "plain")
    assert {"PKL110", "PKL112", "PKL101"} <= rules_of(ctx)


def test_empty_stream() -> None:
    ctx = make_ctx()
    scan_pickle(b"", LOC, ctx, "plain")
    (finding,) = ctx.findings
    assert finding.rule_id == "PKL112" and finding.severity is Severity.LOW


def test_nesting_limit_stops_recursion() -> None:
    inner = print_pickle()
    for _ in range(6):
        inner = PickleAsm(4).binbytes(inner).stop()
    ctx = make_ctx(ScanLimits(max_nesting=2))
    scan_pickle(inner, LOC, ctx, "plain")
    rules = rules_of(ctx)
    assert "PKL116" in rules and "PKL102" not in rules
    deep = make_ctx(ScanLimits(max_nesting=10))
    scan_pickle(inner, LOC, deep, "plain")
    assert "PKL102" in rules_of(deep)
    nested = next(f for f in deep.findings if f.rule_id == "PKL102")
    assert len(nested.location.members) == 6


def test_finding_flood_is_capped() -> None:
    asm = PickleAsm(2)
    for index in range(30):
        asm.global_("fixture_mod", f"name{index}").op(b"0")
    ctx = make_ctx(ScanLimits(max_findings_per_rule=5))
    scan_pickle(asm.none().stop(), LOC, ctx, "plain")
    pkl101 = [f for f in ctx.findings if f.rule_id == "PKL101"]
    assert len(pkl101) == 6
    assert pkl101[-1].severity is Severity.INFO and pkl101[-1].details["suppressed"]


def test_indirect_call_through_persistent_object() -> None:
    data = PickleAsm(2).unicode("k").binpersid().empty_tuple().reduce().stop()
    ctx = make_ctx()
    scan_pickle(data, LOC, ctx, "torch")
    assert rules_of(ctx) == {"PKL106"}
