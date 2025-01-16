"""Command-line interface: ``picklescope scan | disasm | policy | rules``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path

from picklescope import __version__
from picklescope.blob import open_buffer
from picklescope.findings import Severity
from picklescope.limits import ScanLimits
from picklescope.output import render_json, render_text, to_sarif
from picklescope.pkl.analyzer import CallValue, GlobalValue, analyze_pickle, describe
from picklescope.pkl.disasm import format_ops
from picklescope.policy import (
    PROFILES,
    Pattern,
    PatternError,
    Policy,
    PolicyError,
    default_policy,
    load_policy,
)
from picklescope.rules import RULES
from picklescope.scanner import scan_paths

EXIT_OK, EXIT_FINDINGS, EXIT_ERROR = 0, 1, 2


def _severity(text: str) -> Severity:
    try:
        return Severity.parse(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _size(text: str) -> int:
    units = {"k": 1024, "m": 1024**2, "g": 1024**3}
    text = text.strip().lower()
    factor = units.get(text[-1:], 1)
    number = text[:-1] if text[-1:] in units else text
    if not number.isdigit() or int(number) <= 0:
        raise argparse.ArgumentTypeError(f"invalid size {text!r} (examples: 512m, 4g)")
    return int(number) * factor


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="picklescope",
        description="Static scanner for ML model files. Nothing is ever deserialized.",
    )
    parser.add_argument("--version", action="version", version=f"picklescope {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="more logging")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="scan files or directories")
    scan.add_argument("paths", nargs="+", type=Path)
    scan.add_argument("-f", "--format", choices=("text", "json", "sarif"), default="text")
    scan.add_argument("-o", "--output", type=Path, help="write the report to a file")
    _policy_args(scan)
    scan.add_argument(
        "--min-severity",
        type=_severity,
        default=Severity.LOW,
        help="lowest severity to report (default: low)",
    )
    scan.add_argument(
        "--fail-on",
        type=_severity,
        default=Severity.MEDIUM,
        help="exit 1 if any finding is at least this severe (default: medium)",
    )
    scan.add_argument(
        "--show-imports",
        action="store_true",
        help="list every imported reference with its policy decision",
    )
    scan.add_argument("--max-file-size", type=_size, default=ScanLimits().max_file_size)
    scan.add_argument("--no-hash", action="store_true", help="skip SHA-256 of scanned files")
    scan.add_argument("--follow-symlinks", action="store_true")
    scan.add_argument("--color", choices=("auto", "always", "never"), default="auto")

    disasm = sub.add_parser("disasm", help="annotated disassembly of a pickle")
    disasm.add_argument("path", type=Path)
    disasm.add_argument("--member", help="zip member holding the pickle (e.g. archive/data.pkl)")
    _policy_args(disasm)

    policy = sub.add_parser("policy", help="inspect the import policy")
    policy_sub = policy.add_subparsers(dest="policy_command", required=True)
    check = policy_sub.add_parser("check", help="classify module:name references")
    check.add_argument("references", nargs="+", metavar="MODULE:NAME")
    _policy_args(check)
    show = policy_sub.add_parser("show", help="print the effective policy as JSON")
    _policy_args(show)

    rules = sub.add_parser("rules", help="list the rule catalogue")
    rules.add_argument("-f", "--format", choices=("text", "json", "markdown"), default="text")
    return parser


def _policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", type=Path, help="TOML policy file")
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        choices=sorted(PROFILES),
        help="enable an extra built-in allow profile (repeatable)",
    )


def _load_policy(args: argparse.Namespace) -> Policy:
    if args.policy:
        return load_policy(args.policy, args.profile)
    base = default_policy()
    return base.with_profiles(args.profile) if args.profile else base


def _write(text: str, output: Path | None) -> None:
    if output is None:
        sys.stdout.write(text + "\n")
    else:
        output.write_text(text + "\n", encoding="utf-8")


def _cmd_scan(args: argparse.Namespace) -> int:
    policy = _load_policy(args)
    limits = ScanLimits(max_file_size=args.max_file_size)
    missing = [str(p) for p in args.paths if not p.exists()]
    if missing:
        print(f"picklescope: no such file or directory: {', '.join(missing)}", file=sys.stderr)
        return EXIT_ERROR
    report = scan_paths(
        args.paths,
        policy,
        limits,
        follow_symlinks=args.follow_symlinks,
        hash_files=not args.no_hash,
    )
    if args.format == "json":
        text = render_json(report)
    elif args.format == "sarif":
        text = json.dumps(to_sarif(report, args.min_severity), indent=2, default=str)
    else:
        color = args.color == "always" or (
            args.color == "auto"
            and args.output is None
            and sys.stdout.isatty()
            and "NO_COLOR" not in os.environ
        )
        text = render_text(report, args.min_severity, color=color, show_imports=args.show_imports)
    _write(text, args.output)
    return EXIT_FINDINGS if report.exceeds(args.fail_on) else EXIT_OK


def _disasm_bytes(data: bytes, policy: Policy) -> str:
    out: list[str] = []
    pos = 0
    while pos < len(data):
        analysis = analyze_pickle(data, pos, keep_ops=True)
        notes: dict[int, str] = {}
        for event in analysis.imports:
            verdict = policy.classify(event.module, event.name)
            notes[event.offset] = f"import {event.module}.{event.name} -> {verdict.decision.value}"
        for dyn in analysis.dynamic_imports:
            notes[dyn.offset] = "dynamic import (operands not constant)"
        for call in analysis.calls:
            callee = call.callee
            if isinstance(callee, CallValue):
                note = f"call the result of {describe(callee)}"
            elif isinstance(callee, GlobalValue):
                note = f"call {describe(callee)}"
            else:
                note = f"call {describe(callee)} (not an imported name)"
            existing = notes.get(call.offset)
            notes[call.offset] = f"{existing}; {note}" if existing else note
        out.append(f"--- pickle at byte {pos} (protocol {analysis.protocol}) ---")
        out.append(format_ops(analysis.ops or [], notes))
        if analysis.problem is not None:
            out.append(f"!!! {analysis.problem.kind}: {analysis.problem.message}")
            break
        pos = analysis.end
    return "\n".join(out)


def _cmd_disasm(args: argparse.Namespace) -> int:
    policy = _load_policy(args)
    if args.member:
        try:
            with zipfile.ZipFile(args.path) as zf:
                data = zf.read(args.member)
        except (KeyError, zipfile.BadZipFile, OSError) as exc:
            print(f"picklescope: cannot read member {args.member!r}: {exc}", file=sys.stderr)
            return EXIT_ERROR
    else:
        try:
            with open_buffer(args.path) as buf:
                data = bytes(buf)
        except OSError as exc:
            print(f"picklescope: {exc}", file=sys.stderr)
            return EXIT_ERROR
    _write(_disasm_bytes(data, policy), None)
    return EXIT_OK


def _cmd_policy(args: argparse.Namespace) -> int:
    policy = _load_policy(args)
    if args.policy_command == "show":
        _write(json.dumps(policy.describe(), indent=2), None)
        return EXIT_OK
    status = EXIT_OK
    for ref in args.references:
        try:
            pattern = Pattern.parse(ref)
        except PatternError as exc:
            print(f"picklescope: {exc}", file=sys.stderr)
            return EXIT_ERROR
        if pattern.module_tree or pattern.name in ("*", "**"):
            print(
                f"picklescope: {ref!r} must be a concrete reference, not a pattern", file=sys.stderr
            )
            return EXIT_ERROR
        verdict = policy.classify(pattern.module, pattern.name)
        entry = verdict.entry
        why = f"{entry.pattern} [{entry.source}] {entry.reason}" if entry else "no matching entry"
        print(f"{verdict.decision.value:<8} {pattern.module}:{pattern.name}  ({why})")
        if verdict.decision.value != "allow":
            status = EXIT_FINDINGS
    return status


def _cmd_rules(args: argparse.Namespace) -> int:
    rules = list(RULES.values())
    if args.format == "json":
        payload = [
            {
                "id": r.id,
                "name": r.name,
                "severity": r.severity.label,
                "summary": r.summary,
                "rationale": r.rationale,
            }
            for r in rules
        ]
        _write(json.dumps(payload, indent=2), None)
    elif args.format == "markdown":
        lines = ["| ID | Name | Severity | Summary |", "|---|---|---|---|"]
        lines += [f"| {r.id} | {r.name} | {r.severity.label} | {r.summary} |" for r in rules]
        lines.append("")
        for r in rules:
            lines += [
                f"### {r.id} {r.name}",
                "",
                f"**{r.severity.label}** - {r.summary}",
                "",
                r.rationale,
                "",
            ]
        _write("\n".join(lines).rstrip(), None)
    else:
        for r in rules:
            print(f"{r.id:<7} {r.severity.label:<9} {r.name:<30} {r.summary}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose, 2),
        format="%(levelname)s %(name)s: %(message)s",
    )
    handlers = {
        "scan": _cmd_scan,
        "disasm": _cmd_disasm,
        "policy": _cmd_policy,
        "rules": _cmd_rules,
    }
    try:
        return handlers[args.command](args)
    except PolicyError as exc:
        print(f"picklescope: {exc}", file=sys.stderr)
        return EXIT_ERROR
