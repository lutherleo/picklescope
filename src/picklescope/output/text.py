"""Human-readable and JSON renderings of a scan report."""

from __future__ import annotations

import json
from typing import Any

from picklescope import __version__
from picklescope.findings import FileReport, ScanReport, Severity

_COLORS = {
    Severity.CRITICAL: "\033[1;31m",
    Severity.HIGH: "\033[31m",
    Severity.MEDIUM: "\033[33m",
    Severity.LOW: "\033[36m",
    Severity.INFO: "\033[2m",
}
_RESET = "\033[0m"


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"  # pragma: no cover - loop always returns


def _file_block(file: FileReport, min_severity: Severity, color: bool, imports: bool) -> list[str]:
    shown = [f for f in file.findings if f.severity >= min_severity]
    header = f"{file.path}  [{file.format}, {_human_size(file.size)}]"
    if file.skipped_reason:
        return [f"{header}  skipped: {file.skipped_reason}"] if not shown else [header]
    lines = [f"{header}  {'ok' if not shown else ''}".rstrip()]
    for item in sorted(shown, key=lambda f: (-f.severity, f.rule_id, f.location.offset or 0)):
        label = f"{item.severity.name:<8}"
        if color:
            label = f"{_COLORS[item.severity]}{label}{_RESET}"
        where = f" [{'!'.join(item.location.members)}]" if item.location.members else ""
        if item.location.offset is not None:
            where += f" @{item.location.offset}"
        lines.append(f"  {label} {item.rule_id}{where} {item.message}")
    if imports and file.imports:
        listed = ", ".join(
            f"{i['module']}.{i['name']} ({i['decision']})"
            for i in sorted(file.imports, key=lambda i: (i["decision"], i["module"], i["name"]))
        )
        lines.append(f"  imports: {listed}")
    return lines


def render_text(
    report: ScanReport,
    min_severity: Severity = Severity.LOW,
    *,
    color: bool = False,
    show_imports: bool = False,
) -> str:
    """Render a report for terminals."""
    lines = [f"picklescope {__version__} - policy {report.policy_name}", ""]
    for file in report.files:
        lines.extend(_file_block(file, min_severity, color, show_imports))
    counts = report.count_by_severity()
    skipped = sum(1 for f in report.files if f.skipped_reason)
    breakdown = ", ".join(f"{name} {counts[name]}" for name in reversed(counts))
    flagged = sum(1 for f in report.files if f.findings_at_least(Severity.MEDIUM))
    lines += [
        "",
        f"{len(report.files)} file(s) scanned, {skipped} skipped, "
        f"{flagged} with medium-or-higher findings",
        f"findings: {breakdown}",
    ]
    return "\n".join(lines)


def to_json(report: ScanReport) -> dict[str, Any]:
    data = report.to_dict()
    data["tool"] = {"name": "picklescope", "version": __version__}
    return data


def render_json(report: ScanReport) -> str:
    return json.dumps(to_json(report), indent=2, sort_keys=False, default=str)
