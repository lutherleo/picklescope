"""Core result types: severities, locations, findings and per-file reports."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Severity(enum.IntEnum):
    """Ordered finding severity. Higher values are worse."""

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def parse(cls, text: str) -> Severity:
        """Parse a case-insensitive severity name (``"high"`` -> ``Severity.HIGH``)."""
        try:
            return cls[text.strip().upper()]
        except KeyError:
            names = ", ".join(s.name.lower() for s in cls)
            raise ValueError(f"unknown severity {text!r} (expected one of: {names})") from None

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True, slots=True)
class Location:
    """Where a finding was observed.

    ``path`` is the file on disk as given to the scanner. ``members`` is the chain of
    container members leading to the scanned blob (for example
    ``("archive/data.pkl",)`` for a pickle inside a PyTorch zip, or
    ``("archive/data.pkl", "nested@0x1f")`` for a pickle embedded in that pickle).
    ``offset`` is a byte offset inside the innermost blob, when meaningful.
    """

    path: str
    members: tuple[str, ...] = ()
    offset: int | None = None

    def child(self, member: str) -> Location:
        """Return the location of a member nested inside this location."""
        return Location(self.path, (*self.members, member), None)

    def at(self, offset: int | None) -> Location:
        """Return this location pinned to a byte offset."""
        return Location(self.path, self.members, offset)

    def display(self) -> str:
        text = self.path
        for member in self.members:
            text += f"!{member}"
        if self.offset is not None:
            text += f"@{self.offset}"
        return text


@dataclass(frozen=True, slots=True)
class Finding:
    """A single scanner observation tied to a rule."""

    rule_id: str
    severity: Severity
    message: str
    location: Location
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.label,
            "message": self.message,
            "path": self.location.path,
            "members": list(self.location.members),
            "offset": self.location.offset,
            "details": self.details,
        }


@dataclass(slots=True)
class FileReport:
    """Everything the scanner learned about one file on disk."""

    path: str
    format: str
    size: int
    sha256: str | None = None
    findings: list[Finding] = field(default_factory=list)
    imports: list[dict[str, Any]] = field(default_factory=list)
    skipped_reason: str | None = None

    @property
    def max_severity(self) -> Severity | None:
        if not self.findings:
            return None
        return max(f.severity for f in self.findings)

    def findings_at_least(self, threshold: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity >= threshold]

    def to_dict(self) -> dict[str, Any]:
        top = self.max_severity
        return {
            "path": self.path,
            "format": self.format,
            "size": self.size,
            "sha256": self.sha256,
            "skipped_reason": self.skipped_reason,
            "max_severity": top.label if top is not None else None,
            "findings": [f.to_dict() for f in self.findings],
            "imports": self.imports,
        }


@dataclass(slots=True)
class ScanReport:
    """Aggregate result for a scan invocation."""

    files: list[FileReport] = field(default_factory=list)
    policy_name: str = "default"

    @property
    def findings(self) -> list[Finding]:
        return [f for report in self.files for f in report.findings]

    def count_by_severity(self) -> dict[str, int]:
        counts = {s.label: 0 for s in Severity}
        for finding in self.findings:
            counts[finding.severity.label] += 1
        return counts

    def exceeds(self, threshold: Severity) -> bool:
        return any(f.severity >= threshold for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy_name,
            "summary": {
                "files": len(self.files),
                "skipped": sum(1 for f in self.files if f.skipped_reason),
                "findings": len(self.findings),
                "by_severity": self.count_by_severity(),
            },
            "files": [f.to_dict() for f in self.files],
        }
