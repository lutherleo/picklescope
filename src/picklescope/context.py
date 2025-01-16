"""Shared scanning state passed to every format scanner."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from picklescope.findings import Finding, Location, Severity
from picklescope.limits import ScanLimits
from picklescope.policy import Policy, Verdict
from picklescope.rules import finding

BlobScanner = Callable[[bytes, Location, "ScanContext", str], str]
"""Signature of the dispatcher used to scan nested blobs: (data, location, ctx,
name hint) -> detected format name."""


@dataclass(slots=True)
class ImportRecord:
    module: str
    name: str
    decision: str
    count: int = 0
    members: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "name": self.name,
            "decision": self.decision,
            "count": self.count,
            "members": sorted(self.members),
        }


@dataclass(slots=True)
class ScanContext:
    """Mutable state for scanning one file (and everything nested inside it)."""

    policy: Policy
    limits: ScanLimits
    dispatch: BlobScanner | None = None
    depth: int = 0
    findings: list[Finding] = field(default_factory=list)
    imports: dict[tuple[str, str, str], ImportRecord] = field(default_factory=dict)
    _rule_counts: Counter[tuple[str, tuple[str, ...]]] = field(default_factory=Counter)

    def add(self, item: Finding) -> None:
        """Record a finding, collapsing floods of the same rule in the same blob."""
        key = (item.rule_id, item.location.members)
        self._rule_counts[key] += 1
        count = self._rule_counts[key]
        cap = self.limits.max_findings_per_rule
        if count <= cap:
            self.findings.append(item)
        elif count == cap + 1:
            self.findings.append(
                Finding(
                    rule_id=item.rule_id,
                    severity=Severity.INFO,
                    message=f"further {item.rule_id} findings in this blob were suppressed "
                    f"(cap {cap}, see max_findings_per_rule)",
                    location=item.location,
                    details={"suppressed": True},
                )
            )

    def emit(
        self,
        rule_id: str,
        location: Location,
        message: str,
        /,
        *,
        severity: Severity | None = None,
        **details: Any,
    ) -> None:
        self.add(finding(rule_id, location, message, severity=severity, **details))

    def record_import(self, verdict: Verdict, location: Location, count: int = 1) -> None:
        key = (verdict.module, verdict.name, verdict.decision.value)
        record = self.imports.get(key)
        if record is None:
            record = ImportRecord(verdict.module, verdict.name, verdict.decision.value)
            self.imports[key] = record
        record.count += count
        record.members.add("!".join(location.members) or "<file>")

    def nested(self) -> ScanContext:
        """A context one level deeper that shares this context's result lists."""
        return ScanContext(
            policy=self.policy,
            limits=self.limits,
            dispatch=self.dispatch,
            depth=self.depth + 1,
            findings=self.findings,
            imports=self.imports,
            _rule_counts=self._rule_counts,
        )

    def can_nest(self) -> bool:
        return self.depth < self.limits.max_nesting

    def scan_nested(self, data: bytes, location: Location, hint: str) -> str | None:
        """Scan an embedded blob with the format dispatcher, respecting max_nesting."""
        if self.dispatch is None:
            return None
        if not self.can_nest():
            self.emit(
                "PKL116",
                location,
                f"nested blob not scanned: nesting depth limit {self.limits.max_nesting} reached",
            )
            return None
        return self.dispatch(data, location, self.nested(), hint)
