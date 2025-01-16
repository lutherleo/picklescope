"""SARIF 2.1.0 rendering.

Each finding becomes a ``result``. The physical location is the scanned file; findings
inside archive members or nested blobs additionally carry a ``logicalLocation`` whose
fully-qualified name is the member chain (``archive/data.pkl!embedded@42``). Byte
regions are emitted only for offsets in the top-level file, because member offsets do
not address bytes of the file on disk. Rules carry GitHub's ``security-severity``
property so code-scanning ranks alerts correctly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePath
from typing import Any
from urllib.parse import quote

from picklescope import __version__
from picklescope.findings import Finding, ScanReport, Severity
from picklescope.rules import RULES, Rule

SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"

_LEVEL = {
    Severity.CRITICAL: "error",
    Severity.HIGH: "error",
    Severity.MEDIUM: "warning",
    Severity.LOW: "note",
    Severity.INFO: "note",
}
_SECURITY_SEVERITY = {
    Severity.CRITICAL: "9.5",
    Severity.HIGH: "8.0",
    Severity.MEDIUM: "5.0",
    Severity.LOW: "2.0",
    Severity.INFO: "0.0",
}
_CATEGORY_TAGS = {
    "PKL": "pickle",
    "ZIP": "archive",
    "ST": "safetensors",
    "GG": "gguf",
    "OX": "onnx",
    "KR": "keras",
    "NP": "numpy",
    "GEN": "general",
}


def _tag(rule_id: str) -> str:
    prefix = rule_id.rstrip("0123456789")
    return _CATEGORY_TAGS.get(prefix, "general")


def _rule_descriptor(rule: Rule) -> dict[str, Any]:
    return {
        "id": rule.id,
        "name": rule.name,
        "shortDescription": {"text": rule.summary},
        "fullDescription": {"text": rule.rationale},
        "help": {
            "text": f"{rule.summary} {rule.rationale}",
            "markdown": f"**{rule.summary}**\n\n{rule.rationale}\n\n"
            "See docs/RULES.md in the picklescope repository.",
        },
        "defaultConfiguration": {"level": _LEVEL[rule.severity]},
        "properties": {
            "security-severity": _SECURITY_SEVERITY[rule.severity],
            "tags": ["security", "ml-supply-chain", _tag(rule.id)],
            "precision": "high" if rule.severity >= Severity.HIGH else "medium",
        },
    }


def _artifact_uri(path: str) -> tuple[str, str | None]:
    pure = PurePath(path)
    if pure.is_absolute():
        return Path(path).as_uri(), None
    return quote(pure.as_posix()), "%SRCROOT%"


def _fingerprint(item: Finding) -> str:
    key = "|".join(
        [item.rule_id, item.location.path, "!".join(item.location.members), item.message]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _result(item: Finding, rule_index: dict[str, int]) -> dict[str, Any]:
    uri, base = _artifact_uri(item.location.path)
    artifact: dict[str, Any] = {"uri": uri}
    if base:
        artifact["uriBaseId"] = base
    physical: dict[str, Any] = {"artifactLocation": artifact}
    if not item.location.members and item.location.offset is not None:
        physical["region"] = {"byteOffset": item.location.offset}
    location: dict[str, Any] = {"physicalLocation": physical}
    if item.location.members:
        name = "!".join(item.location.members)
        if item.location.offset is not None:
            name += f"@{item.location.offset}"
        location["logicalLocations"] = [{"fullyQualifiedName": name, "kind": "resource"}]
    return {
        "ruleId": item.rule_id,
        "ruleIndex": rule_index[item.rule_id],
        "level": _LEVEL[item.severity],
        "message": {"text": item.message},
        "locations": [location],
        "partialFingerprints": {"picklescopeFinding/v1": _fingerprint(item)},
        "properties": {"severity": item.severity.label, "details": item.details},
    }


def to_sarif(report: ScanReport, min_severity: Severity = Severity.LOW) -> dict[str, Any]:
    """Render ``report`` as a SARIF 2.1.0 log (a JSON-serialisable dict)."""
    rules = list(RULES.values())
    rule_index = {rule.id: index for index, rule in enumerate(rules)}
    results = [
        _result(item, rule_index) for item in report.findings if item.severity >= min_severity
    ]
    artifacts = []
    for file in report.files:
        uri, base = _artifact_uri(file.path)
        entry: dict[str, Any] = {"location": {"uri": uri}, "length": file.size}
        if base:
            entry["location"]["uriBaseId"] = base
        if file.sha256:
            entry["hashes"] = {"sha-256": file.sha256}
        artifacts.append(entry)
    return {
        "$schema": SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "picklescope",
                        "version": __version__,
                        "semanticVersion": __version__,
                        "rules": [_rule_descriptor(rule) for rule in rules],
                    }
                },
                "artifacts": artifacts,
                "results": results,
                "invocations": [
                    {
                        "executionSuccessful": True,
                        "properties": {"policy": report.policy_name},
                    }
                ],
            }
        ],
    }
