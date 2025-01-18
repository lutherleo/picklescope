"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from picklescope.context import ScanContext
from picklescope.findings import FileReport, Location
from picklescope.limits import ScanLimits
from picklescope.policy import default_policy
from picklescope.scanner import _scan_nested, scan_file

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


def make_ctx(limits: ScanLimits | None = None, **kwargs: object) -> ScanContext:
    return ScanContext(default_policy(), limits or ScanLimits(), dispatch=_scan_nested, **kwargs)  # type: ignore[arg-type]


def rules_of(report_or_ctx: FileReport | ScanContext) -> set[str]:
    return {f.rule_id for f in report_or_ctx.findings}


def scan_fixture(rel: str, **kwargs: object) -> FileReport:
    return scan_file(FIXTURES / rel, display_path=rel, hash_file=False, **kwargs)  # type: ignore[arg-type]


LOC = Location("test")
