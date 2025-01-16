"""picklescope: a static scanner for ML model files that never deserializes them.

Quick use::

    from picklescope import scan_paths, Severity
    report = scan_paths(["model.bin"])
    if report.exceeds(Severity.HIGH):
        ...
"""

__version__ = "0.1.0"

from picklescope.findings import FileReport, Finding, Location, ScanReport, Severity
from picklescope.limits import ScanLimits
from picklescope.policy import Policy, PolicyError, default_policy, load_policy
from picklescope.scanner import scan_bytes, scan_file, scan_paths

__all__ = [
    "FileReport",
    "Finding",
    "Location",
    "Policy",
    "PolicyError",
    "ScanLimits",
    "ScanReport",
    "Severity",
    "__version__",
    "default_policy",
    "load_policy",
    "scan_bytes",
    "scan_file",
    "scan_paths",
]
