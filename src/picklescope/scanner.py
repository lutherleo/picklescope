"""Top-level scanning: walk paths, detect formats, dispatch to format scanners."""

from __future__ import annotations

import hashlib
import logging
import os
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

from picklescope.blob import Buffer, open_buffer
from picklescope.context import ScanContext
from picklescope.findings import FileReport, Location, ScanReport
from picklescope.formats import archive, gguf, keras, npy, onnx, safetensors
from picklescope.formats.detect import detect_format, extension_mismatch
from picklescope.limits import ScanLimits
from picklescope.pkl.scan import scan_pickle
from picklescope.policy import Policy, default_policy

log = logging.getLogger(__name__)

_HASH_CHUNK = 1024 * 1024


def _dispatch(
    fmt: str,
    buf: Buffer,
    loc: Location,
    ctx: ScanContext,
    base_dir: Path | None,
    zip_source: str | bytes,
    name: str,
) -> str:
    """Run the scanner for ``fmt``; returns the refined format name (zip layouts
    become ``zip:torch``, ``zip:keras`` ...). ``name`` is the file or member name."""
    if fmt == "pickle":
        scan_pickle(buf, loc, ctx, "plain")
    elif fmt == "zip":
        return f"zip:{archive.scan_zip(zip_source, loc, ctx, name)}"
    elif fmt == "safetensors":
        safetensors.scan_safetensors(buf, loc, ctx)
    elif fmt == "gguf":
        gguf.scan_gguf(buf, loc, ctx)
    elif fmt == "onnx":
        onnx.scan_onnx(buf, loc, ctx, base_dir)
    elif fmt == "npy":
        npy.scan_npy(buf, loc, ctx)
    elif fmt == "hdf5":
        keras.scan_h5(buf, loc, ctx)
    elif fmt == "keras-config":
        keras.scan_keras_config(bytes(buf), loc, ctx)
    return fmt


def _scan_nested(data: bytes, loc: Location, ctx: ScanContext, hint: str) -> str:
    """Dispatcher for blobs found inside other files (zip members, npz entries)."""
    return _dispatch(detect_format(data, hint), data, loc, ctx, None, data, hint)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def scan_file(
    path: str | os.PathLike[str],
    policy: Policy | None = None,
    limits: ScanLimits | None = None,
    *,
    display_path: str | None = None,
    hash_file: bool = True,
) -> FileReport:
    """Scan a single file and return its :class:`FileReport`. Never raises for
    content problems; I/O and internal errors become ``GEN901`` findings."""
    file = Path(path)
    shown = display_path or str(path)
    limits = limits or ScanLimits()
    ctx = ScanContext(policy or default_policy(), limits, dispatch=_scan_nested)
    loc = Location(shown)
    try:
        size = file.stat().st_size
    except OSError as exc:
        report = FileReport(shown, "unreadable", 0, skipped_reason=str(exc))
        ctx.emit("GEN901", loc, f"cannot stat file: {exc.strerror or exc}")
        report.findings = ctx.findings
        return report
    report = FileReport(shown, "unknown", size)
    if size > limits.max_file_size:
        report.skipped_reason = "file too large"
        ctx.emit("GEN902", loc, f"{size} bytes exceeds max_file_size {limits.max_file_size}")
        report.findings = ctx.findings
        return report
    try:
        if hash_file:
            report.sha256 = _sha256(file)
        with open_buffer(file) as buf:
            fmt = detect_format(buf, file.name)
            report.format = fmt
            mismatch = extension_mismatch(file.name, fmt)
            if mismatch:
                ctx.emit("GEN900", loc, mismatch, detected=fmt)
            if fmt not in ("zip", "unknown", "empty") and zipfile.is_zipfile(file):
                ctx.emit("GEN903", loc, f"file is a {fmt} and also a valid zip archive")
            if fmt in ("unknown", "empty"):
                report.skipped_reason = f"{fmt} format"
            else:
                report.format = _dispatch(fmt, buf, loc, ctx, file.parent, str(file), file.name)
    except OSError as exc:
        ctx.emit("GEN901", loc, f"I/O error while scanning: {exc.strerror or exc}")
    except Exception as exc:
        log.exception("internal error scanning %s", shown)
        ctx.emit("GEN901", loc, f"internal error ({type(exc).__name__}): {exc}")
    report.findings = ctx.findings
    report.imports = [record.to_dict() for record in ctx.imports.values()]
    return report


def scan_bytes(
    data: bytes,
    name: str = "<bytes>",
    policy: Policy | None = None,
    limits: ScanLimits | None = None,
) -> FileReport:
    """Scan an in-memory blob (no on-disk checks such as ONNX external data)."""
    ctx = ScanContext(policy or default_policy(), limits or ScanLimits(), dispatch=_scan_nested)
    loc = Location(name)
    fmt = detect_format(data, name)
    report = FileReport(name, fmt, len(data), sha256=hashlib.sha256(data).hexdigest())
    mismatch = extension_mismatch(name, fmt)
    if mismatch:
        ctx.emit("GEN900", loc, mismatch, detected=fmt)
    if fmt in ("unknown", "empty"):
        report.skipped_reason = f"{fmt} format"
    else:
        try:
            report.format = _dispatch(fmt, data, loc, ctx, None, data, name)
        except Exception as exc:
            log.exception("internal error scanning %s", name)
            ctx.emit("GEN901", loc, f"internal error ({type(exc).__name__}): {exc}")
    report.findings = ctx.findings
    report.imports = [record.to_dict() for record in ctx.imports.values()]
    return report


def iter_files(
    paths: Iterable[str | os.PathLike[str]], follow_symlinks: bool = False
) -> Iterator[Path]:
    """Expand files and directories (recursively, sorted) into a file list."""
    for raw in paths:
        path = Path(raw)
        if path.is_dir() and (follow_symlinks or not path.is_symlink()):
            for root, dirs, files in os.walk(path, followlinks=follow_symlinks):
                dirs.sort()
                for name in sorted(files):
                    candidate = Path(root) / name
                    if candidate.is_symlink() and not follow_symlinks:
                        log.info("skipping symlink %s", candidate)
                        continue
                    yield candidate
        else:
            yield path


def scan_paths(
    paths: Iterable[str | os.PathLike[str]],
    policy: Policy | None = None,
    limits: ScanLimits | None = None,
    *,
    follow_symlinks: bool = False,
    hash_files: bool = True,
) -> ScanReport:
    """Scan files and directories; returns a :class:`ScanReport`."""
    policy = policy or default_policy()
    report = ScanReport(policy_name=policy.name)
    for file in iter_files(paths, follow_symlinks):
        log.info("scanning %s", file)
        report.files.append(scan_file(file, policy, limits, hash_file=hash_files))
    return report
