"""Zip containers: PyTorch ``torch.save`` archives, Keras v3 ``.keras``, NumPy ``.npz``.

The scanner reads the central directory, applies structural checks (zip-slip, duplicate
names, symlinks, encryption, decompression bombs), classifies the archive, and then
reads only the members that can carry code: pickles, Keras configs, embedded
archives. Tensor storage members are never decompressed.
"""

from __future__ import annotations

import io
import json
import logging
import re
import stat
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import IO

from picklescope.context import ScanContext
from picklescope.findings import Location
from picklescope.formats import keras
from picklescope.pkl.analyzer import Const, GlobalValue, PersistentEvent, TupleValue, describe
from picklescope.pkl.disasm import looks_like_pickle
from picklescope.pkl.scan import scan_pickle

log = logging.getLogger(__name__)

_DRIVE = re.compile(r"^[A-Za-z]:")
_EXECUTABLE_SUFFIXES = frozenset(
    {".py", ".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe", ".sh", ".bat", ".cmd", ".ps1"}
)
_PICKLE_SUFFIXES = (".pkl", ".pickle", ".debug_pkl")
_SNIFF_BYTES = 16
# Keras 3 saving_lib reads config.json first, then whichever weights store is present
_KERAS_MARKERS = frozenset({"metadata.json", "model.weights.h5", "model.weights.npz"})
# zipfile raises RuntimeError for some encrypted entries, the others for damaged data
_READ_ERRORS = (zipfile.BadZipFile, NotImplementedError, OSError, EOFError, ValueError,
                RuntimeError)  # fmt: skip
_MAGIC_SNIFFERS: tuple[tuple[bytes, str], ...] = (
    (b"PK\x03\x04", "zip"),
    (b"\x93NUMPY", "npy"),
    (b"GGUF", "gguf"),
)


@dataclass(slots=True)
class ArchiveLayout:
    """Classified archive. ``infos`` keeps every central-directory entry, duplicates
    included: which copy of a repeated name a loader resolves is implementation
    defined, so each copy is scanned on its own."""

    kind: str  # torch | torchscript | keras | npz | zip
    torch_prefix: str | None = None
    infos: list[zipfile.ZipInfo] = field(default_factory=list)
    names: frozenset[str] = frozenset()

    def labelled(self) -> Iterator[tuple[str, str, zipfile.ZipInfo]]:
        """Yield ``(name, label, info)`` for every entry; ``label`` is the location
        label, ``name#k`` for the k-th copy of a duplicated name."""
        counts = Counter(info.filename for info in self.infos)
        seen: Counter[str] = Counter()
        for info in self.infos:
            name = info.filename
            seen[name] += 1
            yield name, f"{name}#{seen[name]}" if counts[name] > 1 else name, info


def unsafe_member_name(name: str) -> str | None:
    """Explain why an archive member name escapes its extraction root, or None."""
    if name.startswith(("/", "\\")):
        return "absolute path"
    if _DRIVE.match(name):
        return "drive-letter path"
    parts = re.split(r"[\\/]", name)
    if ".." in parts:
        return "parent-directory segment"
    if "\x00" in name:
        return "NUL byte in name"
    return None


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return info.create_system == 3 and stat.S_ISLNK(mode)


def _structural_checks(infos: list[zipfile.ZipInfo], loc: Location, ctx: ScanContext) -> None:
    counts = Counter(info.filename for info in infos)
    for name, count in counts.items():
        if count > 1:
            ctx.emit(
                "ZIP201", loc.child(name), f"member {name!r} appears {count} times", count=count
            )
    for info in infos:
        member = loc.child(info.filename)
        reason = unsafe_member_name(info.filename)
        if reason:
            ctx.emit("ZIP200", member, f"member name {info.filename!r} is unsafe: {reason}")
        if info.flag_bits & 0x1:
            ctx.emit("ZIP203", member, f"member {info.filename!r} is encrypted")
        if _is_symlink(info):
            ctx.emit("ZIP204", member, f"member {info.filename!r} is a symbolic link")
        ratio = info.file_size / info.compress_size if info.compress_size else 0.0
        if info.file_size > ctx.limits.max_member_size:
            ctx.emit(
                "ZIP202",
                member,
                f"member declares {info.file_size} bytes, above max_member_size",
                declared_size=info.file_size,
            )
        elif ratio > ctx.limits.max_compression_ratio and info.file_size > 1024**2:
            ctx.emit(
                "ZIP202",
                member,
                f"compression ratio {ratio:.0f}:1 exceeds {ctx.limits.max_compression_ratio:.0f}:1",
                ratio=round(ratio, 1),
            )


def classify(infos: Iterable[zipfile.ZipInfo], filename: str | None = None) -> ArchiveLayout:
    """Recognise the archive flavour from member names and, for Keras, the file name.

    A root ``config.json`` next to a Keras marker, or in a file named ``*.keras``, is a
    Keras archive. :func:`scan_zip` also promotes a generic zip whose root
    ``config.json`` parses as a Keras object, since Keras loads that file before it
    looks for any weights.
    """
    entries = list(infos)
    names = frozenset(info.filename for info in entries)
    for name in names:
        path = PurePosixPath(name)
        if path.name in ("data.pkl", "constants.pkl") and len(path.parts) == 2:
            prefix = path.parts[0]
            kind = "torchscript" if any(n.startswith(f"{prefix}/code/") for n in names) else "torch"
            if kind == "torch" and path.name == "constants.pkl":
                continue
            return ArchiveLayout(kind, prefix, entries, names)
    keras_name = filename is not None and PurePosixPath(filename).suffix.lower() == ".keras"
    if "config.json" in names and (keras_name or _KERAS_MARKERS & names):
        return ArchiveLayout("keras", None, entries, names)
    if names and all(n.endswith(".npy") for n in names if not n.endswith("/")):
        return ArchiveLayout("npz", None, entries, names)
    return ArchiveLayout("zip", None, entries, names)


def _read_member(
    zf: zipfile.ZipFile, info: zipfile.ZipInfo, member: Location, ctx: ScanContext, limit: int
) -> bytes | None:
    if info.flag_bits & 0x1 or info.file_size > ctx.limits.max_member_size:
        return None
    try:
        with zf.open(info) as handle:
            data = handle.read(limit + 1)
    except _READ_ERRORS as exc:
        ctx.emit("ZIP205", member, f"cannot read member {info.filename!r}: {exc}")
        return None
    if len(data) > limit:
        ctx.emit("ZIP202", member, f"member {info.filename!r} inflates beyond {limit} bytes")
        return None
    return data


def _read_head(
    zf: zipfile.ZipFile, info: zipfile.ZipInfo, member: Location, ctx: ScanContext
) -> bytes | None:
    """First bytes of a member, for content sniffing (no size-limit findings)."""
    if info.flag_bits & 0x1 or info.file_size > ctx.limits.max_member_size:
        return None
    try:
        with zf.open(info) as handle:
            return handle.read(_SNIFF_BYTES)
    except _READ_ERRORS as exc:
        ctx.emit("ZIP205", member, f"cannot read member {info.filename!r}: {exc}")
        return None


def _is_keras_config(zf: zipfile.ZipFile, info: zipfile.ZipInfo, ctx: ScanContext) -> bool:
    """Quietly read a root ``config.json`` and report whether it is a Keras object.
    Read errors are left for the member scan to report."""
    if info.flag_bits & 0x1 or info.file_size > ctx.limits.max_member_size:
        return False
    try:
        with zf.open(info) as handle:
            data = handle.read(ctx.limits.max_member_size + 1)
        if len(data) > ctx.limits.max_member_size:
            return False
        return keras.looks_like_keras_object(json.loads(data.decode("utf-8")))
    except (*_READ_ERRORS, UnicodeDecodeError, RecursionError):  # JSONDecodeError is a ValueError
        return False


def _sniff(data: bytes) -> str | None:
    for magic, kind in _MAGIC_SNIFFERS:
        if data.startswith(magic):
            return kind
    if looks_like_pickle(data):
        return "pickle"
    return None


def _storage_key(pid: object) -> tuple[str | None, str]:
    """Extract (key, description) from a torch persistent id tuple."""
    if not isinstance(pid, TupleValue) or len(pid.items) < 3:
        return None, describe(pid)  # type: ignore[arg-type]
    tag, _storage_type, key = pid.items[:3]
    if not (isinstance(tag, Const) and tag.value == "storage"):
        return None, describe(pid)
    if isinstance(key, Const) and isinstance(key.value, str):
        return key.value, describe(pid)
    return None, describe(pid)


def _check_torch_storages(
    pids: list[PersistentEvent], layout: ArchiveLayout, pkl_loc: Location, ctx: ScanContext
) -> None:
    prefix = layout.torch_prefix
    for event in pids:
        key, text = _storage_key(event.pid)
        here = pkl_loc.at(event.offset)
        if key is None:
            ctx.emit("ZIP207", here, f"persistent id is not a torch storage reference: {text}")
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key) or key in (".", ".."):
            ctx.emit("ZIP207", here, f"storage key {key!r} is not a plain record name")
            continue
        if f"{prefix}/data/{key}" not in layout.names:
            ctx.emit("ZIP207", here, f"storage key {key!r} has no {prefix}/data/{key} record")
        storage_type = event.pid.items[1] if isinstance(event.pid, TupleValue) else None
        if storage_type is not None and not isinstance(storage_type, GlobalValue):
            ctx.emit("ZIP207", here, f"storage type is not an imported class: {text}")


def _scan_members(
    zf: zipfile.ZipFile, layout: ArchiveLayout, loc: Location, ctx: ScanContext
) -> None:
    limit = ctx.limits.max_member_size
    code_members = 0
    for name, label, info in layout.labelled():
        if name.endswith("/"):
            continue
        member = loc.child(label)
        suffix = PurePosixPath(name).suffix.lower()
        is_torch = layout.kind in ("torch", "torchscript")
        if is_torch and layout.torch_prefix and name.startswith(f"{layout.torch_prefix}/data/"):
            continue  # raw tensor storage: never decompressed
        if layout.kind == "torchscript" and name.startswith(f"{layout.torch_prefix}/code/"):
            if suffix == ".py":
                code_members += 1
                continue
        elif suffix in _EXECUTABLE_SUFFIXES:
            ctx.emit("ZIP208", member, f"archive embeds {suffix} file {name!r}")
            continue

        if name.lower().endswith(_PICKLE_SUFFIXES):
            data = _read_member(zf, info, member, ctx, limit)
            if data is None:
                continue
            result = scan_pickle(data, member, ctx, "torch" if is_torch else "plain")
            if is_torch and result.persistent_ids:
                _check_torch_storages(result.persistent_ids, layout, member, ctx)
            continue
        if layout.kind == "keras" and name == "config.json":
            data = _read_member(zf, info, member, ctx, limit)
            if data is not None:
                keras.scan_keras_config(data, member, ctx)
            continue
        if (
            suffix in (".h5", ".hdf5", ".json", ".txt", ".vocab", ".model")
            and layout.kind == "keras"
        ):
            continue  # weights and assets: data only
        head = _read_head(zf, info, member, ctx)
        if head is None:
            continue
        kind = _sniff(head)
        if kind is None:
            continue
        data = _read_member(zf, info, member, ctx, limit)
        if data is None:
            continue
        if kind == "pickle":
            scan_pickle(data, member, ctx, "plain")
        else:
            ctx.scan_nested(data, member, name)
    if code_members:
        ctx.emit(
            "ZIP206",
            loc.child(f"{layout.torch_prefix}/code/"),
            f"TorchScript archive contains {code_members} code file(s) compiled by torch.jit.load",
            code_files=code_members,
        )


def scan_zip(
    source: str | IO[bytes] | bytes, loc: Location, ctx: ScanContext, filename: str | None = None
) -> str:
    """Scan a zip archive given as a path, a binary file object or raw bytes.

    ``filename`` (the file or member name, when known) lets a ``.keras`` extension
    select the Keras layout.

    Returns the detected layout (``torch``, ``torchscript``, ``keras``, ``npz`` or
    ``zip``), or ``zip`` when the archive cannot be opened.
    """
    handle: str | IO[bytes] = io.BytesIO(source) if isinstance(source, bytes) else source
    try:
        zf = zipfile.ZipFile(handle)
    except _READ_ERRORS as exc:
        ctx.emit("ZIP205", loc, f"cannot open zip archive: {exc}")
        return "zip"
    with zf:
        infos = zf.infolist()
        _structural_checks(infos, loc, ctx)
        layout = classify(infos, filename)
        if layout.kind == "zip" and any(
            info.filename == "config.json" and _is_keras_config(zf, info, ctx) for info in infos
        ):
            layout.kind = "keras"
        log.debug("%s: zip layout %s (%d members)", loc.display(), layout.kind, len(infos))
        if layout.kind == "npz":
            for name, label, info in layout.labelled():
                member = loc.child(label)
                data = _read_member(zf, info, member, ctx, ctx.limits.max_member_size)
                if data is not None:
                    ctx.scan_nested(data, member, name)
        else:
            _scan_members(zf, layout, loc, ctx)
    return layout.kind
