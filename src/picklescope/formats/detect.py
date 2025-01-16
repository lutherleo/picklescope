"""Content-based format detection with extension hints."""

from __future__ import annotations

import json
from pathlib import PurePath

from picklescope.blob import Buffer
from picklescope.formats.keras import looks_like_keras_config
from picklescope.formats.onnx import looks_like_onnx
from picklescope.formats.safetensors import looks_like_safetensors
from picklescope.pkl.disasm import looks_like_pickle
from picklescope.pkl.opcodes import OPCODES

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
KNOWN_FORMATS = ("zip", "gguf", "npy", "hdf5", "safetensors", "pickle", "onnx", "keras-config")

EXTENSION_FALLBACK = {".safetensors": "safetensors", ".gguf": "gguf", ".npy": "npy",
                      ".onnx": "onnx"}  # fmt: skip
PICKLE_EXTENSIONS = frozenset({".pkl", ".pickle", ".pt", ".pth", ".bin", ".ckpt", ".dat"})

# extension -> formats it may legitimately contain, and whether "unknown" content is
# itself suspicious (strict) for that extension.
EXPECTED: dict[str, tuple[frozenset[str], bool]] = {
    ".safetensors": (frozenset({"safetensors"}), True),
    ".gguf": (frozenset({"gguf"}), True),
    ".onnx": (frozenset({"onnx"}), True),
    ".npy": (frozenset({"npy"}), True),
    ".npz": (frozenset({"zip"}), True),
    ".keras": (frozenset({"zip"}), True),
    ".h5": (frozenset({"hdf5"}), True),
    ".hdf5": (frozenset({"hdf5"}), True),
    ".pkl": (frozenset({"pickle"}), False),
    ".pickle": (frozenset({"pickle"}), False),
    ".pt": (frozenset({"zip", "pickle"}), False),
    ".pth": (frozenset({"zip", "pickle"}), False),
    ".ckpt": (frozenset({"zip", "pickle"}), False),
    ".bin": (frozenset({"zip", "pickle"}), False),
}


def _first_opcode_plausible(buf: Buffer) -> bool:
    """Protocol 0/1 pickles have no prologue; require a valid first opcode that is
    not STOP and a STOP byte somewhere in the stream."""
    return len(buf) >= 2 and buf[0] in OPCODES and buf[0] != 0x2E and buf.find(b".") > 0


def _json_keras(buf: Buffer) -> bool:
    if len(buf) > 64 * 1024**2 or buf[:1] not in (b"{", b" ", b"\n"):
        return False
    try:
        return looks_like_keras_config(json.loads(bytes(buf).decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        return False


def detect_format(buf: Buffer, name: str = "") -> str:
    """Return one of :data:`KNOWN_FORMATS`, ``"empty"`` or ``"unknown"``."""
    if len(buf) == 0:
        return "empty"
    head = buf[:8]
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        return "zip"
    if head.startswith(b"GGUF"):
        return "gguf"
    if head.startswith(b"\x93NUMPY"):
        return "npy"
    if head == HDF5_MAGIC:
        return "hdf5"
    if looks_like_safetensors(buf):
        return "safetensors"
    if looks_like_pickle(buf):
        return "pickle"
    suffix = PurePath(name).suffix.lower()
    if suffix == ".onnx" and looks_like_onnx(buf):
        return "onnx"
    if suffix in PICKLE_EXTENSIONS and _first_opcode_plausible(buf):
        return "pickle"
    if suffix == ".json" and _json_keras(buf):
        return "keras-config"
    # A strongly typed extension routes damaged content to its validator, which explains
    # precisely what is wrong (e.g. a safetensors length prefix past the end of file).
    return EXTENSION_FALLBACK.get(suffix, "unknown")


def extension_mismatch(name: str, detected: str) -> str | None:
    """Explain an extension/content mismatch, or return None if consistent."""
    suffix = PurePath(name).suffix.lower()
    expected = EXPECTED.get(suffix)
    if expected is None or detected == "empty":
        return None
    formats, strict = expected
    if detected in formats:
        return None
    if detected == "unknown" and not strict:
        return None
    wanted = "/".join(sorted(formats))
    return f"extension {suffix} implies {wanted} but the content is {detected}"
