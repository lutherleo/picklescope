"""Byte-source helpers: read-only memory maps for files, plain bytes for members."""

from __future__ import annotations

import mmap
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

Buffer = bytes | mmap.mmap
"""Anything the parsers accept: supports len(), slicing, indexing, find() and
struct.unpack_from()."""


@contextmanager
def open_buffer(path: str | os.PathLike[str]) -> Iterator[Buffer]:
    """Map ``path`` read-only; empty files yield ``b""`` (mmap rejects length 0)."""
    file = Path(path)
    with file.open("rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        if size == 0:
            yield b""
            return
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            yield mapped
        finally:
            mapped.close()
