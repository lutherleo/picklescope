"""Resource limits that keep the scanner safe against hostile inputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScanLimits:
    """Upper bounds applied while scanning.

    Every parser in picklescope reads untrusted bytes; these limits bound CPU, memory
    and recursion regardless of what length fields inside a file claim.
    """

    max_file_size: int = 4 * 1024**3
    """Files larger than this are reported (GEN902) and skipped."""

    max_member_size: int = 1024**3
    """Largest archive member that is decompressed for inspection."""

    max_compression_ratio: float = 200.0
    """Members with uncompressed/compressed above this ratio are flagged (ZIP202)."""

    max_pickle_ops: int = 20_000_000
    """Opcodes analysed per pickle stream before PKL116 is raised."""

    max_stack_depth: int = 1_000_000
    """Abstract stack size before PKL116."""

    max_memo_entries: int = 1_000_000
    """Memo slots tracked per pickle before PKL116."""

    max_nesting: int = 4
    """Depth of recursive scans (pickle in bytes in pickle in zip ...)."""

    max_pickles_per_stream: int = 64
    """Consecutive pickles followed in one stream (legacy torch uses five)."""

    max_header_bytes: int = 100 * 1024**2
    """Largest safetensors JSON header accepted (matches the reference loader)."""

    max_gguf_array: int = 10_000_000
    """Largest GGUF metadata array element count examined."""

    max_protobuf_depth: int = 64
    """Deepest ONNX graph/message nesting followed."""

    max_findings_per_rule: int = 50
    """Identical-rule findings per blob beyond this are summarised into one."""

    def __post_init__(self) -> None:
        for name in (
            "max_file_size",
            "max_member_size",
            "max_pickle_ops",
            "max_stack_depth",
            "max_memo_entries",
            "max_nesting",
            "max_pickles_per_stream",
            "max_header_bytes",
            "max_gguf_array",
            "max_protobuf_depth",
            "max_findings_per_rule",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_compression_ratio <= 1:
            raise ValueError("max_compression_ratio must be greater than 1")
