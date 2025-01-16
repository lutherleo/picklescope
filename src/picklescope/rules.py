"""Catalog of every rule the scanner can emit.

Rule ids are stable: ``PKL`` pickle bytecode, ``ZIP`` archive containers (PyTorch,
Keras, NumPy ``.npz``), ``ST`` safetensors, ``GG`` GGUF, ``OX`` ONNX, ``KR`` Keras
configs, ``NP`` NumPy ``.npy``, ``GEN`` format-independent checks. The catalog feeds
SARIF ``tool.driver.rules`` and the ``picklescope rules`` command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from picklescope.findings import Finding, Location, Severity


@dataclass(frozen=True, slots=True)
class Rule:
    id: str
    name: str
    severity: Severity
    summary: str
    rationale: str


def _r(rule_id: str, name: str, severity: Severity, summary: str, rationale: str) -> Rule:
    return Rule(rule_id, name, severity, summary, rationale)


C, H, M, L = Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW

_RULES: tuple[Rule, ...] = (
    # --- pickle bytecode -------------------------------------------------------------
    _r(
        "PKL100",
        "dangerous-import",
        C,
        "Pickle imports a callable on the deny-list.",
        "Unpickling resolves GLOBAL/STACK_GLOBAL/INST through find_class, which imports the "
        "module and fetches the attribute. Deny-listed names (process execution, code "
        "evaluation, import machinery, file/network access, nested deserializers, "
        "attribute-access gadgets) give an attacker code execution or an equivalent "
        "primitive.",
    ),
    _r(
        "PKL101",
        "unknown-import",
        M,
        "Pickle imports a name that is not on the allow-list.",
        "Importing a module runs its top-level code, and the imported object may be "
        "invoked later through REDUCE/BUILD. Names outside the allow-list need review.",
    ),
    _r(
        "PKL102",
        "unknown-callable-invoked",
        H,
        "Pickle invokes a callable that is not on the allow-list.",
        "REDUCE, NEWOBJ, NEWOBJ_EX, OBJ and INST call the object they reference. Calling an "
        "unreviewed callable with attacker-chosen arguments is the core pickle exploit.",
    ),
    _r(
        "PKL103",
        "dynamic-import",
        H,
        "STACK_GLOBAL operands are not compile-time string constants.",
        "When the module or attribute name is computed at runtime the scanner cannot tell "
        "what is imported; benign serializers always emit literal strings.",
    ),
    _r(
        "PKL104",
        "attribute-traversal",
        H,
        "Imported name walks through a dunder attribute chain.",
        "Protocol 4 find_class resolves dotted names attribute by attribute, so a name such "
        "as 'Foo.__init__.__globals__' reaches arbitrary objects from an allowed module.",
    ),
    _r(
        "PKL105",
        "extension-registry",
        H,
        "Pickle uses EXT1/EXT2/EXT4 extension-registry opcodes.",
        "Extension codes are resolved through copyreg's process-global registry, so the "
        "imported callable cannot be determined statically.",
    ),
    _r(
        "PKL106",
        "indirect-call",
        H,
        "A runtime-computed value is called.",
        "REDUCE on the result of another call (for example getattr(...)(...)) invokes a "
        "callable chosen at load time. Benign pickles call imported names directly.",
    ),
    _r(
        "PKL107",
        "global-mutation",
        H,
        "BUILD/SETITEM/APPEND targets an imported module-level object.",
        "Applying state to a class or module object rather than to a fresh instance "
        "mutates interpreter-global state, a known pickle persistence and hijack trick.",
    ),
    _r(
        "PKL108",
        "persistent-load",
        M,
        "Pickle uses PERSID/BINPERSID outside a recognised container.",
        "Persistent IDs are resolved by the loader's persistent_load hook; the behaviour "
        "depends entirely on application code the scanner cannot see.",
    ),
    _r(
        "PKL109",
        "nested-pickle",
        M,
        "A bytes constant inside the pickle is itself a pickle.",
        "Embedded pickles are usually unpacked by a later loads() call; the inner stream is "
        "scanned recursively and its findings are reported with a nested location.",
    ),
    _r(
        "PKL110",
        "multiple-pickles",
        M,
        "The stream contains more pickles after the first STOP.",
        "Loaders that call load() repeatedly execute every pickle in the stream, while "
        "naive scanners stop at the first STOP opcode.",
    ),
    _r(
        "PKL111",
        "trailing-data",
        L,
        "Unparsed bytes follow the final STOP opcode.",
        "Trailing data is ignored by pickle.loads but may hide a second payload or indicate "
        "a polyglot file.",
    ),
    _r(
        "PKL112",
        "malformed-pickle",
        M,
        "The pickle stream is truncated or structurally invalid.",
        "Malformed streams cause differences between scanners and loaders; the part before "
        "the error is still analysed.",
    ),
    _r(
        "PKL113",
        "unknown-opcode",
        H,
        "An opcode byte that no pickle protocol defines was found.",
        "The stream cannot be analysed past this point.",
    ),
    _r(
        "PKL114",
        "unsupported-protocol",
        H,
        "PROTO announces a protocol newer than 5.",
        "No released Python understands it; the file was crafted or corrupted.",
    ),
    _r(
        "PKL115",
        "malformed-global-name",
        H,
        "GLOBAL/INST/STACK_GLOBAL names are not valid dotted identifiers.",
        "Odd characters, empty segments or relative imports in a global name indicate "
        "tampering or an attempt to confuse name-based filters.",
    ),
    _r(
        "PKL116",
        "resource-limit",
        M,
        "Analysis stopped at a configured resource limit.",
        "Opcode, stack, memo or nesting limits protect the scanner from hostile inputs; "
        "the stream may contain more content than was analysed.",
    ),
    # --- archive containers ----------------------------------------------------------
    _r(
        "ZIP200",
        "path-traversal",
        H,
        "Archive member name escapes the extraction directory.",
        "Absolute names, drive letters or '..' segments let extraction write outside the "
        "target directory (zip-slip).",
    ),
    _r(
        "ZIP201",
        "duplicate-member",
        M,
        "Archive contains two members with the same name.",
        "Different readers pick different copies, so a scanner and a loader can see "
        "different content. Every copy is scanned; findings in the k-th copy are "
        "located at `name#k`.",
    ),
    _r(
        "ZIP202",
        "decompression-bomb",
        M,
        "Member compression ratio or size exceeds limits.",
        "Highly compressed members exhaust memory or disk when extracted.",
    ),
    _r(
        "ZIP203",
        "encrypted-member",
        M,
        "Archive member is encrypted.",
        "Encrypted content cannot be inspected; ML archives are never encrypted.",
    ),
    _r(
        "ZIP204",
        "symlink-member",
        H,
        "Archive member is a symbolic link.",
        "Symlinks in extracted archives redirect later writes or reads outside the tree.",
    ),
    _r(
        "ZIP205",
        "malformed-archive",
        M,
        "The zip structure could not be read.",
        "Damaged central directories are interpreted differently by different readers.",
    ),
    _r(
        "ZIP206",
        "torchscript-code",
        M,
        "PyTorch archive contains TorchScript source code.",
        "torch.jit.load compiles and runs the code/ directory; it is a program, not data.",
    ),
    _r(
        "ZIP207",
        "missing-storage",
        M,
        "A persistent storage reference has no matching archive record.",
        "PyTorch resolves storages by key; dangling or path-like keys are inconsistent "
        "with torch.save output.",
    ),
    _r(
        "ZIP208",
        "embedded-executable",
        M,
        "Archive embeds source code or native binaries.",
        "Python sources, shared libraries and scripts inside model archives are a common "
        "way to smuggle code next to weights.",
    ),
    # --- safetensors -----------------------------------------------------------------
    _r(
        "ST300",
        "header-size-invalid",
        M,
        "Header length prefix is zero, too large or beyond the file.",
        "The 8-byte little-endian prefix must describe a header that fits in the file and "
        "stays under the 100 MB limit used by the reference implementation.",
    ),
    _r(
        "ST301",
        "header-not-json",
        M,
        "Header is not a UTF-8 JSON object.",
        "Only a JSON object is valid; parsers disagree on anything else.",
    ),
    _r(
        "ST302",
        "invalid-dtype",
        M,
        "Tensor dtype is not a safetensors dtype.",
        "Unknown dtypes are rejected by the reference loader.",
    ),
    _r(
        "ST303",
        "invalid-shape",
        M,
        "Tensor shape is not a list of non-negative integers.",
        "Malformed shapes break size computations in loaders.",
    ),
    _r(
        "ST304",
        "offset-size-mismatch",
        M,
        "Tensor byte range does not match dtype size times element count.",
        "Mismatched ranges let a reader access memory outside the intended tensor.",
    ),
    _r(
        "ST305",
        "overlap-or-gap",
        M,
        "Tensor byte ranges overlap or leave holes.",
        "The format requires tensors to tile the data buffer exactly; overlaps alias data "
        "and holes can hide content.",
    ),
    _r(
        "ST306",
        "duplicate-key",
        M,
        "Header JSON contains duplicate keys.",
        "JSON parsers keep either the first or the last duplicate, a classic parser differential.",
    ),
    _r(
        "ST307",
        "bad-metadata",
        L,
        "__metadata__ is not a string-to-string map.",
        "The specification restricts metadata to string values.",
    ),
    _r(
        "ST308",
        "data-length-mismatch",
        M,
        "Data section length does not match the tensor table.",
        "Truncated files or appended payloads (polyglots) change the byte buffer size.",
    ),
    # --- GGUF --------------------------------------------------------------------------
    _r(
        "GG400",
        "malformed-header",
        M,
        "GGUF header or key/value section is truncated or invalid.",
        "Readers that trust truncated headers read out of bounds.",
    ),
    _r(
        "GG401",
        "unsupported-version",
        M,
        "GGUF version is not 1, 2 or 3.",
        "Unknown versions change field widths.",
    ),
    _r(
        "GG402",
        "implausible-count",
        H,
        "Tensor or metadata count cannot fit in the file.",
        "Huge counts drive allocations sized from attacker input (llama.cpp has had "
        "several heap overflows of this shape).",
    ),
    _r(
        "GG403",
        "string-out-of-bounds",
        H,
        "A length-prefixed string or array extends beyond the file.",
        "Length fields are the primary out-of-bounds read vector in GGUF parsers.",
    ),
    _r(
        "GG404",
        "invalid-value-type",
        M,
        "Metadata value type id is unknown.",
        "Unknown types make the remainder of the header unparseable.",
    ),
    _r(
        "GG405",
        "duplicate-key",
        M,
        "Metadata key appears more than once.",
        "Readers disagree on which value wins.",
    ),
    _r(
        "GG406",
        "invalid-tensor-dims",
        H,
        "Tensor dimension count or sizes are invalid or overflow.",
        "ggml supports at most 4 dimensions; overflowing element counts corrupt size arithmetic.",
    ),
    _r(
        "GG407",
        "unknown-tensor-type",
        M,
        "Tensor uses an unknown ggml type id.",
        "The data size cannot be validated.",
    ),
    _r(
        "GG408",
        "tensor-out-of-bounds",
        H,
        "Tensor data extends beyond the end of the file.",
        "Readers that map tensors by offset would read past the buffer.",
    ),
    _r(
        "GG409",
        "misaligned-tensor",
        M,
        "Tensor offset or general.alignment is invalid.",
        "Offsets must be multiples of a power-of-two alignment.",
    ),
    _r(
        "GG410",
        "tensor-overlap",
        M,
        "Two tensors share bytes.",
        "Overlapping tensors alias each other's data.",
    ),
    _r(
        "GG411",
        "chat-template-injection",
        H,
        "Chat template contains Jinja constructs that reach Python internals.",
        "Templates are rendered by Jinja; dunder access such as __class__ or __globals__ "
        "is the server-side template injection pattern behind CVE-2024-34359.",
    ),
    _r(
        "GG412",
        "duplicate-tensor-name",
        M,
        "Two tensors share a name.",
        "Loaders resolve tensors by name.",
    ),
    _r(
        "GG413",
        "oversized-array",
        M,
        "Metadata array or nesting exceeds limits.",
        "Large arrays and deep nesting are resource-exhaustion vectors.",
    ),
    _r(
        "GG414",
        "invalid-utf8",
        L,
        "A GGUF string is not valid UTF-8.",
        "The specification requires UTF-8 strings.",
    ),
    _r(
        "GG415",
        "block-size-mismatch",
        M,
        "Tensor row length is not a multiple of the quantisation block size.",
        "Quantised types store whole blocks; a partial block means the size arithmetic is "
        "inconsistent.",
    ),
    # --- ONNX --------------------------------------------------------------------------
    _r(
        "OX500",
        "malformed-protobuf",
        M,
        "ONNX protobuf could not be decoded.",
        "A damaged model is parsed differently by different runtimes.",
    ),
    _r(
        "OX501",
        "external-data-absolute-path",
        H,
        "External tensor data uses an absolute path.",
        "Absolute locations make the runtime read arbitrary files on the host.",
    ),
    _r(
        "OX502",
        "external-data-path-traversal",
        H,
        "External tensor data path escapes the model directory.",
        "'..' segments or symlinks let a model read files outside its directory "
        "(CVE-2022-25882 class).",
    ),
    _r(
        "OX503",
        "external-data-missing",
        M,
        "External data file does not exist next to the model.",
        "The model is incomplete or expects files placed by another channel.",
    ),
    _r(
        "OX504",
        "external-data-range-invalid",
        M,
        "External data offset/length is invalid or beyond the file.",
        "Out-of-range offsets cause out-of-bounds reads in runtimes.",
    ),
    _r(
        "OX505",
        "custom-op-domain",
        L,
        "Graph uses operators from a non-standard domain.",
        "Custom operators require native libraries registered at load time.",
    ),
    _r(
        "OX506",
        "nesting-too-deep",
        M,
        "Graph or message nesting exceeds the configured depth.",
        "Deep nesting is a stack-exhaustion vector for recursive protobuf parsers.",
    ),
    # --- Keras ---------------------------------------------------------------------------
    _r(
        "KR600",
        "lambda-layer",
        H,
        "Keras model contains a Lambda layer.",
        "Lambda layers carry Python bytecode (marshal) that runs when the model is loaded "
        "with safe_mode=False or by Keras 2.",
    ),
    _r(
        "KR601",
        "dangerous-reference",
        C,
        "Keras config references a deny-listed callable or class.",
        "Keras deserialises {'module', 'class_name'} references by importing them; "
        "deny-listed targets are code execution (CVE-2025-1550 class).",
    ),
    _r(
        "KR602",
        "unknown-function-reference",
        H,
        "Keras config references a function outside the allow-list.",
        "Function references are imported and called during deserialisation.",
    ),
    _r(
        "KR603",
        "custom-object",
        M,
        "Keras config instantiates a class from outside the Keras namespaces.",
        "Custom objects require user code at load time and are a supply-chain surface.",
    ),
    _r(
        "KR604",
        "malformed-config",
        M,
        "Keras archive config is missing or is not valid JSON.",
        "The model cannot be validated.",
    ),
    # --- NumPy ---------------------------------------------------------------------------
    _r(
        "NP700",
        "object-array",
        M,
        "NumPy array uses object dtype, stored as a pickle.",
        "Loading needs allow_pickle=True; the embedded pickle is scanned recursively.",
    ),
    _r(
        "NP701",
        "malformed-npy",
        M,
        "NumPy .npy header is invalid.",
        "The header must be a Python literal dict with descr/fortran_order/shape.",
    ),
    # --- general -------------------------------------------------------------------------
    _r(
        "GEN900",
        "format-mismatch",
        M,
        "File extension does not match the detected content.",
        "Loaders choose a deserializer by extension; a pickle named .safetensors defeats "
        "format-based trust decisions.",
    ),
    _r(
        "GEN901",
        "scan-error",
        M,
        "The file could not be fully scanned.",
        "I/O failures or parser limits prevented a complete analysis.",
    ),
    _r(
        "GEN903",
        "polyglot",
        H,
        "File is simultaneously valid as two different formats.",
        "A file that is both a pickle and a zip archive (or a safetensors file with a zip "
        "trailer) is loaded differently depending on which reader opens it.",
    ),
    _r(
        "GEN902",
        "file-too-large",
        L,
        "File exceeds the configured size limit and was not scanned.",
        "Raise --max-file-size to scan it.",
    ),
)

RULES: dict[str, Rule] = {rule.id: rule for rule in _RULES}


def get_rule(rule_id: str) -> Rule:
    try:
        return RULES[rule_id]
    except KeyError:
        raise KeyError(f"unknown rule id {rule_id!r}") from None


def finding(
    rule_id: str,
    location: Location,
    message: str,
    /,
    *,
    severity: Severity | None = None,
    **details: Any,
) -> Finding:
    """Build a :class:`Finding`, defaulting severity to the rule's severity."""
    rule = get_rule(rule_id)
    return Finding(
        rule_id=rule.id,
        severity=rule.severity if severity is None else severity,
        message=message,
        location=location,
        details=details,
    )
