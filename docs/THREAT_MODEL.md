# Threat model

## Scope and intended use

picklescope is a **defensive** pre-load gate: run it on model files obtained from
outside your trust boundary (model hubs, partners, CI artifacts) *before* any loader
touches them. It is a static triage tool, not a sandbox and not a proof of safety.

## Assets

* The host that will later load the model (code execution, credentials, data).
* The integrity of the model itself (what tensors a runtime will actually read).
* The scanner host (the scanner must not become the vulnerability).

## Adversary

Someone who can publish or substitute a model file: a malicious hub account, a
compromised CI job, a dependency-confusion style upload, a tampered mirror. They control
every byte of the file and its name, and they know which scanners are popular.

## Attack surface by format and what picklescope checks

| Format | Attack | Check |
|---|---|---|
| pickle (`.pkl`, `.pt`, `.bin`, `.ckpt`) | `GLOBAL`/`STACK_GLOBAL` import + `REDUCE` call of any callable | allow-list policy, `PKL100`-`PKL102` |
| | hide the import: memo indirection, computed `STACK_GLOBAL`, `EXT*` registry | memo-aware analysis, `PKL103`, `PKL105` |
| | gadget chains: `getattr(...)(...)`, dunder traversal in dotted names | `PKL106`, `PKL104`, deny-list `attribute-gadget` |
| | mutate classes/modules via `BUILD`/`SETITEM` | `PKL107` |
| | second pickle after `STOP`, pickle inside a bytes constant | `PKL110`, `PKL109` (recursive scan) |
| | malformed streams that scanners and loaders read differently | `PKL112`-`PKL115` |
| PyTorch zip | hostile `data.pkl`, extra pickles, TorchScript `code/` | member scanning, `ZIP206`, `ZIP207` |
| any zip | zip-slip, symlinks, duplicates, bombs, encrypted entries, embedded `.so`/`.py` | `ZIP200`-`ZIP208` |
| NumPy | object arrays (pickle payload) | `NP700` + pickle scan |
| Keras v3 | `Lambda` bytecode; `module`/`class_name` references to arbitrary functions (CVE-2025-1550 class); unsafe-mode switches | `KR600`-`KR603`, deny-list `safety-switch` |
| safetensors | header/length manipulation, overlaps, duplicate keys, appended payloads, zip polyglots | `ST300`-`ST308`, `GEN903` |
| GGUF | absurd counts and lengths (heap overflows in native readers), bad dims/types, Jinja SSTI in `tokenizer.chat_template` (CVE-2024-34359 class) | `GG400`-`GG415` |
| ONNX | external data that reads arbitrary host files (absolute, `..`, symlinks; CVE-2022-25882 class), custom-op native code | `OX501`-`OX505` |
| all | wrong extension to steer a loader (`pickle` named `.safetensors`) | `GEN900` |

## Trust boundary of the scanner

* Input files are untrusted. The scanner maps them read-only, never writes, never
  extracts archives to disk, never follows ONNX external-data paths outside the model
  directory (and only `stat`s paths proven to be inside it), and does not follow
  directory symlinks unless asked (`--follow-symlinks`).
* Nothing is executed: no unpickling, no `marshal`, no `eval`, no imports of names
  found in files. An AST test enforces this on the source tree.
* Resource exhaustion is bounded by `ScanLimits` (file size, member size, compression
  ratio, opcode count, stack, memo, nesting, header size, array size, protobuf depth,
  finding flood cap).
* Parser failures are contained per file and reported (`GEN901`), never propagated.

## Out of scope / known gaps

* **Behaviour of allowed callables with crafted arguments.** The model assumes the
  allow-listed reconstruction functions are safe for any arguments; a vulnerability
  inside, say, a NumPy or PyTorch rebuild function would not be detected.
* **Formats not parsed:** TensorFlow SavedModel / `.pb`, TFLite, joblib's format (it
  interleaves raw array bytes with pickle opcodes, which the analyser does not model),
  compressed pickles (gzip/zlib/lz4), tar/7z containers, legacy GGML (pre-GGUF),
  Core ML, PaddlePaddle.
* **HDF5** is not parsed; the Keras config is located heuristically.
* **TorchScript code** is reported, not analysed.
* **Model-level attacks** (backdoored weights, poisoned data, adversarial behaviour)
  are invisible to a format scanner.
* **Loader-specific behaviour:** custom `Unpickler.find_class` or `persistent_load`
  hooks in the consuming application can make a flagged file safe or an allowed file
  unsafe; picklescope models the default CPython unpickler and `torch.load`.

## Ethics

All test fixtures are generated in-repo by `tests/fixturegen.py`; every callable they
reference is harmless (`print`, `len`, `str.upper`, `os.getcwd`, `colorsys`,
`textwrap`), and a test fails if any fixture references anything else. No fixture is
ever loaded. Real models used for the false-positive evaluation are benign public
files, fetched by checksum and only scanned.
