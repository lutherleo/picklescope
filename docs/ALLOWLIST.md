# Allow-list design

picklescope decides whether an importable reference is acceptable with an **allow-list
first, deny-list second** policy. This document explains why, how a decision is made,
and how to extend the policy safely.

## What is a "reference"?

A reference is a `(module, name)` pair that a loader will resolve by importing `module`
and fetching attribute `name`:

| Source | Opcode / field | Example |
|---|---|---|
| pickle, protocol 0-3 | `GLOBAL`, `INST` (`c` / `i` + two text lines) | `ccollections\nOrderedDict\n` |
| pickle, protocol 4+ | `STACK_GLOBAL` (module and name popped from the stack) | `'torch._utils' '_rebuild_tensor_v2' \x93` |
| Keras 3 config | `{"module": ..., "class_name": ...}` object | `{"module": "keras.layers", "class_name": "Dense"}` |
| Keras 3 function | `{"module": ..., "class_name": "function", "config": name}` | `{"module": "keras.activations", "class_name": "function", "config": "relu"}` |

`EXT1/EXT2/EXT4` opcodes also import objects, through `copyreg`'s process-global
extension registry. They cannot be resolved statically and are always reported
(`PKL105`).

## Why an allow-list

A deny-list answers "is this one of the bad things I know about?". For pickle the set of
bad things is open-ended: any importable callable that reaches `exec`, `system`,
`open`, `getattr` or another deserializer is an exploit primitive, and the Python
ecosystem keeps adding them. Public bypasses of deny-list scanners have used modules the
list did not mention, for example `pip.main` (CVE-2025-1716) and
`numpy.testing._private.utils.runstring`; nested `pickle.loads` calls and
`operator.methodcaller`-style gadgets follow the same pattern.

An allow-list answers "is this one of the few things a model file legitimately needs?".
That set is small and stable: `torch.save` state dicts need about a dozen references,
NumPy arrays need four. Everything else is reported. The deny-list is kept only as a
**severity backstop**: it turns "unknown" into "critical" for references that are known
to be exploit primitives, and it overrides a profile that is broader than it should be
(for example `keras.utils:get_file` inside the allowed `keras.*` namespace).

## Decision procedure

`Policy.classify(module, name)` (in `src/picklescope/policy/model.py`) returns `ALLOW`,
`DENY` or `UNKNOWN`. The first matching entry wins, in this order:

1. user deny entries (policy file)
2. user allow entries (policy file)
3. built-in deny-list
4. built-in allow entries of the **active profiles**
5. otherwise `UNKNOWN`

User entries come first so an organisation can both forbid something the built-ins
allow and deliberately accept something the built-ins deny (e.g. `os:getcwd` for a
legacy model), with the reason recorded in the file.

### Python 2 names

For protocols < 3 the unpickler rewrites Python 2 names through
`_compat_pickle.IMPORT_MAPPING` / `NAME_MAPPING` (`__builtin__` -> `builtins`,
`copy_reg` -> `copyreg`, ...). picklescope classifies **both** spellings: a deny on
either wins; an allow on the mapped name is accepted. `__builtin__.getattr` is therefore
denied exactly like `builtins.getattr`.

### Dotted names

Since protocol 4, `find_class` resolves a dotted `name` attribute by attribute, so
`colorsys` + `rgb_to_hsv.__code__` reaches an object that no allow-list entry for
`colorsys.rgb_to_hsv` intended to expose. Two rules apply:

* allow patterns match the *whole* dotted name; `*` matches one identifier only, so an
  allow of `numpy.dtypes:*` does not allow `numpy.dtypes:Float64DType.__init__`;
* any dotted name that walks through a `__dunder__` segment is reported as `PKL104`
  (high) regardless of the policy.

Names that are not dotted identifiers at all (spaces, empty segments, relative
modules) are reported as `PKL115`.

## Pattern syntax

```
module:name
```

| Part | Form | Matches |
|---|---|---|
| module | `a.b` | exactly `a.b` |
| module | `a.b.*` | `a.b` and every submodule `a.b.x.y` |
| name | `x` or `x.y` | exactly that (dotted) name |
| name | `*` | any single identifier (no dots) |
| name | `**` | any name, dotted paths included |

The built-in deny-list uses `**` (conservative); allow entries use exact names or `*`.

## From decision to severity

| Decision | Referenced only | Invoked (`REDUCE`, `NEWOBJ`, `NEWOBJ_EX`, `OBJ`, `INST`) |
|---|---|---|
| DENY | `PKL100` critical | `PKL100` critical |
| UNKNOWN | `PKL101` medium | `PKL102` high |
| ALLOW | recorded in the import inventory | recorded in the import inventory |

"Referenced only" is still reported because importing a module executes its top-level
code, and an imported object can be invoked indirectly (as a `copyreg._reconstructor`
argument, through `BUILD`/`__setstate__`, or by an allowed function that calls its
argument). Keras references use the same table with their own rule ids (`KR601` deny,
`KR602` unknown function, `KR603` unknown class).

Calls whose target is not an imported name at all - the result of another call
(`getattr(x, 'y')(...)`) or a persistent object - are reported as `PKL106` (high)
independently of the policy, because the callee is chosen at load time.

## Built-in profiles

| Profile | Default | Contents | Notes |
|---|---|---|---|
| `core` | on | builtin value types, `collections`/`datetime`/`decimal`/... value types, `copyreg._reconstructor`, `copyreg.__newobj__`, `_codecs.encode` | What the pickler itself emits for plain data. |
| `numpy` | on | `numpy.core.multiarray._reconstruct`, `numpy.dtype`, `numpy.ndarray`, scalar/`_frombuffer` helpers (1.x and 2.x module paths), `numpy.dtypes:*`, RandomState constructors | Covers arrays and dtypes inside pickles. |
| `torch` | on | `torch._utils._rebuild_*`, `torch._tensor._rebuild_from_type_v2`, storage classes, dtypes, `torch.Size`, `torch.device`, `nn.Parameter` | Mirrors the spirit of PyTorch's `weights_only` unpickler allow-list. |
| `sklearn` | **off** | `sklearn.*:*`, `scipy.sparse.*:*` | Whole-package allow; estimators only restore attributes in `__setstate__`, but the surface is large. Enable explicitly with `--profile sklearn`. |
| `keras` | used for Keras configs | `keras.*`, `keras_hub.*`, `keras_nlp.*`, `keras_cv.*`, `tf_keras.*` | The deny-list still blocks `keras.utils.get_file`, `enable_unsafe_deserialization` and nested `load_model`. |

### Allowed does not mean harmless

Some allowed callables take other callables as arguments
(`torch._tensor._rebuild_from_type_v2(func, ...)`, `copyreg._reconstructor(cls, ...)`).
They are acceptable only because their callable arguments must themselves be imported
through `GLOBAL`/`STACK_GLOBAL`, and those imports are classified independently. A
callable argument that is *computed* (the result of a call) is caught by `PKL106`.
This composition argument is the core assumption of the model; see the limitations in
the README.

## Deny-list categories

Each built-in deny entry carries a category that is shown in findings:
`code-execution`, `process-execution`, `import-machinery`, `attribute-gadget`,
`filesystem`, `network`, `nested-deserializer`, `safety-switch`. The full list lives in
`src/picklescope/policy/builtin.py` and can be printed with `picklescope policy show`.

## Extending the policy

```toml
# acme-policy.toml
name = "acme-ml"
profiles = ["core", "numpy", "torch"]

[[allow]]
pattern = "acme.models.*:*"
reason = "first-party model classes, reviewed in SEC-123"

[[deny]]
pattern = "acme.debug:*"
reason = "debug helpers spawn shells"
category = "process-execution"
```

```console
$ picklescope policy check acme.models.vision:ResNet os:system --policy acme-policy.toml
$ picklescope scan --policy acme-policy.toml models/
```

Review checklist before adding an allow entry:

1. Does the callable (or the class's `__new__`, `__init__`, `__setstate__`,
   `__reduce__`) touch files, processes, the network or the import system?
2. Does it call any of its arguments, or look up attributes by a string argument?
3. Is the entry as narrow as possible (exact name rather than `*`, no `**`)?
4. Is the reason specific enough that a later reviewer can re-check it?
