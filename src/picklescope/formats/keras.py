"""Keras model configs: ``.keras`` archives, KerasHub presets and legacy HDF5 files.

Keras deserialises a model by walking a JSON config. Two constructs execute code:

* ``Lambda`` layers, whose function is stored as base64-encoded ``marshal`` bytecode
  (Keras 3: ``{"class_name": "__lambda__", "config": {"code": ...}}``; Keras 2:
  ``"function": [b64, defaults, closure]`` with ``"function_type": "lambda"``).
* ``{"module": ..., "class_name": ...}`` object references, which Keras resolves by
  importing ``module`` and fetching the attribute. ``class_name == "function"``
  references a function by name in ``config``.

References are classified with the same import policy as pickle globals, plus the
``keras`` profile.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from picklescope.blob import Buffer
from picklescope.context import ScanContext
from picklescope.findings import Location
from picklescope.policy import Decision, Policy

log = logging.getLogger(__name__)

_MAX_JSON_NODES = 2_000_000
_H5_MARKER = b'{"class_name"'
_H5_MAX_CANDIDATES = 16
_H5_MAX_CONFIG = 64 * 1024**2


def _has_bytecode(function: Any, layer_config: dict[str, Any]) -> bool:
    if isinstance(function, dict) and function.get("class_name") == "__lambda__":
        return True
    return isinstance(function, list) and layer_config.get("function_type") == "lambda"


def _layer_name(node: dict[str, Any]) -> str:
    config = node.get("config")
    if isinstance(config, dict) and isinstance(config.get("name"), str):
        return str(config["name"])
    return "<unnamed>"


def _check_reference(
    node: dict[str, Any],
    path: str,
    loc: Location,
    ctx: ScanContext,
    policy: Policy,
    origin: str,
) -> None:
    module = node.get("module")
    class_name = node.get("class_name")
    if not isinstance(module, str) or not isinstance(class_name, str):
        registered = node.get("registered_name")
        if isinstance(registered, str) and ">" in registered:
            ctx.emit(
                "KR603",
                loc,
                f"{path}: custom object {registered!r} must be registered by user code",
                json_path=path,
                origin=origin,
            )
        return
    is_function = class_name == "function"
    name = node.get("config") if is_function else class_name
    if not isinstance(name, str):
        ctx.emit(
            "KR604",
            loc,
            f"{path}: function reference without a string name",
            json_path=path,
            origin=origin,
        )
        return
    verdict = policy.classify(module, name)
    ctx.record_import(verdict, loc)
    kind = "function" if is_function else "class"
    if verdict.decision is Decision.DENY:
        entry = verdict.entry
        assert entry is not None
        ctx.emit(
            "KR601",
            loc,
            f"{path}: {kind} reference {module}.{name} is deny-listed ({entry.category})",
            json_path=path,
            module=module,
            name=name,
            category=entry.category,
            origin=origin,
        )
    elif verdict.decision is Decision.UNKNOWN:
        rule = "KR602" if is_function else "KR603"
        ctx.emit(
            rule,
            loc,
            f"{path}: {kind} reference {module}.{name} is outside the allow-list",
            json_path=path,
            module=module,
            name=name,
            origin=origin,
        )


def walk_config(config: Any, loc: Location, ctx: ScanContext, origin: str) -> int:
    """Walk a parsed Keras config iteratively; return the number of Lambda layers."""
    lambdas = 0
    policy = ctx.policy.with_profiles(["keras"])
    stack: list[tuple[Any, str]] = [(config, "$")]
    lambda_functions: set[int] = set()
    visited = 0
    while stack:
        node, path = stack.pop()
        visited += 1
        if visited > _MAX_JSON_NODES:
            ctx.emit("PKL116", loc, "Keras config walk stopped at the node limit")
            break
        if isinstance(node, dict):
            class_name = node.get("class_name")
            if class_name in ("Lambda", "__lambda__") and id(node) not in lambda_functions:
                lambdas += 1
                layer_config = node.get("config")
                layer_config = layer_config if isinstance(layer_config, dict) else {}
                lambda_functions.add(id(layer_config.get("function")))
                bytecode = class_name == "__lambda__" or _has_bytecode(
                    layer_config.get("function"), layer_config
                )
                what = "serialized Python bytecode" if bytecode else "a Python callable"
                ctx.emit(
                    "KR600",
                    loc,
                    f"{path}: Lambda layer {_layer_name(node)!r} carries {what}",
                    json_path=path,
                    bytecode=bytecode,
                    origin=origin,
                )
            if "module" in node or "registered_name" in node:
                _check_reference(node, path, loc, ctx, policy, origin)
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    stack.append((value, f"{path}.{key}"))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                if isinstance(value, (dict, list)):
                    stack.append((value, f"{path}[{index}]"))
    return lambdas


def scan_keras_config(
    data: bytes, loc: Location, ctx: ScanContext, origin: str = "config.json"
) -> bool:
    """Parse and walk a Keras JSON config. Returns False if it is not valid JSON."""
    try:
        config = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        ctx.emit("KR604", loc, f"config is not valid JSON: {exc}")
        return False
    walk_config(config, loc, ctx, origin)
    return True


def looks_like_keras_object(config: Any) -> bool:
    """True for anything ``keras.saving.deserialize_keras_object`` accepts as a
    serialized object: a dict with a string ``class_name`` and a ``config`` key
    (``module`` is optional for legacy and built-in names)."""
    return (
        isinstance(config, dict)
        and isinstance(config.get("class_name"), str)
        and ("config" in config)
    )


def looks_like_keras_config(config: Any) -> bool:
    """True for a top-level Keras object: has module, class_name and config keys."""
    return isinstance(config, dict) and {"module", "class_name", "config"} <= set(config)


def scan_h5(buf: Buffer, loc: Location, ctx: ScanContext) -> int:
    """Heuristically locate the ``model_config`` JSON inside a legacy Keras HDF5 file.

    HDF5 attributes are stored as contiguous strings, so the config can be found by
    searching for its opening bytes and decoding JSON from there. This is a heuristic
    (no HDF5 parser): weights-only files have no config and produce no findings.
    Returns the number of configs found.
    """
    found = 0
    start = 0
    decoder = json.JSONDecoder()
    for _ in range(_H5_MAX_CANDIDATES):
        pos = buf.find(_H5_MARKER, start)
        if pos < 0:
            break
        start = pos + 1
        raw = buf[pos : pos + _H5_MAX_CONFIG]
        nul = raw.find(b"\x00")
        if nul >= 0:
            raw = raw[:nul]
        try:
            config, end = decoder.raw_decode(raw.decode("utf-8", "replace"))
        except (json.JSONDecodeError, RecursionError):
            continue
        if not isinstance(config, dict) or "config" not in config:
            continue
        found += 1
        log.debug("%s: HDF5 model config at byte %d (%d chars)", loc.display(), pos, end)
        walk_config(config, loc.at(pos), ctx, "hdf5-heuristic")
        start = pos + max(end, 1)
    return found
