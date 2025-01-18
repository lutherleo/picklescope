"""Build :class:`Policy` objects from built-in data and optional TOML files.

Policy file format (all keys optional)::

    name = "acme-ml"
    profiles = ["core", "numpy", "torch"]   # built-in allow profiles to enable

    [[allow]]
    pattern = "acme.models.*:*"
    reason  = "our own model classes, reviewed in SEC-123"

    [[deny]]
    pattern  = "acme.debug:shell"
    reason   = "debug helper that spawns a shell"
    category = "process-execution"
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from picklescope.policy.builtin import DEFAULT_PROFILES, DENY, PROFILES
from picklescope.policy.model import Entry, Pattern, PatternError, Policy

_TOP_KEYS = {"name", "profiles", "allow", "deny"}
_ENTRY_KEYS = {"pattern", "reason", "category"}


class PolicyError(ValueError):
    """Raised for unreadable or invalid policy files."""


def default_policy(profiles: Sequence[str] | None = None) -> Policy:
    """The built-in policy with ``profiles`` (default: core, numpy, torch) enabled."""
    return Policy(
        name="default",
        builtin_deny=DENY,
        profiles=PROFILES,
        active_profiles=tuple(profiles) if profiles else DEFAULT_PROFILES,
    )


def _parse_entries(raw: Any, kind: str, source: str) -> tuple[Entry, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise PolicyError(f"{source}: '{kind}' must be an array of tables ([[{kind}]])")
    entries: list[Entry] = []
    for index, item in enumerate(raw):
        where = f"{source}: {kind}[{index}]"
        if not isinstance(item, dict):
            raise PolicyError(f"{where} must be a table")
        extra = set(item) - _ENTRY_KEYS
        if extra:
            raise PolicyError(f"{where}: unknown key(s) {', '.join(sorted(extra))}")
        pattern_text = item.get("pattern")
        if not isinstance(pattern_text, str):
            raise PolicyError(f"{where}: 'pattern' is required and must be a string")
        try:
            pattern = Pattern.parse(pattern_text)
        except PatternError as exc:
            raise PolicyError(f"{where}: {exc}") from None
        reason = item.get("reason", "")
        category = item.get("category", "user")
        if not isinstance(reason, str) or not isinstance(category, str):
            raise PolicyError(f"{where}: 'reason' and 'category' must be strings")
        entries.append(Entry(pattern, reason, category, source))
    return tuple(entries)


def policy_from_mapping(
    data: dict[str, Any], source: str, extra_profiles: Iterable[str] = ()
) -> Policy:
    """Build a policy from an already-parsed TOML mapping."""
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise PolicyError(f"{source}: unknown top-level key(s) {', '.join(sorted(unknown))}")
    name = data.get("name", Path(source).stem)
    if not isinstance(name, str):
        raise PolicyError(f"{source}: 'name' must be a string")
    profiles = data.get("profiles", list(DEFAULT_PROFILES))
    if not isinstance(profiles, list) or not all(isinstance(p, str) for p in profiles):
        raise PolicyError(f"{source}: 'profiles' must be an array of strings")
    active = tuple(dict.fromkeys([*profiles, *extra_profiles]))
    try:
        return Policy(
            name=name,
            builtin_deny=DENY,
            profiles=PROFILES,
            active_profiles=active,
            user_allow=_parse_entries(data.get("allow"), "allow", source),
            user_deny=_parse_entries(data.get("deny"), "deny", source),
        )
    except ValueError as exc:
        if isinstance(exc, PolicyError):
            raise
        raise PolicyError(f"{source}: {exc}") from None


def load_policy(path: str | Path, extra_profiles: Iterable[str] = ()) -> Policy:
    """Read a TOML policy file."""
    file = Path(path)
    try:
        data = tomllib.loads(file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PolicyError(f"cannot read policy {file}: {exc.strerror or exc}") from None
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise PolicyError(f"{file}: invalid TOML: {exc}") from None
    return policy_from_mapping(data, str(file), extra_profiles)
