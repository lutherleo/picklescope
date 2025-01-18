"""Import policy: patterns, entries and the allow/deny decision procedure.

A *reference* is a ``(module, name)`` pair exactly as a pickle's GLOBAL opcode (or a
Keras ``{"module", "class_name"}`` object) names it. The policy classifies every
reference as ``ALLOW``, ``DENY`` or ``UNKNOWN``.

Pattern syntax is ``module:name``:

* module ``a.b`` matches exactly ``a.b``; ``a.b.*`` matches ``a.b`` and every submodule.
* name ``x`` / ``x.y`` matches exactly; ``*`` matches one identifier (no dots);
  ``**`` matches any name including dotted attribute paths.

Precedence: user deny > user allow > built-in deny > built-in allow (active profiles)
> unknown. Python-2 names (``__builtin__``, ``copy_reg`` ...) are evaluated both as
written and after the ``_compat_pickle`` mapping the unpickler applies; a deny on
either spelling wins.
"""

from __future__ import annotations

import _compat_pickle
import enum
from collections.abc import Iterable
from dataclasses import dataclass, field


class Decision(enum.Enum):
    ALLOW = "allow"
    DENY = "deny"
    UNKNOWN = "unknown"


class PatternError(ValueError):
    """Raised for syntactically invalid policy patterns."""


def _valid_dotted(text: str) -> bool:
    return bool(text) and all(part.isidentifier() for part in text.split("."))


@dataclass(frozen=True, slots=True)
class Pattern:
    module: str
    module_tree: bool
    name: str  # exact name, "*" or "**"

    @classmethod
    def parse(cls, text: str) -> Pattern:
        if text.count(":") != 1:
            raise PatternError(f"pattern {text!r} must have the form 'module:name'")
        module, name = (part.strip() for part in text.split(":"))
        tree = module.endswith(".*")
        if tree:
            module = module[:-2]
        if not _valid_dotted(module):
            raise PatternError(f"pattern {text!r}: invalid module {module!r}")
        if name not in ("*", "**") and not _valid_dotted(name):
            raise PatternError(f"pattern {text!r}: invalid name {name!r}")
        return cls(module, tree, name)

    def matches(self, module: str, name: str) -> bool:
        if self.module_tree:
            if module != self.module and not module.startswith(self.module + "."):
                return False
        elif module != self.module:
            return False
        if self.name == "**":
            return True
        if self.name == "*":
            return "." not in name
        return name == self.name

    def __str__(self) -> str:
        return f"{self.module}{'.*' if self.module_tree else ''}:{self.name}"


@dataclass(frozen=True, slots=True)
class Entry:
    """A policy entry: pattern plus why it exists."""

    pattern: Pattern
    reason: str
    category: str = ""
    source: str = "builtin"


@dataclass(frozen=True, slots=True)
class Verdict:
    decision: Decision
    module: str
    name: str
    entry: Entry | None = None

    @property
    def qualified(self) -> str:
        return f"{self.module}.{self.name}"


@dataclass(frozen=True, slots=True)
class Profile:
    """A named group of built-in allow entries (e.g. ``torch``)."""

    name: str
    description: str
    allow: tuple[Entry, ...]


def normalize(module: str, name: str) -> tuple[str, str]:
    """Apply the Python 2 -> 3 renames the unpickler uses for protocols < 3."""
    if (module, name) in _compat_pickle.NAME_MAPPING:
        mapped: tuple[str, str] = _compat_pickle.NAME_MAPPING[(module, name)]
        return mapped
    return _compat_pickle.IMPORT_MAPPING.get(module, module), name


def is_valid_reference(module: str, name: str) -> bool:
    """True when both parts are dotted identifiers (what a real serializer emits)."""
    return _valid_dotted(module) and _valid_dotted(name)


def has_dunder_traversal(name: str) -> bool:
    """True when a dotted name walks through a ``__dunder__`` attribute."""
    parts = name.split(".")
    return any(p.startswith("__") and p.endswith("__") and len(p) > 4 for p in parts[1:])


@dataclass(slots=True)
class Policy:
    """An evaluated allow/deny policy."""

    name: str
    builtin_deny: tuple[Entry, ...]
    profiles: dict[str, Profile]
    active_profiles: tuple[str, ...]
    user_allow: tuple[Entry, ...] = ()
    user_deny: tuple[Entry, ...] = ()
    _cache: dict[tuple[str, str], Verdict] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        unknown = [p for p in self.active_profiles if p not in self.profiles]
        if unknown:
            known = ", ".join(sorted(self.profiles))
            raise ValueError(f"unknown profile(s) {', '.join(unknown)} (known: {known})")

    def with_profiles(self, extra: Iterable[str]) -> Policy:
        """Return a copy with additional built-in profiles enabled."""
        active = tuple(dict.fromkeys((*self.active_profiles, *extra)))
        return Policy(
            name=self.name,
            builtin_deny=self.builtin_deny,
            profiles=self.profiles,
            active_profiles=active,
            user_allow=self.user_allow,
            user_deny=self.user_deny,
        )

    def _builtin_allow(self) -> Iterable[Entry]:
        for profile_name in self.active_profiles:
            yield from self.profiles[profile_name].allow

    @staticmethod
    def _first(entries: Iterable[Entry], module: str, name: str) -> Entry | None:
        for entry in entries:
            if entry.pattern.matches(module, name):
                return entry
        return None

    def _classify_one(self, module: str, name: str) -> Verdict:
        for entries, decision in (
            (self.user_deny, Decision.DENY),
            (self.user_allow, Decision.ALLOW),
            (self.builtin_deny, Decision.DENY),
            (self._builtin_allow(), Decision.ALLOW),
        ):
            entry = self._first(entries, module, name)
            if entry is not None:
                return Verdict(decision, module, name, entry)
        return Verdict(Decision.UNKNOWN, module, name)

    def classify(self, module: str, name: str) -> Verdict:
        """Classify a reference. Results are memoised per policy instance."""
        key = (module, name)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        verdict = self._classify_one(module, name)
        mapped = normalize(module, name)
        if mapped != key:
            alt = self._classify_one(*mapped)
            if alt.decision is Decision.DENY or (
                verdict.decision is Decision.UNKNOWN and alt.decision is Decision.ALLOW
            ):
                verdict = Verdict(alt.decision, module, name, alt.entry)
        self._cache[key] = verdict
        return verdict

    def describe(self) -> dict[str, object]:
        def dump(entries: Iterable[Entry]) -> list[dict[str, str]]:
            return [
                {
                    "pattern": str(e.pattern),
                    "reason": e.reason,
                    "category": e.category,
                    "source": e.source,
                }
                for e in entries
            ]

        return {
            "name": self.name,
            "active_profiles": list(self.active_profiles),
            "user_deny": dump(self.user_deny),
            "user_allow": dump(self.user_allow),
            "builtin_deny": dump(self.builtin_deny),
            "builtin_allow": {p: dump(self.profiles[p].allow) for p in self.active_profiles},
        }
