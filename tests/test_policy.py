"""Allow/deny policy semantics and policy files."""

from __future__ import annotations

from pathlib import Path

import pytest

from picklescope.policy import (
    DEFAULT_PROFILES,
    Decision,
    Pattern,
    PatternError,
    PolicyError,
    default_policy,
    has_dunder_traversal,
    is_valid_reference,
    load_policy,
    normalize,
)
from picklescope.policy.loader import policy_from_mapping


@pytest.mark.parametrize(
    ("pattern", "module", "name", "matches"),
    [
        ("os:system", "os", "system", True),
        ("os:system", "os", "popen", False),
        ("os:*", "os", "popen", True),
        ("os:*", "os", "path.join", False),
        ("os:**", "os", "path.join", True),
        ("os.*:**", "os", "x", True),
        ("os.*:**", "os.path", "join", True),
        ("os.*:**", "osx", "join", False),
        ("torch:Size", "torch", "Size", True),
        ("a.b:C.d", "a.b", "C.d", True),
    ],
)
def test_pattern_matching(pattern: str, module: str, name: str, matches: bool) -> None:
    assert Pattern.parse(pattern).matches(module, name) is matches


@pytest.mark.parametrize("bad", ["os", "os:a:b", ":x", "os:", "1os:x", "os:a b", "os.:x"])
def test_pattern_errors(bad: str) -> None:
    with pytest.raises(PatternError):
        Pattern.parse(bad)


def test_pattern_str_round_trip() -> None:
    for text in ("os.*:**", "torch:Size", "numpy.dtypes:*"):
        assert str(Pattern.parse(text)) == text


@pytest.mark.parametrize(
    ("module", "name", "decision"),
    [
        ("torch._utils", "_rebuild_tensor_v2", Decision.ALLOW),
        ("collections", "OrderedDict", Decision.ALLOW),
        ("numpy.core.multiarray", "_reconstruct", Decision.ALLOW),
        ("numpy.dtypes", "Float64DType", Decision.ALLOW),
        ("builtins", "set", Decision.ALLOW),
        ("__builtin__", "set", Decision.ALLOW),  # protocol-2 spelling
        ("copy_reg", "_reconstructor", Decision.ALLOW),
        ("builtins", "eval", Decision.DENY),
        ("builtins", "getattr", Decision.DENY),
        ("__builtin__", "getattr", Decision.DENY),
        ("os", "system", Decision.DENY),
        ("posix", "system", Decision.DENY),
        ("nt", "system", Decision.DENY),
        ("subprocess", "Popen", Decision.DENY),
        ("os.path", "join", Decision.DENY),
        ("runpy", "_run_code", Decision.DENY),
        ("pickle", "loads", Decision.DENY),
        ("_pickle", "loads", Decision.DENY),
        ("operator", "attrgetter", Decision.DENY),
        ("numpy.testing._private.utils", "runstring", Decision.DENY),
        ("torch", "load", Decision.DENY),
        ("torch.hub", "load", Decision.DENY),
        ("keras.config", "enable_unsafe_deserialization", Decision.DENY),
        ("webbrowser", "open", Decision.DENY),
        ("sklearn.linear_model", "LogisticRegression", Decision.UNKNOWN),
        ("builtins", "print", Decision.UNKNOWN),
        ("mypackage", "Model", Decision.UNKNOWN),
    ],
)
def test_default_policy(module: str, name: str, decision: Decision) -> None:
    assert default_policy().classify(module, name).decision is decision


def test_sklearn_profile_is_opt_in() -> None:
    policy = default_policy().with_profiles(["sklearn"])
    assert policy.classify("sklearn.linear_model", "LogisticRegression").decision is Decision.ALLOW
    assert policy.classify("sklearn.linear_model", "Cls.__init__").decision is Decision.UNKNOWN
    assert policy.active_profiles == (*DEFAULT_PROFILES, "sklearn")


def test_builtin_deny_beats_builtin_allow() -> None:
    policy = default_policy().with_profiles(["keras"])
    assert policy.classify("keras.layers", "Dense").decision is Decision.ALLOW
    assert policy.classify("keras.utils", "get_file").decision is Decision.DENY


def test_user_entries_take_precedence(tmp_path: Path) -> None:
    file = tmp_path / "p.toml"
    file.write_text(
        'name = "custom"\nprofiles = ["core"]\n'
        '[[allow]]\npattern = "mypkg.*:*"\nreason = "reviewed"\n'
        '[[allow]]\npattern = "os:getcwd"\nreason = "override builtin deny"\n'
        '[[deny]]\npattern = "collections:OrderedDict"\ncategory = "test"\n'
    )
    policy = load_policy(file)
    assert policy.name == "custom"
    assert policy.classify("mypkg.models", "Net").decision is Decision.ALLOW
    assert policy.classify("os", "getcwd").decision is Decision.ALLOW
    assert policy.classify("os", "system").decision is Decision.DENY
    verdict = policy.classify("collections", "OrderedDict")
    assert verdict.decision is Decision.DENY
    assert verdict.entry is not None and verdict.entry.source == str(file)
    # torch profile not enabled in this file
    assert policy.classify("torch._utils", "_rebuild_tensor_v2").decision is Decision.UNKNOWN
    described = policy.describe()
    assert described["name"] == "custom" and described["user_allow"]


def test_classify_is_cached() -> None:
    policy = default_policy()
    first = policy.classify("os", "system")
    assert policy.classify("os", "system") is first


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("bogus = 1", "unknown top-level"),
        ("name = 3", "'name' must be a string"),
        ("profiles = 'core'", "'profiles' must be an array"),
        ("profiles = ['nope']", "unknown profile"),
        ("allow = 3", "array of tables"),
        ("allow = [1]", "must be a table"),
        ("[[allow]]\nreason = 'x'", "'pattern' is required"),
        ("[[allow]]\npattern = 'bad'", "module:name"),
        ("[[allow]]\npattern = 'os:x'\nextra = 1", "unknown key"),
        ("[[deny]]\npattern = 'os:x'\nreason = 1", "must be strings"),
        ("name = [", "invalid TOML"),
    ],
)
def test_policy_file_errors(tmp_path: Path, content: str, message: str) -> None:
    file = tmp_path / "bad.toml"
    file.write_text(content)
    with pytest.raises(PolicyError, match=message):
        load_policy(file)


def test_missing_policy_file(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match="cannot read"):
        load_policy(tmp_path / "absent.toml")


def test_policy_from_mapping_defaults_name_to_stem() -> None:
    policy = policy_from_mapping({}, "/x/acme.toml", ["sklearn"])
    assert policy.name == "acme"
    assert "sklearn" in policy.active_profiles


def test_reference_helpers() -> None:
    assert normalize("__builtin__", "set") == ("builtins", "set")
    assert normalize("__builtin__", "xrange") == ("builtins", "range")
    assert normalize("torch", "Size") == ("torch", "Size")
    assert is_valid_reference("a.b", "C.d")
    assert not is_valid_reference("a b", "C")
    assert not is_valid_reference(".a", "C")
    assert has_dunder_traversal("Foo.__init__.__globals__")
    assert not has_dunder_traversal("__reduce__")  # first segment is a plain attribute
    assert not has_dunder_traversal("Foo._private")
