"""SARIF/JSON/text output, the scanner entry points and the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from picklescope import ScanLimits, Severity, __version__, scan_bytes, scan_file, scan_paths
from picklescope.cli import main
from picklescope.output import render_json, render_text, to_sarif
from picklescope.rules import RULES, finding, get_rule

from conftest import FIXTURES

SCHEMA = json.loads((Path(__file__).parent / "data" / "sarif-schema-2.1.0.json").read_text())


@pytest.fixture(scope="module")
def corpus_report() -> object:
    return scan_paths([FIXTURES])


def test_sarif_validates_against_official_schema(corpus_report: object) -> None:
    log = to_sarif(corpus_report)  # type: ignore[arg-type]
    jsonschema.validate(log, SCHEMA)
    run = log["runs"][0]
    assert run["tool"]["driver"]["version"] == __version__
    assert {r["id"] for r in run["tool"]["driver"]["rules"]} == set(RULES)
    for result in run["results"]:
        rule = run["tool"]["driver"]["rules"][result["ruleIndex"]]
        assert rule["id"] == result["ruleId"]
        assert result["level"] in ("error", "warning", "note")


def test_sarif_locations(corpus_report: object) -> None:
    log = to_sarif(corpus_report)  # type: ignore[arg-type]
    results = log["runs"][0]["results"]
    member = next(
        r
        for r in results
        if r["ruleId"] == "PKL102"
        and "print_in_data_pkl" in r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
    )
    assert member["locations"][0]["logicalLocations"][0]["fullyQualifiedName"].startswith(
        "archive/data.pkl@"
    )
    assert "region" not in member["locations"][0]["physicalLocation"]
    top = next(r for r in results if r["ruleId"] == "PKL113")
    assert top["locations"][0]["physicalLocation"]["region"]["byteOffset"] == 2
    severities = {r["properties"]["severity"] for r in results}
    assert "critical" in severities
    fingerprints = [r["partialFingerprints"]["picklescopeFinding/v1"] for r in results]
    assert len(set(fingerprints)) == len(fingerprints)


def test_sarif_min_severity_and_absolute_uris(tmp_path: Path) -> None:
    target = tmp_path / "x.pkl"
    target.write_bytes((FIXTURES / "pickle" / "trailing_data.pkl").read_bytes())
    report = scan_paths([target])
    assert to_sarif(report, Severity.MEDIUM)["runs"][0]["results"] == []
    log = to_sarif(report, Severity.LOW)
    uri = log["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]
    assert uri["uri"].startswith("file://") and "uriBaseId" not in uri
    jsonschema.validate(log, SCHEMA)


def test_json_and_text_rendering(corpus_report: object) -> None:
    data = json.loads(render_json(corpus_report))  # type: ignore[arg-type]
    assert data["tool"]["version"] == __version__
    assert data["summary"]["files"] == len(data["files"])
    assert data["summary"]["by_severity"]["critical"] >= 1
    text = render_text(corpus_report, color=True, show_imports=True)  # type: ignore[arg-type]
    assert "\033[" in text and "imports:" in text and "skipped" in text
    plain = render_text(corpus_report, Severity.CRITICAL)  # type: ignore[arg-type]
    assert "PKL100" in plain and "PKL102" not in plain


def test_scan_file_size_limit_and_missing(tmp_path: Path) -> None:
    big = tmp_path / "big.pkl"
    big.write_bytes(b"N." * 100)
    report = scan_file(big, limits=ScanLimits(max_file_size=10))
    assert report.skipped_reason == "file too large"
    assert [f.rule_id for f in report.findings] == ["GEN902"]
    missing = scan_file(tmp_path / "nope.pkl")
    assert missing.format == "unreadable"
    assert [f.rule_id for f in missing.findings] == ["GEN901"]


def test_scan_file_internal_error_becomes_finding(monkeypatch: pytest.MonkeyPatch,
                                                  tmp_path: Path) -> None:  # fmt: skip
    import picklescope.scanner as scanner

    def boom(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(scanner, "_dispatch", boom)
    target = tmp_path / "a.pkl"
    target.write_bytes(b"\x80\x04N.")
    report = scan_file(target)
    assert [f.rule_id for f in report.findings] == ["GEN901"]
    assert "synthetic failure" in report.findings[0].message
    in_memory = scan_bytes(b"\x80\x04N.", "a.pkl")
    assert [f.rule_id for f in in_memory.findings] == ["GEN901"]


def test_scan_bytes_and_symlink_handling(tmp_path: Path) -> None:
    report = scan_bytes((FIXTURES / "pickle" / "call_print.pkl").read_bytes(), "x.safetensors")
    assert {f.rule_id for f in report.findings} == {"GEN900", "PKL102"}
    assert scan_bytes(b"", "e.bin").skipped_reason == "empty format"
    (tmp_path / "real.pkl").write_bytes(b"\x80\x04N.")
    (tmp_path / "link.pkl").symlink_to(tmp_path / "real.pkl")
    assert [f.path for f in scan_paths([tmp_path]).files] == [str(tmp_path / "real.pkl")]
    assert len(scan_paths([tmp_path], follow_symlinks=True).files) == 2


def test_rules_catalogue() -> None:
    assert get_rule("PKL100").severity is Severity.CRITICAL
    with pytest.raises(KeyError):
        get_rule("NOPE")
    item = finding("ST300", __import__("picklescope").Location("f"), "m", severity=Severity.LOW)
    assert item.severity is Severity.LOW
    assert all(rule.summary.endswith(".") for rule in RULES.values())


# --------------------------------------------------------------------------- CLI


def test_cli_scan_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["scan", str(FIXTURES / "safetensors" / "valid.safetensors")]) == 0
    assert main(["scan", str(FIXTURES / "pickle" / "call_print.pkl")]) == 1
    assert main(["scan", "--fail-on", "critical", str(FIXTURES / "pickle" / "call_print.pkl")]) == 0
    assert main(["scan", str(FIXTURES / "does-not-exist")]) == 2
    out = capsys.readouterr()
    assert "PKL102" in out.out and "no such file" in out.err


def test_cli_scan_formats(tmp_path: Path) -> None:
    sarif = tmp_path / "out.sarif"
    code = main(["scan", "-f", "sarif", "-o", str(sarif), "--no-hash", str(FIXTURES / "torch")])
    assert code == 1
    jsonschema.validate(json.loads(sarif.read_text()), SCHEMA)
    report = tmp_path / "out.json"
    main(["scan", "-f", "json", "-o", str(report), "--max-file-size", "1m", str(FIXTURES / "npy")])
    assert json.loads(report.read_text())["summary"]["files"] == 5


def test_cli_policy_and_profiles(capsys: pytest.CaptureFixture[str]) -> None:
    policy = str(FIXTURES / "policy-test.toml")
    target = str(FIXTURES / "pickle" / "colorsys_call.pkl")
    assert main(["scan", "--policy", policy, "--min-severity", "critical", target]) == 1
    assert "PKL100" in capsys.readouterr().out
    assert main(["policy", "check", "torch:Size", "--profile", "sklearn"]) == 0
    assert main(["policy", "check", "os:system", "sklearn.svm:SVC"]) == 1
    out = capsys.readouterr().out
    assert "deny" in out and "unknown" in out
    assert main(["policy", "check", "os:*"]) == 2
    assert main(["policy", "check", "bad"]) == 2
    assert main(["policy", "show", "--policy", policy]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "fixture-test"


def test_cli_bad_policy_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("bogus = 1")
    assert main(["scan", "--policy", str(bad), str(FIXTURES / "npy")]) == 2
    assert "unknown top-level" in capsys.readouterr().err


def test_cli_disasm(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["disasm", str(FIXTURES / "pickle" / "getattr_gadget.pkl")]) == 0
    out = capsys.readouterr().out
    assert "import builtins.getattr -> deny" in out and "call builtins.getattr" in out
    assert main(["disasm", str(FIXTURES / "torch" / "benign_state_dict.pt"),
                 "--member", "archive/data.pkl"]) == 0  # fmt: skip
    assert "torch._utils._rebuild_tensor_v2 -> allow" in capsys.readouterr().out
    assert main(["disasm", str(FIXTURES / "pickle" / "two_pickles.pkl")]) == 0
    assert capsys.readouterr().out.count("--- pickle at byte") == 2
    assert main(["disasm", str(FIXTURES / "pickle" / "truncated.pkl")]) == 0
    assert "!!! truncated" in capsys.readouterr().out
    assert main(["disasm", str(FIXTURES / "zip" / "bomb.zip"), "--member", "nope"]) == 2
    assert main(["disasm", str(FIXTURES / "nope.pkl")]) == 2


@pytest.mark.parametrize("fmt", ["text", "json", "markdown"])
def test_cli_rules(fmt: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["rules", "-f", fmt]) == 0
    out = capsys.readouterr().out
    assert "PKL100" in out and "GEN903" in out


def test_cli_rejects_bad_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    for argv in (["scan", "--fail-on", "urgent", "x"], ["scan", "--max-file-size", "0", "x"]):
        with pytest.raises(SystemExit) as info:
            main(argv)
        assert info.value.code == 2
    assert "invalid" in capsys.readouterr().err


def test_rules_doc_is_in_sync(capsys: pytest.CaptureFixture[str]) -> None:
    main(["rules", "-f", "markdown"])
    generated = capsys.readouterr().out
    doc = (Path(__file__).parent.parent / "docs" / "RULES.md").read_text()
    assert generated.strip() in doc, "docs/RULES.md is stale: run `make docs`"
