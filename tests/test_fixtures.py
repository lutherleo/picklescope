"""Golden tests over the generated fixture corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from picklescope.findings import Severity
from picklescope.pkl.analyzer import analyze_pickle
from picklescope.pkl.disasm import looks_like_pickle
from picklescope.policy import load_policy

import fixturegen
from conftest import FIXTURES, rules_of, scan_fixture

# fixture path -> exact set of rule ids expected under the default policy
EXPECTED: dict[str, set[str]] = {
    "gguf/bad_magic.gguf": {"GG400"},
    "gguf/bad_value_type.gguf": {"GG404"},
    "gguf/bad_version.gguf": {"GG401"},
    "gguf/block_mismatch.gguf": {"GG415"},
    "gguf/chat_template_dunder.gguf": {"GG411"},
    "gguf/duplicate_key.gguf": {"GG405"},
    "gguf/duplicate_tensor.gguf": {"GG412"},
    "gguf/huge_counts.gguf": {"GG402"},
    "gguf/misaligned.gguf": {"GG409"},
    "gguf/overlap.gguf": {"GG410"},
    "gguf/string_oob.gguf": {"GG403"},
    "gguf/tensor_oob.gguf": {"GG408"},
    "gguf/too_many_dims.gguf": {"GG406"},
    "gguf/unknown_type.gguf": {"GG407"},
    "gguf/valid_big_endian.gguf": set(),
    "gguf/valid_v1.gguf": set(),
    "gguf/valid_v3.gguf": set(),
    "keras/allowed_function.keras": set(),
    "keras/bad_config.keras": {"KR604"},
    "keras/config_only.keras": {"KR601"},
    "keras/benign.keras": set(),
    "keras/custom_object.keras": {"KR603"},
    "keras/function_denied.keras": {"KR601"},
    "keras/function_print.keras": {"KR602"},
    "keras/lambda.keras": {"KR600"},
    "keras/lambda_npz_weights.keras": {"KR600"},
    "keras/legacy_lambda.h5": {"KR600"},
    "keras/preset/config.json": {"KR600"},
    "npy/arrays.npz": set(),
    "npy/bad_header.npy": {"NP701"},
    "npy/float32.npy": set(),
    "npy/object_print.npy": {"NP700", "PKL102"},
    "npy/object_strings.npy": {"NP700"},
    "onnx/absolute_path.onnx": {"OX501"},
    "onnx/bad_range/model.onnx": {"OX504"},
    "onnx/bad_range/weights.bin": set(),
    "onnx/custom_domain.onnx": {"OX505"},
    "onnx/external_ok/model.onnx": set(),
    "onnx/external_ok/weights.bin": set(),
    "onnx/garbage.onnx": {"OX500"},
    "onnx/missing_external.onnx": {"OX503"},
    "onnx/traversal.onnx": {"OX502"},
    "onnx/valid.onnx": set(),
    "pickle/benign_ordereddict_p2.pkl": set(),
    "pickle/benign_text_p0.pkl": set(),
    "pickle/call_print.pkl": {"PKL102"},
    "pickle/colorsys_call.pkl": {"PKL102"},
    "pickle/dunder_traversal.pkl": {"PKL101", "PKL104"},
    "pickle/dynamic_stack_global.pkl": {"PKL103"},
    "pickle/ext_opcode.pkl": {"PKL105"},
    "pickle/getattr_gadget.pkl": {"PKL100", "PKL106"},
    "pickle/global_mutation.pkl": {"PKL107"},
    "pickle/import_only_dedent.pkl": {"PKL101"},
    "pickle/inst_p0.pkl": {"PKL102"},
    "pickle/legacy_torch.pt": set(),
    "pickle/malformed_name.pkl": {"PKL101", "PKL115"},
    "pickle/nested_pickle.pkl": {"PKL102", "PKL109"},
    "pickle/newobj_ex_p4.pkl": {"PKL102"},
    "pickle/obj_p1.pkl": {"PKL102"},
    "pickle/persistent_id.pkl": {"PKL108"},
    "pickle/protocol_9.pkl": {"PKL114"},
    "pickle/py2_builtin_getattr.pkl": {"PKL100"},
    "pickle/stack_global_memo.pkl": {"PKL102"},
    "pickle/trailing_data.pkl": {"PKL111"},
    "pickle/truncated.pkl": {"PKL101", "PKL112"},
    "pickle/two_pickles.pkl": {"PKL102", "PKL110"},
    "pickle/unknown_opcode.pkl": {"PKL113"},
    "safetensors/appended_payload.safetensors": {"ST308"},
    "safetensors/bad_dtype.safetensors": {"ST302"},
    "safetensors/bad_metadata.safetensors": {"ST307"},
    "safetensors/bad_shape.safetensors": {"ST303"},
    "safetensors/duplicate_key.safetensors": {"ST306"},
    "safetensors/gap.safetensors": {"ST305"},
    "safetensors/header_past_eof.safetensors": {"ST300"},
    "safetensors/header_too_large.safetensors": {"ST300"},
    "safetensors/not_json.safetensors": {"ST301"},
    "safetensors/overlap.safetensors": {"ST305"},
    "safetensors/pickle_in_disguise.safetensors": {"GEN900", "PKL102"},
    "safetensors/size_mismatch.safetensors": {"ST304"},
    "safetensors/valid.safetensors": set(),
    "safetensors/zip_polyglot.safetensors": {"GEN903", "ST308"},
    "torch/benign_state_dict.pt": set(),
    "torch/missing_storage.pt": {"ZIP207"},
    "torch/print_in_data_pkl.pt": {"PKL102"},
    "torch/shadowed_data_pkl.pt": {"PKL102", "ZIP201"},
    "torch/torchscript.pt": {"ZIP206"},
    "zip/bomb.zip": {"ZIP202"},
    "zip/encrypted.zip": {"ZIP203"},
    "zip/hostile_layout.zip": {"ZIP200", "ZIP201", "ZIP204", "ZIP208"},
    "zip/nested_npy.zip": {"NP700", "PKL102"},
    "zip/not_really.zip": {"ZIP205"},
}


def test_expectations_cover_every_fixture() -> None:
    catalogue = json.loads((FIXTURES / "catalogue.json").read_text())
    assert set(catalogue) == set(EXPECTED)


@pytest.mark.parametrize("rel", sorted(EXPECTED))
def test_fixture_findings_match_golden(rel: str) -> None:
    report = scan_fixture(rel)
    assert rules_of(report) == EXPECTED[rel], [f.message for f in report.findings]


@pytest.mark.parametrize(
    "rel",
    sorted(
        k
        for k, v in json.loads((FIXTURES / "catalogue.json").read_text()).items()
        if v["malicious_pattern"]
    ),
)
def test_malicious_patterns_are_high_or_worse(rel: str) -> None:
    report = scan_fixture(rel)
    top = report.max_severity
    assert top is not None and top >= Severity.HIGH, rel


def test_test_policy_raises_colorsys_to_critical() -> None:
    policy = load_policy(FIXTURES / "policy-test.toml")
    report = scan_fixture("pickle/colorsys_call.pkl", policy=policy)
    assert rules_of(report) == {"PKL100"}
    assert report.max_severity is Severity.CRITICAL
    assert "test" in report.findings[0].message
    inst = scan_fixture("pickle/inst_p0.pkl", policy=policy)
    assert rules_of(inst) == {"PKL100"}


def test_fixtures_are_regenerated_byte_identically(tmp_path: Path) -> None:
    fixturegen.write_all(tmp_path)
    for fixture in fixturegen.FIXTURES_LIST:
        assert (tmp_path / fixture.path).read_bytes() == (FIXTURES / fixture.path).read_bytes(), (
            f"{fixture.path} is stale: run `make fixtures`"
        )
    assert (tmp_path / "catalogue.json").read_text() == (FIXTURES / "catalogue.json").read_text()


def _all_pickles(data: bytes) -> list[bytes]:
    """Every pickle stream reachable in a fixture blob (top level and embedded)."""
    found = []
    pos = 0
    while pos < len(data):
        pos = data.find(b"\x80", pos)
        if pos < 0:
            break
        if looks_like_pickle(data, pos):
            found.append(data[pos:])
        pos += 1
    return found


def test_fixtures_only_reference_harmless_callables() -> None:
    """Safety net for the corpus: no fixture may import anything dangerous."""
    seen: set[tuple[str, str]] = set()
    for fixture in fixturegen.FIXTURES_LIST:
        report = scan_fixture(fixture.path)
        seen |= {(i["module"], i["name"]) for i in report.imports}
        data = (FIXTURES / fixture.path).read_bytes()
        for stream in _all_pickles(data):
            analysis = analyze_pickle(stream)
            seen |= {(e.module, e.name) for e in analysis.imports}
        # text-protocol fixtures have no PROTO prologue
        if fixture.path.endswith((".pkl", ".pt")):
            seen |= {(e.module, e.name) for e in analyze_pickle(data).imports}
    unexpected = seen - fixturegen.HARMLESS_REFERENCES
    assert not unexpected, f"fixtures reference non-allow-listed callables: {unexpected}"
