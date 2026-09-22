"""Threshold and per-turn regression tests for the quantization quality gate."""

import math
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ppl.artifacts import digest, read_json, write_json  # noqa: E402
from reporting.quality_gate import (
    audit_report,
    evaluate_gate,
    load_policy,
    write_report,
)  # noqa: E402


def _policy():
    return {
        "schema_version": 1,
        "ppl": {
            "hf_to_pt2_max_relative_increase": 0.05,
            "pt2_to_pte_max_relative_increase": 0.01,
            "pt2_to_pte_max_mean_nll_increase": 0.01,
            "pt2_to_pte_min_top1_agreement": 0.98,
        },
        "generation": {
            "hf_to_pt2_max_pass_rate_drop_pp": 2.0,
            "hf_to_pt2_min_quality_retention": 0.98,
            "pt2_to_pte_max_pass_rate_drop_pp": 1.0,
            "hf_to_pte_max_pass_rate_drop_pp": 2.0,
            "max_eos_rate_drop_pp": 1.0,
            "max_repetition_rate_increase_pp": 1.0,
            "max_garbled_rate_increase_pp": 1.0,
            "critical_turn_ids": ["T10"],
        },
        "correctness": {
            "require_complete_runs": True,
            "require_same_ppl_inputs": True,
            "require_same_generation_cases": True,
            "require_same_generation_config": True,
            "require_lineage": True,
        },
    }


def _ppl():
    metric = {"name": "standard_teacher_forced_perplexity"}
    result = {}
    for engine, value, top1 in (
        ("hf", 10.0, list(range(100))),
        ("pt2", 10.2, list(range(100))),
        ("pte", 10.25, list(range(98)) + [999, 998]),
    ):
        result[engine] = {
            "root": f"/{engine}-ppl",
            "summary": {
                "complete": True,
                "ppl": value,
                "mean_nll": math.log(value),
                "target_count": 100,
                "model_sha256": f"{engine}-model",
                "tokens_sha256": "tokens",
                "targets_sha256": "targets",
                "tokenizer_sha256": "tokenizer",
                "context_len": 32,
                "stride": 16,
                "metric": metric,
            },
            "top1_token_ids": top1,
        }
        result[engine]["evidence_sha256"] = digest(result[engine]["summary"])
    return result


def _generation():
    config = {
        "temperature": 0,
        "seq_len": 4096,
        "repetition_penalty": 1.0,
        "repeat_last_n": 128,
        "thinking_disabled": False,
        "repair_think_eos": False,
        "enable_think_budget": False,
    }
    result = {}
    for engine in ("hf", "pt2", "pte"):
        rows = [
            {
                "id": f"T{index}",
                "passed": True,
                "reviewed": True,
                "critical": index == 10,
                "eos": True,
                "repetition": False,
                "garbled": False,
                "record_hash": f"{engine}-{index}",
                "incremental_input_ids": [100, index],
                "native_input_ids": [100, index],
            }
            for index in range(100)
        ]
        result[engine] = {
            "root": f"/{engine}-generation",
            "identity": {
                "config": dict(config),
                "tokenizer_hash": "tokenizer-json",
                "evaluation": {
                    "kind": "business",
                    "model_sha256": f"{engine}-model",
                    "scorer_hashes": {"fixture.py": "shared-scorer"},
                },
            },
            "cases_sha256": "cases",
            "summary": {"run_fingerprint": f"{engine}-run"},
            "rows": rows,
            "evidence_sha256": f"{engine}-generation-evidence",
        }
    result["pt2"]["rows"][0]["passed"] = False
    result["pte"]["rows"][0]["passed"] = False
    result["pte"]["rows"][1].update(
        passed=False, eos=False, repetition=True, garbled=True
    )
    return result


def _lineage(path):
    configuration = {"quantization_mode": "8a4w"}
    body = {
        "schema": "executorch.quantization.lineage.v1",
        "complete": True,
        "sources": {
            "safetensors": {"kind": "local", "sha256": "hf-model"},
            "checkpoint": {"kind": "local", "sha256": "checkpoint"},
            "runtime_tokenizer": {"kind": "local", "sha256": "tokenizer"},
            "calibration": [
                {"option": "calib_samples", "kind": "local", "sha256": "calib"}
            ],
        },
        "configuration": configuration,
        "configuration_sha256": digest(configuration),
        "outputs": {
            "pt2": {"kind": "local", "sha256": "pt2-model"},
            "pte": {"kind": "local", "sha256": "pte-model"},
        },
        "environment": {
            "code": {
                "commit": "a" * 40,
                "dirty": False,
                "tracked_diff_sha256": "b" * 64,
                "untracked_sources": {},
            },
            "code_commit": "a" * 40,
            "qnn_sdk_build_id": "2.49",
        },
    }
    code = body["environment"]["code"]
    code["sha256"] = digest(code)
    body["lineage_sha256"] = digest(body)
    write_json(path, body)
    return path


def test_gate_passes_at_boundaries_and_writes_report(tmp_path):
    result = evaluate_gate(
        _policy(), _ppl(), _generation(), _lineage(tmp_path / "lineage.json")
    )
    assert result["passed"] is True
    assert result["metrics"]["ppl"]["pt2_to_pte_top1_agreement"] == 0.98
    assert result["metrics"]["generation"]["hf_to_pte"][
        "pass_rate_drop_pp"
    ] == pytest.approx(2.0)
    assert result["metrics"]["generation"]["critical_regressions"] == []
    write_report(tmp_path / "report", result)
    assert (tmp_path / "report/gate.json").is_file()
    assert "PASS" in (tmp_path / "report/REPORT.md").read_text(encoding="utf-8")


def test_gate_reports_critical_regression_for_each_transition(tmp_path):
    generation = _generation()
    generation["pt2"]["rows"][10]["passed"] = False
    generation["pte"]["rows"][10]["passed"] = False
    result = evaluate_gate(
        _policy(), _ppl(), generation, _lineage(tmp_path / "lineage.json")
    )
    assert result["passed"] is False
    critical = result["metrics"]["generation"]["critical_regressions_by_comparison"]
    assert critical["hf_to_pt2"] == ["T10"]
    assert critical["hf_to_pte"] == ["T10"]
    assert any(
        check["name"] == "critical new regressions" and not check["passed"]
        for check in result["checks"]
    )


def test_correctness_requirements_cannot_be_disabled(tmp_path):
    policy = _policy()
    policy["correctness"]["require_lineage"] = False
    path = tmp_path / "policy.json"
    write_json(path, policy)
    try:
        load_policy(path)
    except ValueError as error:
        assert "mandatory" in str(error)
    else:
        raise AssertionError("Disabled correctness requirement was accepted")


def test_shipped_policy_validates():
    assert (
        load_policy(ROOT / "configs/quality_gate.example.json")["schema_version"] == 1
    )


def test_generation_case_configuration_or_model_drift_is_invalid(tmp_path):
    for mutation in ("cases", "config", "tokenizer", "scorer", "model"):
        generation = deepcopy(_generation())
        if mutation == "cases":
            generation["pte"]["cases_sha256"] = "other"
        elif mutation == "config":
            generation["pte"]["identity"]["config"]["seq_len"] = 2048
        elif mutation == "tokenizer":
            generation["pte"]["identity"]["tokenizer_hash"] = "other"
        elif mutation == "scorer":
            generation["pte"]["identity"]["evaluation"]["scorer_hashes"] = {
                "fixture.py": "other"
            }
        else:
            generation["pte"]["identity"]["evaluation"]["model_sha256"] = "other"
        try:
            evaluate_gate(
                _policy(),
                _ppl(),
                generation,
                _lineage(tmp_path / f"{mutation}.json"),
            )
        except ValueError as error:
            assert any(
                message in str(error)
                for message in (
                    "identical cases",
                    "configurations differ",
                    "same tokenizer.json",
                    "same business scorer",
                    "not bound",
                )
            )
        else:
            raise AssertionError("Incomparable generation evidence was accepted")


def test_generation_input_token_drift_is_invalid(tmp_path):
    generation = _generation()
    generation["pte"]["rows"][7]["incremental_input_ids"] = [999]
    with pytest.raises(ValueError, match="input token IDs differ"):
        evaluate_gate(
            _policy(), _ppl(), generation, _lineage(tmp_path / "lineage.json")
        )


def test_pte_native_tokenizer_drift_is_invalid(tmp_path):
    generation = _generation()
    generation["pte"]["rows"][7]["native_input_ids"] = [999]
    with pytest.raises(ValueError, match="native tokenizer token IDs differ"):
        evaluate_gate(
            _policy(), _ppl(), generation, _lineage(tmp_path / "lineage.json")
        )


def test_gate_self_hash_rejects_report_tampering(tmp_path):
    result = evaluate_gate(
        _policy(), _ppl(), _generation(), _lineage(tmp_path / "lineage.json")
    )
    output = tmp_path / "report"
    write_report(output, result)
    gate = read_json(output / "gate.json")
    gate["passed"] = False
    write_json(output / "gate.json", gate)
    with pytest.raises(ValueError, match="report hash"):
        audit_report(output)
