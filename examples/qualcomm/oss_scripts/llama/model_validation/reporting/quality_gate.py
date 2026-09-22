"""Audit HF/PT2/PTE evidence and apply a configurable quantization quality gate."""

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.audit import audit_run, generation_evidence  # noqa: E402
from common.output import atomic_directory  # noqa: E402
from ppl.artifacts import digest, read_json, write_json  # noqa: E402


PPL_COMPARABILITY_FIELDS = (
    "tokens_sha256",
    "targets_sha256",
    "tokenizer_sha256",
    "context_len",
    "stride",
    "target_count",
    "metric",
)
GENERATION_CONFIG_FIELDS = {
    "temperature": 0,
    "seq_len": None,
    "repetition_penalty": 1.0,
    "repeat_last_n": None,
    "ignore_eos": False,
    "max_new_tokens": -1,
    "thinking_disabled": False,
    "repair_think_eos": False,
    "enable_think_budget": False,
}


def _finite_number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def validate_policy(policy):
    if set(policy) != {"schema_version", "ppl", "generation", "correctness"}:
        raise ValueError(
            "Policy requires schema_version, ppl, generation and correctness"
        )
    if policy["schema_version"] != 1:
        raise ValueError("Unsupported quality-gate policy schema")
    required_ppl = {
        "hf_to_pt2_max_relative_increase",
        "pt2_to_pte_max_relative_increase",
        "pt2_to_pte_max_mean_nll_increase",
        "pt2_to_pte_min_top1_agreement",
    }
    required_generation = {
        "hf_to_pt2_max_pass_rate_drop_pp",
        "hf_to_pt2_min_quality_retention",
        "pt2_to_pte_max_pass_rate_drop_pp",
        "hf_to_pte_max_pass_rate_drop_pp",
        "max_eos_rate_drop_pp",
        "max_repetition_rate_increase_pp",
        "max_garbled_rate_increase_pp",
        "critical_turn_ids",
    }
    required_correctness = {
        "require_complete_runs",
        "require_same_ppl_inputs",
        "require_same_generation_cases",
        "require_same_generation_config",
        "require_lineage",
    }
    for section, required in (
        ("ppl", required_ppl),
        ("generation", required_generation),
        ("correctness", required_correctness),
    ):
        if set(policy[section]) != required:
            raise ValueError(
                f"Policy {section} keys must be exactly {sorted(required)}"
            )
    for key, value in policy["ppl"].items():
        number = _finite_number(value, f"ppl.{key}")
        if number < 0 or ("agreement" in key and number > 1):
            raise ValueError(f"Invalid threshold ppl.{key}")
    for key, value in policy["generation"].items():
        if key == "critical_turn_ids":
            if (
                not isinstance(value, list)
                or len(set(value)) != len(value)
                or any(not isinstance(item, str) or not item for item in value)
            ):
                raise ValueError("critical_turn_ids must be unique nonempty strings")
        else:
            number = _finite_number(value, f"generation.{key}")
            if number < 0 or ("retention" in key and number > 1):
                raise ValueError(f"Invalid threshold generation.{key}")
    if any(value is not True for value in policy["correctness"].values()):
        raise ValueError("All correctness requirements are mandatory and must be true")
    return policy


def load_policy(path):
    return validate_policy(read_json(path))


def load_ppl(directory):
    root = Path(directory).resolve()
    audit_run(root)
    summary = read_json(root / "summary.json")
    if summary.get("schema") != "executorch.ppl.summary.v1":
        raise ValueError(f"Not a standard PPL run: {root}")
    top1 = []
    for path in sorted(root.glob("W*.json")):
        record = read_json(path)
        values = record.get("top1_token_ids")
        if values is None:
            raise ValueError(f"PPL run lacks top-1 evidence: {root}")
        top1.extend(values)
    if len(top1) != summary["target_count"]:
        raise ValueError(f"PPL top-1 coverage mismatch: {root}")
    return {
        "root": str(root),
        "summary": summary,
        "top1_token_ids": top1,
        "evidence_sha256": digest(summary),
    }


def require_comparable_ppl(runs):
    reference = runs["hf"]["summary"]
    mismatches = []
    for engine in ("pt2", "pte"):
        summary = runs[engine]["summary"]
        for field in PPL_COMPARABILITY_FIELDS:
            if summary.get(field) != reference.get(field):
                mismatches.append(f"{engine}.{field}")
    if mismatches:
        raise ValueError("PPL runs are not comparable: " + ", ".join(mismatches))


def _generation_config(identity):
    config = identity.get("config", {})
    return {
        key: config.get(key, default)
        for key, default in GENERATION_CONFIG_FIELDS.items()
    }


def require_comparable_generation(runs, *, compare_config):
    hashes = {run["cases_sha256"] for run in runs.values()}
    if len(hashes) != 1:
        raise ValueError("Generation runs do not use identical cases")
    ids = [{row["id"] for row in run["rows"]} for run in runs.values()]
    if any(value != ids[0] for value in ids[1:]):
        raise ValueError("Generation runs do not cover identical turn IDs")
    tokenizer_hashes = {run["identity"].get("tokenizer_hash") for run in runs.values()}
    if None in tokenizer_hashes or len(tokenizer_hashes) != 1:
        raise ValueError("Generation runs do not use the same tokenizer.json")
    scorer_hashes = [
        run["identity"].get("evaluation", {}).get("scorer_hashes")
        for run in runs.values()
    ]
    if any(not isinstance(value, dict) or not value for value in scorer_hashes) or any(
        value != scorer_hashes[0] for value in scorer_hashes[1:]
    ):
        raise ValueError("Generation runs do not use the same business scorer")
    if compare_config:
        configs = [_generation_config(run["identity"]) for run in runs.values()]
        if any(value != configs[0] for value in configs[1:]):
            raise ValueError(f"Generation configurations differ: {configs}")
    by_engine = {
        engine: {row["id"]: row for row in run["rows"]} for engine, run in runs.items()
    }
    for turn_id in sorted(ids[0]):
        inputs = {
            engine: rows[turn_id].get("incremental_input_ids")
            for engine, rows in by_engine.items()
        }
        if any(
            not isinstance(values, list)
            or not values
            or any(type(value) is not int or value < 0 for value in values)
            for values in inputs.values()
        ):
            raise ValueError(f"Generation lacks input-token evidence: {turn_id}")
        if any(values != inputs["hf"] for values in inputs.values()):
            raise ValueError(f"Generation input token IDs differ: {turn_id}")
        native = by_engine["pte"][turn_id].get("native_input_ids")
        if (
            not isinstance(native, list)
            or not native
            or any(type(value) is not int or value < 0 for value in native)
        ):
            raise ValueError(f"PTE lacks native tokenizer evidence: {turn_id}")
        if native != inputs["hf"]:
            raise ValueError(f"PTE native tokenizer token IDs differ: {turn_id}")
    for engine, run in runs.items():
        pending = [row["id"] for row in run["rows"] if not row["reviewed"]]
        if pending:
            raise ValueError(f"{engine} generation has pending reviews: {pending[:8]}")


def require_generation_model_bindings(generation, ppl):
    for engine in ("hf", "pt2", "pte"):
        evaluation = generation[engine]["identity"].get("evaluation", {})
        if (
            evaluation.get("kind") != "business"
            or evaluation.get("model_sha256") != ppl[engine]["summary"]["model_sha256"]
        ):
            raise ValueError(
                f"{engine} generation is not bound to its evaluated PPL model"
            )


def _rates(run):
    rows = run["rows"]
    count = len(rows)
    if not count:
        raise ValueError("Generation evidence is empty")
    passed = sum(row["passed"] is True for row in rows)
    return {
        "count": count,
        "passed": passed,
        "pass_rate": passed / count,
        "eos_rate": sum(row["eos"] for row in rows) / count,
        "repetition_rate": sum(row["repetition"] for row in rows) / count,
        "garbled_rate": sum(row["garbled"] for row in rows) / count,
    }


def _pair(baseline, candidate):
    left = {row["id"]: row for row in baseline["rows"]}
    right = {row["id"]: row for row in candidate["rows"]}
    base_rate = _rates(baseline)["pass_rate"]
    candidate_rate = _rates(candidate)["pass_rate"]
    regressions = [
        turn_id
        for turn_id in sorted(left)
        if left[turn_id]["passed"] is True and right[turn_id]["passed"] is not True
    ]
    recoveries = [
        turn_id
        for turn_id in sorted(left)
        if left[turn_id]["passed"] is not True and right[turn_id]["passed"] is True
    ]
    return {
        "baseline_pass_rate": base_rate,
        "candidate_pass_rate": candidate_rate,
        "pass_rate_drop_pp": 100 * (base_rate - candidate_rate),
        "quality_retention": candidate_rate / base_rate if base_rate else None,
        "regressions": regressions,
        "recoveries": recoveries,
    }


def validate_lineage(path, ppl):
    lineage = read_json(path)
    saved_hash = lineage.get("lineage_sha256")
    body = {key: value for key, value in lineage.items() if key != "lineage_sha256"}
    if (
        lineage.get("schema") != "executorch.quantization.lineage.v1"
        or saved_hash != digest(body)
        or lineage.get("complete") is not True
        or lineage.get("configuration_sha256") != digest(lineage.get("configuration"))
    ):
        raise ValueError("Export lineage is incomplete or has an invalid hash")
    sources = lineage.get("sources", {})
    checkpoint = sources.get("checkpoint")
    tokenizer = sources.get("runtime_tokenizer")
    calibration = sources.get("calibration")
    configuration = lineage.get("configuration", {})
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("kind") != "local"
        or not isinstance(tokenizer, dict)
        or tokenizer.get("kind") != "local"
        or not isinstance(calibration, list)
        or not calibration
        or any(
            not isinstance(item, dict) or item.get("kind") != "local"
            for item in calibration
        )
    ):
        raise ValueError("Export lineage lacks checkpoint/tokenizer/calibration inputs")
    if configuration.get("qat"):
        training = sources.get("training")
        train_config = sources.get("train_config")
        lr_config = sources.get("lr_config")
        if (
            not isinstance(training, list)
            or not training
            or any(
                not isinstance(item, dict) or item.get("kind") != "local"
                for item in training
            )
            or not isinstance(train_config, dict)
            or train_config.get("kind") != "local"
            or not isinstance(lr_config, dict)
            or lr_config.get("kind") != "local"
        ):
            raise ValueError("QAT lineage lacks content-bound training inputs")
    environment = lineage.get("environment", {})
    code = environment.get("code")
    if not isinstance(code, dict):
        raise ValueError("Export lineage lacks content-bound code identity")
    code_body = {key: value for key, value in code.items() if key != "sha256"}
    if (
        code.get("sha256") != digest(code_body)
        or len(code.get("commit", "")) != 40
        or len(code.get("tracked_diff_sha256", "")) != 64
        or not isinstance(code.get("untracked_sources"), dict)
        or environment.get("code_commit") != code.get("commit")
        or not environment.get("qnn_sdk_build_id")
    ):
        raise ValueError("Export lineage lacks code or QNN SDK identity")
    expected = {
        "hf": sources.get("safetensors"),
        "pt2": lineage.get("outputs", {}).get("pt2"),
        "pte": lineage.get("outputs", {}).get("pte"),
    }
    for engine, identity in expected.items():
        if (
            not isinstance(identity, dict)
            or identity.get("kind") != "local"
            or identity.get("sha256") != ppl[engine]["summary"]["model_sha256"]
        ):
            raise ValueError(f"Lineage does not bind the evaluated {engine} model")
    return lineage


def _check(checks, name, actual, relation, threshold):
    equal = math.isclose(actual, threshold, rel_tol=1e-12, abs_tol=1e-12)
    passed = (
        actual <= threshold or equal
        if relation == "<="
        else actual >= threshold or equal
    )
    checks.append(
        {
            "name": name,
            "actual": actual,
            "relation": relation,
            "threshold": threshold,
            "passed": passed,
        }
    )


def evaluate_gate(policy, ppl, generation, lineage_path=None):
    validate_policy(policy)
    require_comparable_ppl(ppl)
    require_comparable_generation(generation, compare_config=True)
    require_generation_model_bindings(generation, ppl)
    for engine in ("hf", "pt2", "pte"):
        if ppl[engine]["summary"]["complete"] is not True:
            raise ValueError(f"Incomplete PPL run: {engine}")
    lineage = None
    if policy["correctness"]["require_lineage"]:
        if lineage_path is None:
            raise ValueError("Policy requires --lineage")
        lineage = validate_lineage(lineage_path, ppl)
    ppl_values = {
        engine: {
            "ppl": run["summary"]["ppl"],
            "mean_nll": run["summary"]["mean_nll"],
        }
        for engine, run in ppl.items()
    }
    ppl_metrics = {
        "values": ppl_values,
        "hf_to_pt2_relative_increase": ppl["pt2"]["summary"]["ppl"]
        / ppl["hf"]["summary"]["ppl"]
        - 1,
        "hf_to_pt2_mean_nll_increase": ppl["pt2"]["summary"]["mean_nll"]
        - ppl["hf"]["summary"]["mean_nll"],
        "pt2_to_pte_relative_increase": ppl["pte"]["summary"]["ppl"]
        / ppl["pt2"]["summary"]["ppl"]
        - 1,
        "pt2_to_pte_mean_nll_increase": ppl["pte"]["summary"]["mean_nll"]
        - ppl["pt2"]["summary"]["mean_nll"],
        "hf_to_pte_relative_increase": ppl["pte"]["summary"]["ppl"]
        / ppl["hf"]["summary"]["ppl"]
        - 1,
        "hf_to_pte_mean_nll_increase": ppl["pte"]["summary"]["mean_nll"]
        - ppl["hf"]["summary"]["mean_nll"],
        "pt2_to_pte_top1_agreement": sum(
            left == right
            for left, right in zip(
                ppl["pt2"]["top1_token_ids"], ppl["pte"]["top1_token_ids"]
            )
        )
        / ppl["pt2"]["summary"]["target_count"],
    }
    generation_metrics = {
        "rates": {engine: _rates(run) for engine, run in generation.items()},
        "hf_to_pt2": _pair(generation["hf"], generation["pt2"]),
        "pt2_to_pte": _pair(generation["pt2"], generation["pte"]),
        "hf_to_pte": _pair(generation["hf"], generation["pte"]),
    }
    rates = generation_metrics["rates"]
    for comparison, baseline, candidate in (
        ("hf_to_pt2", "hf", "pt2"),
        ("pt2_to_pte", "pt2", "pte"),
        ("hf_to_pte", "hf", "pte"),
    ):
        generation_metrics[comparison].update(
            eos_rate_drop_pp=100
            * (rates[baseline]["eos_rate"] - rates[candidate]["eos_rate"]),
            repetition_rate_increase_pp=100
            * (
                rates[candidate]["repetition_rate"] - rates[baseline]["repetition_rate"]
            ),
            garbled_rate_increase_pp=100
            * (rates[candidate]["garbled_rate"] - rates[baseline]["garbled_rate"]),
        )
    configured_critical = set(policy["generation"]["critical_turn_ids"])
    declared_critical = {
        row["id"]
        for run in generation.values()
        for row in run["rows"]
        if row["critical"]
    }
    critical = configured_critical | declared_critical
    unknown = critical - {row["id"] for row in generation["hf"]["rows"]}
    if unknown:
        raise ValueError(f"Unknown critical turn IDs: {sorted(unknown)}")
    critical_regressions_by_comparison = {
        comparison: sorted(
            critical & set(generation_metrics[comparison]["regressions"])
        )
        for comparison in ("hf_to_pt2", "pt2_to_pte", "hf_to_pte")
    }
    critical_regressions = sorted(
        set().union(*map(set, critical_regressions_by_comparison.values()))
    )
    generation_metrics["critical_turn_ids"] = sorted(critical)
    generation_metrics["critical_regressions"] = critical_regressions
    generation_metrics["critical_regressions_by_comparison"] = (
        critical_regressions_by_comparison
    )
    checks = []
    _check(
        checks,
        "HF→PT2 PPL relative increase",
        ppl_metrics["hf_to_pt2_relative_increase"],
        "<=",
        policy["ppl"]["hf_to_pt2_max_relative_increase"],
    )
    _check(
        checks,
        "PT2→PTE PPL relative increase",
        ppl_metrics["pt2_to_pte_relative_increase"],
        "<=",
        policy["ppl"]["pt2_to_pte_max_relative_increase"],
    )
    _check(
        checks,
        "PT2→PTE mean NLL increase",
        ppl_metrics["pt2_to_pte_mean_nll_increase"],
        "<=",
        policy["ppl"]["pt2_to_pte_max_mean_nll_increase"],
    )
    _check(
        checks,
        "PT2→PTE teacher-forcing top-1 agreement",
        ppl_metrics["pt2_to_pte_top1_agreement"],
        ">=",
        policy["ppl"]["pt2_to_pte_min_top1_agreement"],
    )
    for pair, key, threshold in (
        (
            "hf_to_pt2",
            "pass_rate_drop_pp",
            policy["generation"]["hf_to_pt2_max_pass_rate_drop_pp"],
        ),
        (
            "pt2_to_pte",
            "pass_rate_drop_pp",
            policy["generation"]["pt2_to_pte_max_pass_rate_drop_pp"],
        ),
        (
            "hf_to_pte",
            "pass_rate_drop_pp",
            policy["generation"]["hf_to_pte_max_pass_rate_drop_pp"],
        ),
    ):
        _check(
            checks,
            f"{pair} pass-rate drop (pp)",
            generation_metrics[pair][key],
            "<=",
            threshold,
        )
    retention = generation_metrics["hf_to_pt2"]["quality_retention"]
    if retention is None:
        raise ValueError("HF pass rate is zero; quality retention is undefined")
    _check(
        checks,
        "HF→PT2 quality retention",
        retention,
        ">=",
        policy["generation"]["hf_to_pt2_min_quality_retention"],
    )
    for key, policy_key in (
        ("eos_rate_drop_pp", "max_eos_rate_drop_pp"),
        ("repetition_rate_increase_pp", "max_repetition_rate_increase_pp"),
        ("garbled_rate_increase_pp", "max_garbled_rate_increase_pp"),
    ):
        _check(
            checks,
            "HF→PTE " + key,
            generation_metrics["hf_to_pte"][key],
            "<=",
            policy["generation"][policy_key],
        )
    _check(checks, "critical new regressions", len(critical_regressions), "<=", 0)
    result = {
        "schema": "executorch.quantization.quality_gate.v1",
        "passed": all(check["passed"] for check in checks),
        "policy": policy,
        "checks": checks,
        "metrics": {"ppl": ppl_metrics, "generation": generation_metrics},
        "evidence": {
            "ppl": {
                engine: {
                    "root": run["root"],
                    "sha256": run["evidence_sha256"],
                    "identity_sha256": run["summary"].get("identity_sha256"),
                }
                for engine, run in ppl.items()
            },
            "generation": {
                engine: {
                    "root": run["root"],
                    "sha256": run["evidence_sha256"],
                    "run_fingerprint": run["summary"].get("run_fingerprint"),
                }
                for engine, run in generation.items()
            },
            "lineage": (
                {
                    "path": str(Path(lineage_path).resolve()),
                    "sha256": lineage.get("lineage_sha256"),
                }
                if lineage
                else None
            ),
            "policy_sha256": digest(policy),
        },
    }
    result["gate_sha256"] = digest(result)
    return result


def _regression_rows(result):
    critical = set(result["metrics"]["generation"]["critical_turn_ids"])
    return [
        {
            "comparison": comparison,
            "turn_id": turn_id,
            "critical": turn_id in critical,
        }
        for comparison in ("hf_to_pt2", "pt2_to_pte", "hf_to_pte")
        for turn_id in result["metrics"]["generation"][comparison]["regressions"]
    ]


def _markdown_report(result):
    lines = [
        "# 量化质量门禁\n\n",
        f"结论：**{'PASS' if result['passed'] else 'FAIL'}**\n\n",
        "| 检查 | 实际值 | 条件 | 阈值 | 结果 |\n",
        "|---|---:|:---:|---:|:---:|\n",
    ]
    for check in result["checks"]:
        lines.append(
            f"| {check['name']} | {check['actual']:.8g} | {check['relation']} | "
            f"{check['threshold']:.8g} | {'PASS' if check['passed'] else 'FAIL'} |\n"
        )
    lines.extend(
        [
            "\n## 三端 PPL / mean NLL\n\n",
            "| 端 | PPL | mean NLL |\n",
            "|---|---:|---:|\n",
        ]
    )
    for engine in ("hf", "pt2", "pte"):
        values = result["metrics"]["ppl"]["values"][engine]
        lines.append(
            f"| {engine.upper()} | {values['ppl']:.8g} | "
            f"{values['mean_nll']:.8g} |\n"
        )
    lines.extend(
        [
            "\n## 三端生成质量\n\n",
            "| 端 | 通过率 | EOS 率 | 重复率 | 乱码率 |\n",
            "|---|---:|---:|---:|---:|\n",
        ]
    )
    for engine in ("hf", "pt2", "pte"):
        rates = result["metrics"]["generation"]["rates"][engine]
        lines.append(
            f"| {engine.upper()} | {rates['pass_rate']:.4%} | "
            f"{rates['eos_rate']:.4%} | {rates['repetition_rate']:.4%} | "
            f"{rates['garbled_rate']:.4%} |\n"
        )
    lines.extend(
        [
            "\n## 逐题变化\n\n",
            "| 比较 | 新增回归 | 恢复 |\n",
            "|---|---:|---:|\n",
        ]
    )
    for comparison in ("hf_to_pt2", "pt2_to_pte", "hf_to_pte"):
        metrics = result["metrics"]["generation"][comparison]
        lines.append(
            f"| {comparison} | {len(metrics['regressions'])} | "
            f"{len(metrics['recoveries'])} |\n"
        )
    lines.append("\n逐题新增回归见 [regressions.csv](regressions.csv)。\n")
    return "".join(lines)


def audit_report(directory):
    root = Path(directory).resolve()
    saved = read_json(root / "gate.json")
    body = {key: value for key, value in saved.items() if key != "gate_sha256"}
    if (
        saved.get("schema") != "executorch.quantization.quality_gate.v1"
        or saved.get("gate_sha256") != digest(body)
        or saved.get("evidence", {}).get("policy_sha256") != digest(saved.get("policy"))
    ):
        raise ValueError("Quality-gate report hash is invalid")
    policy = validate_policy(saved["policy"])
    ppl = {
        engine: load_ppl(saved["evidence"]["ppl"][engine]["root"])
        for engine in ("hf", "pt2", "pte")
    }
    generation = {
        engine: generation_evidence(saved["evidence"]["generation"][engine]["root"])
        for engine in ("hf", "pt2", "pte")
    }
    lineage = saved["evidence"].get("lineage")
    lineage_path = Path(lineage["path"]) if lineage else None
    recomputed = evaluate_gate(policy, ppl, generation, lineage_path)
    if recomputed != saved:
        raise ValueError("Quality-gate report or source evidence changed")
    if read_json(root / "metrics.json") != saved["metrics"]:
        raise ValueError("Quality-gate metrics changed")
    with (root / "regressions.csv").open(encoding="utf-8-sig", newline="") as stream:
        actual = list(csv.DictReader(stream))
    expected = [
        {
            "comparison": row["comparison"],
            "turn_id": row["turn_id"],
            "critical": str(row["critical"]),
        }
        for row in _regression_rows(saved)
    ]
    if actual != expected:
        raise ValueError("Quality-gate regression report changed")
    if (root / "REPORT.md").read_text(encoding="utf-8") != _markdown_report(saved):
        raise ValueError("Quality-gate Markdown report changed")
    return saved


def write_report(output, result):
    with atomic_directory(output) as working:
        write_json(working / "gate.json", result)
        write_json(working / "metrics.json", result["metrics"])
        with (working / "regressions.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("comparison", "turn_id", "critical")
            )
            writer.writeheader()
            writer.writerows(_regression_rows(result))
        (working / "REPORT.md").write_text(_markdown_report(result), encoding="utf-8")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--policy", type=Path, required=True)
    for engine in ("hf", "pt2", "pte"):
        result.add_argument(f"--{engine}-ppl", type=Path, required=True)
        result.add_argument(f"--{engine}-generation", type=Path, required=True)
    result.add_argument("--lineage", type=Path)
    result.add_argument("--output", type=Path, required=True)
    return result


def main():
    args = parser().parse_args()
    try:
        policy = load_policy(args.policy)
        ppl = {
            engine: load_ppl(getattr(args, f"{engine}_ppl"))
            for engine in ("hf", "pt2", "pte")
        }
        generation = {
            engine: generation_evidence(getattr(args, f"{engine}_generation"))
            for engine in ("hf", "pt2", "pte")
        }
        result = evaluate_gate(policy, ppl, generation, args.lineage)
        if args.output.exists():
            if audit_report(args.output) != result:
                raise ValueError(
                    "Existing quality-gate report belongs to other evidence"
                )
        else:
            write_report(args.output, result)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        print(f"INVALID EVIDENCE: {error}", file=sys.stderr)
        return 2
    print(args.output / "REPORT.md")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
