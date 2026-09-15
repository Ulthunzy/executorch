"""Workflow tests use tiny local commands, never models or devices."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import workflow  # noqa: E402
from common.registry import Tool  # noqa: E402


def configuration(tmp_path, steps=None):
    config = {
        "schema_version": 1,
        "variables": {"name": "a model with spaces"},
        "output_root": str(tmp_path / "results"),
        "steps": steps
        or [
            {
                "id": "matrix",
                "tool": "hf",
                "matrix": {"think": ["on", "off"], "rp": [1.0, 1.05, 1.1]},
                "args": ["--model", "${name}", "--repetition-penalty", "${rp}"],
                "append_when": [
                    {
                        "parameter": "think",
                        "equals": "off",
                        "args": ["--disable-thinking"],
                    }
                ],
                "outputs": ["${output}/${job}/summary.json"],
            }
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def arguments(path, **overrides):
    return argparse.Namespace(
        **(
            {
                "config": path,
                "all": True,
                "stage": None,
                "with_dependencies": False,
                "dry_run": False,
                "resume": False,
                "device_idle_confirmed": False,
            }
            | overrides
        )
    )


def test_matrix_preserves_decimal_penalty_and_literal_arguments(tmp_path):
    _, _, jobs = workflow.load_plan(configuration(tmp_path))
    assert len(jobs) == 6
    assert "rp-1.05" in jobs[1].id
    assert jobs[0].argv[-3] == "a model with spaces"
    assert "--disable-thinking" not in jobs[0].argv
    assert jobs[3].argv[-1] == "--disable-thinking"


@pytest.mark.parametrize(
    "change",
    [
        lambda config: config["steps"][0].update(tool="unknown"),
        lambda config: config["steps"][0].update(depends_on=["missing"]),
        lambda config: config["steps"][0].update(outputs=["${output}/collision.json"]),
        lambda config: config["steps"][0].update(args=["${misspelled}"]),
        lambda config: config["steps"][0].update(outputs=["/tmp/outside-result.json"]),
        lambda config: config["steps"][0].update(matrix={"rp": [1.0, 1.0]}),
        lambda config: config["steps"][0].update(matrix={"output": ["bad"]}),
        lambda config: config["steps"][0].update(matrix={"job": ["bad"]}),
        lambda config: config["variables"].update(job="bad"),
        lambda config: config["steps"][0].update(unrecognized=True),
    ],
)
def test_invalid_config_is_rejected(tmp_path, change):
    path = configuration(tmp_path)
    config = json.loads(path.read_text())
    change(config)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        workflow.load_plan(path)


def test_dry_run_has_no_side_effects(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry-run started a process")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    path = configuration(tmp_path)
    workflow.run_workflow(arguments(path, dry_run=True))
    assert not (tmp_path / "results").exists()


def test_resume_checks_inputs_outputs_and_configuration(tmp_path, monkeypatch):
    script = tmp_path / "worker.py"
    script.write_text(
        "import pathlib,sys\np=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('ok')\n"
    )
    source = tmp_path / "input.txt"
    source.write_text("source")
    monkeypatch.setitem(workflow.TOOLS, "tiny", Tool(script, "test worker"))
    path = configuration(
        tmp_path,
        [
            {
                "id": "tiny",
                "tool": "tiny",
                "args": ["${output}/done.txt"],
                "inputs": [str(source)],
                "outputs": ["${output}/done.txt"],
            }
        ],
    )
    workflow.run_workflow(arguments(path))
    output = tmp_path / "results"
    assert (
        json.loads((output / "workflow.json").read_text())["jobs"]["tiny"]["status"]
        == "complete"
    )
    workflow.run_workflow(arguments(path, resume=True))
    with pytest.raises(ValueError, match="already exists"):
        workflow.run_workflow(arguments(path))
    source.write_text("changed")
    with pytest.raises(ValueError, match="input/code changed"):
        workflow.run_workflow(arguments(path, resume=True))
    source.write_text("source")
    (output / "done.txt").write_text("tampered")
    with pytest.raises(ValueError, match="outputs changed"):
        workflow.run_workflow(arguments(path, resume=True))


def test_failed_job_is_not_complete(tmp_path, monkeypatch):
    script = tmp_path / "fail.py"
    script.write_text("raise SystemExit(9)\n")
    monkeypatch.setitem(workflow.TOOLS, "fail", Tool(script, "failure"))
    path = configuration(
        tmp_path,
        [{"id": "fail", "tool": "fail", "outputs": ["${output}/summary.json"]}],
    )
    with pytest.raises(subprocess.CalledProcessError):
        workflow.run_workflow(arguments(path))
    state = json.loads((tmp_path / "results/workflow.json").read_text())
    assert state["jobs"]["fail"]["status"] == "failed"


def test_selected_stage_requires_completed_dependencies(tmp_path):
    path = configuration(
        tmp_path,
        [
            {"id": "first", "tool": "hf", "outputs": ["${output}/first.json"]},
            {
                "id": "second",
                "tool": "hf",
                "depends_on": ["first"],
                "outputs": ["${output}/second.json"],
            },
        ],
    )
    with pytest.raises(ValueError, match="requires first"):
        workflow.run_workflow(arguments(path, all=False, stage=["second"]))


def test_resource_lock_excludes_other_workflows():
    with workflow.resource_lock("test-gpu"):
        with pytest.raises(ValueError, match="Resource busy"):
            with workflow.resource_lock("test-gpu"):
                pass


def test_atomic_output_retries_after_interruption_without_overwriting(tmp_path):
    from common.output import atomic_directory

    destination = tmp_path / "frozen"
    with pytest.raises(RuntimeError):
        with atomic_directory(destination) as working:
            (working / "partial.json").write_text("partial")
            raise RuntimeError("interrupted")
    assert not destination.exists()
    assert list(tmp_path.glob(".frozen.pending-*/partial.json"))
    with atomic_directory(destination) as working:
        (working / "manifest.json").write_text("complete")
    assert (destination / "manifest.json").read_text() == "complete"
    with pytest.raises(ValueError, match="already exists"):
        with atomic_directory(destination):
            pass


def test_summary_without_evidence_cannot_complete_workflow(tmp_path, monkeypatch):
    script = tmp_path / "worker.py"
    script.write_text(
        "import pathlib,sys\npathlib.Path(sys.argv[1]).write_text('{\"complete\":true}')\n"
    )
    monkeypatch.setitem(workflow.TOOLS, "tiny", Tool(script, "test worker"))
    path = configuration(
        tmp_path,
        [
            {
                "id": "tiny",
                "tool": "tiny",
                "args": ["${output}/summary.json"],
                "outputs": ["${output}/summary.json"],
            }
        ],
    )
    with pytest.raises(ValueError, match="evidence"):
        workflow.run_workflow(arguments(path))
    state = json.loads((tmp_path / "results/workflow.json").read_text())
    assert state["jobs"]["tiny"]["status"] == "failed"


def test_failed_device_job_requires_explicit_idle_confirmation(tmp_path, monkeypatch):
    script = tmp_path / "fail.py"
    script.write_text("raise SystemExit(9)\n")
    monkeypatch.setitem(workflow.TOOLS, "fail", Tool(script, "failure"))
    path = configuration(
        tmp_path,
        [
            {
                "id": "device",
                "tool": "fail",
                "resource": "adb:test-device",
                "outputs": ["${output}/done.txt"],
            }
        ],
    )
    with pytest.raises(subprocess.CalledProcessError):
        workflow.run_workflow(arguments(path))
    with pytest.raises(ValueError, match="device state is unknown"):
        workflow.run_workflow(arguments(path, resume=True))
    with pytest.raises(subprocess.CalledProcessError):
        workflow.run_workflow(arguments(path, resume=True, device_idle_confirmed=True))


def test_report_does_not_promote_historical_ppl_or_pending_review(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "validation_compare", ROOT / "reporting/compare.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"ppl": 12.5, "count": 10, "complete": True}))
    assert module.collect(tmp_path)["status"] == "unverified_or_incomplete"
    path.write_text(
        json.dumps(
            {"planned": 100, "attempted": 100, "reviewed": 50, "normal_rate": 0.9}
        )
    )
    assert module.collect(tmp_path)["normal_rate"] is None
    assert module.collect(tmp_path / "missing")["status"] == "missing"


@pytest.mark.parametrize("name", ["generation.example.json", "ppl.example.json"])
def test_shipped_config_parses(name):
    _, output, jobs = workflow.load_plan(ROOT / "configs" / name)
    assert output.is_relative_to(workflow.REPO / "llama-output")
    assert jobs
