import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_chat  # noqa: E402
from workflow.target import TargetConfig  # noqa: E402
from executorch.backends.qualcomm.serialization.qc_schema import (  # noqa: E402
    _soc_info_table,
    HtpArch,
    QcomChipset,
)
from workflow.config import parse_args  # noqa: E402
from workflow.device import AndroidDevice, DevicePaths  # noqa: E402
from workflow.export_config import load_export_config  # noqa: E402
from workflow.export_pipeline import ExportPipeline  # noqa: E402
from workflow.session import session_command, run_session  # noqa: E402


@pytest.mark.parametrize(
    "soc,arch", [("SM8750", "v79"), ("SM8850", "v81"), ("SM8975", "v85")]
)
def test_target_reaches_export_and_runtime(soc, arch):
    config = load_export_config(soc_model=soc, device="selected-device")
    pipeline = ExportPipeline(config)
    assert config.model_settings.target == TargetConfig(soc)
    assert config.device == "selected-device"
    assert pipeline.environment()["MOE_DSP_ARCH"] == arch[1:]
    assert f"op-package/hexagon-{arch}/libQnnDeltaNetOpPackage_HTP.so" in str(
        pipeline.stages[0].outputs
    )
    args = parse_args(["--soc_model", soc, "--device", "selected-device"])
    assert args.dsp_arch == arch
    assert args.model_settings.target == config.model_settings.target


def test_mismatched_target_cannot_run_or_export():
    with pytest.raises(ValueError, match="requires v79"):
        load_export_config(soc_model="SM8750", dsp_arch="v85")
    with pytest.raises(ValueError, match="Artifact target"):
        TargetConfig("SM8750").check_manifest({})
    TargetConfig().check_manifest({})
    device = AndroidDevice("test")
    device.adb = lambda *args: "SM8850\n"
    with pytest.raises(ValueError, match="device reports"):
        device.check_target(TargetConfig())


def test_target_architecture_comes_from_backend_schema(monkeypatch):
    info = _soc_info_table[QcomChipset.SM8850]
    monkeypatch.setattr(info.htp_info, "htp_arch", HtpArch.V85)
    assert TargetConfig.resolve("SM8850").dsp_arch == "v85"
    with pytest.raises(ValueError, match="requires v85"):
        TargetConfig.resolve("SM8850", "v81")
    with pytest.raises(ValueError, match="requires an HTP architecture"):
        TargetConfig.resolve("SM8650")


def test_generation_flags_and_command_are_used():
    args = parse_args(
        [
            "--temp",
            "0.7",
            "--ignore-eos",
            "--interactive",
            "--no-stream",
            "--with-vision",
            "--prompt",
            "解释 'a'; $(not-a-command)",
        ]
    )
    assert args.model_settings.generation.temperature == 0.7
    assert args.model_settings.generation.ignore_eos
    assert args.model_settings.generation.interactive
    assert not args.model_settings.generation.stream
    paths = DevicePaths("/data/local/tmp/model", "/data/local/tmp/shared")
    command = session_command(args, paths, "/data/local/tmp/photo.jpg")
    tokens = shlex.split(command.split(" && ", 1)[1])
    assert tokens[tokens.index("--temperature") + 1] == "0.7"
    assert tokens[tokens.index("--prompt") + 1] == args.prompt
    assert tokens[tokens.index("--image-path") + 1] == "/data/local/tmp/photo.jpg"
    assert "--ignore-eos" in tokens and "--no-stream" in tokens
    assert "--interactive" in tokens and "--vision" in tokens


def test_deployment_commands_cannot_consume_chat_input(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr("workflow.device.subprocess.run", run)
    AndroidDevice("phone").adb("shell", "mkdir -p /data/local/tmp/chat")
    assert calls[0]["stdin"] == subprocess.DEVNULL


@pytest.mark.parametrize(
    "flags",
    [
        ["--temperature", "nan"],
        ["--temperature", "-1"],
        ["--runner", "diagnostic", "--temp", "0.7"],
        ["--runner", "diagnostic", "--interactive"],
        ["--runner", "chat", "--teacher-tokens", "/tmp/teacher.i64"],
        ["--image-path", "/tmp/image.jpg", "--no-with-vision"],
    ],
)
def test_invalid_generation_modes_are_rejected(flags):
    with pytest.raises(SystemExit):
        parse_args(flags)


def test_weight_root_selects_its_model_configuration(tmp_path):
    (tmp_path / "shared-model").mkdir()
    data = json.loads((ROOT / "configs/model.json").read_text())
    data["target"] = {"soc_model": "SM8850"}
    data["context"]["max_seq_len"] = 2048
    data["generation"].update(
        temperature=0.6, ignore_eos=True, interactive=True, stream=False
    )
    (tmp_path / "MODEL_CONFIG.json").write_text(json.dumps(data))
    args = parse_args(["--weights-path", str(tmp_path), "--no-interactive"])
    assert args.artifacts == tmp_path
    assert args.shared_model == tmp_path / "shared-model"
    assert args.decode_capacity == 2048
    assert args.dsp_arch == "v81"
    assert args.temperature == 0.6 and args.ignore_eos and not args.stream
    assert not args.interactive


def test_python_inference_function_passes_all_controls(monkeypatch):
    forwarded = []
    monkeypatch.setattr(run_chat, "main", lambda argv: forwarded.extend(argv))
    run_chat.run_inference(
        weights_path="/tmp/weights",
        device="phone",
        soc_model="SM8850",
        temperature=0.8,
        ignore_eos=True,
        image_path="/tmp/a.jpg",
        interactive=True,
        stream=False,
    )
    assert forwarded[forwarded.index("--temperature") + 1] == "0.8"
    assert forwarded[forwarded.index("--soc-model") + 1] == "SM8850"
    assert forwarded[forwarded.index("--weights-path") + 1] == "/tmp/weights"
    assert "--interactive" in forwarded and "--no-stream" in forwarded
    assert "--ignore-eos" in forwarded and "--image-path" in forwarded


def test_native_session_has_no_host_tokenization(tmp_path, monkeypatch):
    import workflow.session as session

    args = parse_args(["--output", str(tmp_path), "--interactive"])
    calls = []

    class FakeDevice:
        def __init__(self, serial):
            calls.append(("device", serial))

        def check_target(self, target):
            calls.append(("target", target))

        def adb(self, *items):
            calls.append(items)

        def terminal(self, command, log):
            calls.append(("terminal", command))
            return 0

    monkeypatch.setattr(session, "AndroidDevice", FakeDevice)
    monkeypatch.setattr(session, "select_plans", lambda *args: None)
    monkeypatch.setattr(session, "deploy_artifacts", lambda *args: None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    directory = run_session(args)
    record = json.loads((directory / "RUN.json").read_text())
    assert record["mode"] == "chat" and record["exit_code"] == 0
    assert sum(c[0] == "terminal" for c in calls) == 1
    assert "--interactive" in record["command"]
    assert args.prompt_tokens is None
    assert not (directory / "tokens.i64").exists()


def test_terminal_relays_utf8_without_waiting_for_newlines(
    tmp_path, monkeypatch, capsys
):
    actual_popen = subprocess.Popen
    commands = []

    def spawn(command, **options):
        commands.append(command)
        return actual_popen(
            [
                sys.executable,
                "-c",
                "import os;os.write(1,b'\\xe4');os.write(1,b'\\xb8\\xad\\xe6\\x96\\x87')",
            ],
            **options,
        )

    monkeypatch.setattr(subprocess, "Popen", spawn)
    log = tmp_path / "console.log"
    assert AndroidDevice("phone").terminal("run-model", log) == 0
    assert capsys.readouterr().out == "中文"
    assert log.read_bytes() == "中文".encode()
    assert commands == [["adb", "-s", "phone", "shell", "-T", "run-model"]]


def test_raw_teacher_forcing_keeps_the_diagnostic_path():
    args = parse_args(["--teacher-tokens", "/tmp/teacher.i64", "--ignore-eos"])
    assert args.runner == "diagnostic"


def test_old_runner_is_rejected_before_starting_a_chat(tmp_path):
    from workflow.artifacts import check_artifacts

    args = parse_args(["--build-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="Rebuild the native chat runner"):
        check_artifacts(args)


def test_inference_reads_exported_plans_without_importing_the_exporter(
    tmp_path, monkeypatch
):
    import workflow.plans as plans

    args = parse_args([])
    args.artifacts = tmp_path
    args.local_decode_pte = None
    args.local_prefill_pte = None
    directory = tmp_path / "plans"
    directory.mkdir()
    metadata = dict(
        version=1,
        ar_len=128,
        capacity=args.decode_capacity,
        max_prefill_tokens=args.model_settings.context.max_prefill_tokens,
        prefill_order=args.prefill_order,
        lengths=list(range(128, 1025, 128)),
        model_configuration=args.model_settings.to_dict(),
    )
    (directory / "PLANS.json").write_text(json.dumps(metadata))
    for name in ["decode", *[f"prefill-{n}" for n in metadata["lengths"]]]:
        (directory / f"{name}.pte").write_bytes(b"exported program")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    calls = []

    def check(path, args, phase, expected_tokens=None):
        assert path.is_file()
        calls.append(path.name)
        return {"phase": phase}

    monkeypatch.setattr(plans, "check_plan", check)
    monkeypatch.setitem(sys.modules, "export.program", None)
    monkeypatch.setitem(sys.modules, "export.weights", None)
    plans.select_plans(args)
    assert args.local_decode_pte == directory / "decode.pte"
    assert plans.load_prefill_bundle(args) == directory
    assert len(calls) == 9
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    metadata["capacity"] += 128
    (directory / "PLANS.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="export matching PTEs"):
        plans.load_prefill_bundle(args)


def test_missing_exported_plans_do_not_trigger_runtime_packing(tmp_path):
    import workflow.plans as plans

    args = parse_args([])
    args.artifacts = tmp_path
    args.local_decode_pte = None
    with pytest.raises(FileNotFoundError, match="run export.py first"):
        plans.select_plans(args)
    assert not (tmp_path / "plans").exists()


def test_deployment_failure_preserves_session_record(tmp_path, monkeypatch):
    import workflow.session as session

    args = parse_args(["--output", str(tmp_path)])

    class FakeDevice:
        def __init__(self, serial):
            pass

        def check_target(self, target):
            pass

        def adb(self, *args):
            return ""

    def failed(*args):
        raise RuntimeError("deployment probe failed")

    monkeypatch.setattr(session, "AndroidDevice", FakeDevice)
    monkeypatch.setattr(session, "select_plans", lambda *args: None)
    monkeypatch.setattr(session, "deploy_artifacts", failed)
    with pytest.raises(RuntimeError, match="deployment probe"):
        run_session(args)
    record = json.loads(next(tmp_path.glob("*/RUN.json")).read_text())
    assert "deployment probe failed" in record["error"]
    assert record["exit_code"] is None


def test_weight_override_discards_another_exports_plan(tmp_path):
    (tmp_path / "shared-model").mkdir()
    (tmp_path / "MODEL_CONFIG.json").write_text(
        (ROOT / "configs/model.json").read_text()
    )
    profile = json.loads((ROOT / "configs/sm8975.json").read_text())
    profile["local_decode_pte"] = "/tmp/other-export/decode.pte"
    profile["local_prefill_pte"] = "/tmp/other-export/prefill.pte"
    config = tmp_path / "profile.json"
    config.write_text(json.dumps(profile))
    args = parse_args(["--config", str(config), "--weights-path", str(tmp_path)])
    assert args.local_decode_pte is None and args.local_prefill_pte is None
