import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflow.sdk_runtime import deploy_runtime, runtime_files  # noqa: E402
from workflow.target import TargetConfig  # noqa: E402
from workflow.export_config import load_export_config  # noqa: E402
from workflow.export_pipeline import ExportPipeline  # noqa: E402
from workflow.config import parse_args  # noqa: E402


class Device:
    serial = "SM8850-test"

    def __init__(self):
        self.calls = []

    def check_target(self, target):
        self.calls.append(("target", target))

    def adb(self, *items):
        self.calls.append(items)


@pytest.mark.parametrize("soc,arch", [("SM8750", "V79"), ("SM8850", "V81")])
def test_runtime_deployment_selects_one_architecture_and_records_sdk(
    tmp_path, soc, arch
):
    sdk = tmp_path / "sdk"
    target = TargetConfig.resolve(soc)
    files = runtime_files(sdk, target)
    for index, path in enumerate(files):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([index]))
    device = Device()
    manifest = tmp_path / "runtime.json"
    metadata = deploy_runtime(
        sdk, target, device, "/data/local/tmp/qnn-runtime", manifest
    )
    assert device.calls[0] == ("target", target)
    assert device.calls[2] == ("push", "--sync", *files, "/data/local/tmp/qnn-runtime/")
    assert f"libQnnHtp{arch}Skel.so" in metadata["libraries"]
    assert f"libQnnHtp{arch}Stub.so" in metadata["libraries"]
    assert metadata["target"] == target.to_dict()
    assert json.loads(manifest.read_text()) == metadata


def test_runtime_fails_before_device_changes_when_sdk_is_incomplete(tmp_path):
    device = Device()
    with pytest.raises(FileNotFoundError, match="libQnnHtpV81Skel.so"):
        deploy_runtime(
            tmp_path,
            TargetConfig.resolve("SM8850"),
            device,
            "/data/local/tmp/runtime",
            tmp_path / "manifest.json",
        )
    assert not device.calls
    assert not (tmp_path / "manifest.json").exists()


def test_sm8850_profile_uses_sdk251_and_2048_context():
    config = load_export_config(ROOT / "configs/export_sm8850.json")
    assert config.soc_model == "SM8850" and config.dsp_arch == "v81"
    assert config.model_settings.context.max_seq_len == 2048
    assert config.model_settings.schedule.cache_experts == 64
    assert config.model_settings.schedule.read_chunk_kib == 512
    assert str(config.qnn_sdk_root).endswith("2.51.200.261021")
    assert config.deploy_runtime
    stages = ExportPipeline(config).stages
    assert [stage.name for stage in stages[:3]] == ["build", "runtime", "quantize"]
    assert "--sdk" in stages[1].command
    args = parse_args(["--config", str(ROOT / "configs/sm8850.json")])
    assert args.dsp_arch == "v81" and args.decode_capacity == 2048
    assert args.build_dir == config.build_dir
    assert args.artifacts == config.output
    assert args.model_settings.quantization.expert_bits == 2
