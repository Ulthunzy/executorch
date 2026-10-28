import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflow.config import parse_args
from workflow.deployment import deploy_vision


class Device:
    def __init__(self, digest):
        self.digest, self.calls = digest, []

    def adb(self, *args):
        self.calls.append(args)
        return self.digest + "  /data/local/tmp/vision.bin\n"


@pytest.mark.parametrize("reuse", [False, True])
def test_vision_transport_reuses_only_verified_binary(tmp_path, reuse):
    (tmp_path / "vision.bin").write_bytes(b"real compiled fixture")
    args = SimpleNamespace(
        vision_artifacts=tmp_path,
        runtime_vision_bin="/data/local/tmp/vision.bin" if reuse else None,
    )
    device = Device(hashlib.sha256((tmp_path / "vision.bin").read_bytes()).hexdigest())
    deploy_vision(
        args, device, "/data/local/tmp/session", "/data/local/tmp/session/prefill"
    )
    copies = [
        call
        for call in device.calls
        if call[0] == "push" and Path(call[1]).name == "vision.bin"
    ]
    assert len(copies) == (0 if reuse else 1)
    assert any(
        call[0] == "push" and Path(call[1]).name == "VISION.json"
        for call in device.calls
    )
    assert (
        any(call[0] == "shell" and "ln -sfn" in call[1] for call in device.calls)
        == reuse
    )


def test_incorrect_device_vision_is_rejected_before_alias(tmp_path):
    (tmp_path / "vision.bin").write_bytes(b"selected artifact")
    args = SimpleNamespace(
        vision_artifacts=tmp_path, runtime_vision_bin="/data/local/tmp/vision.bin"
    )
    device = Device("0" * 64)
    with pytest.raises(ValueError, match="differs"):
        deploy_vision(
            args, device, "/data/local/tmp/session", "/data/local/tmp/session/prefill"
        )
    assert len(device.calls) == 1


@pytest.mark.parametrize(
    "arguments",
    [
        ["--runtime-vision-bin", "/data/local/tmp/v.bin", "--no-with-vision"],
        ["--runtime-vision-bin", "../v.bin", "--with-vision"],
        ["--runtime-vision-bin", "/data/local/tmp/v;rm", "--with-vision"],
    ],
)
def test_invalid_runtime_vision_selection_is_rejected(arguments):
    with pytest.raises(SystemExit):
        parse_args(arguments)
