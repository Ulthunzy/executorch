# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise old standalone entry points without importing ExecuTorch."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

LLAMA = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "relative,function,group",
    [
        ("self_identity_eval/benchmark.py", "session_input", "benchmarks"),
        ("self_identity_eval/hf_benchmark.py", "chatml_prompt", "benchmarks"),
        ("calibration/build_think_calibration.py", "format_chatml", "calibration"),
    ],
)
def test_spec_import_uses_real_source_and_patchable_globals(
    relative, function, group, monkeypatch
):
    monkeypatch.syspath_prepend(str(LLAMA / "model_validation/benchmarks"))
    spec = importlib.util.spec_from_file_location(
        "legacy_spec_fixture", LLAMA / relative
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = LLAMA / "model_validation" / group / Path(relative).name
    assert Path(module.__file__) == target
    assert module.__spec__.origin == str(target)
    exported = vars(module)[function]
    assert exported.__code__.co_filename == str(target)
    assert exported.__globals__ is vars(module)


@pytest.mark.parametrize(
    "relative",
    [
        "self_identity_eval/benchmark.py",
        "self_identity_eval/build_driver.py",
        "self_identity_eval/hf_benchmark.py",
        "self_identity_eval/sample_benchmark.py",
        "calibration/build_think_calibration.py",
        "calibration/validate_think_calibration.py",
    ],
)
def test_legacy_help_without_site_packages(relative, tmp_path):
    result = subprocess.run(
        [sys.executable, "-S", str(LLAMA / relative), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
