# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def driver_options(tmp_path_factory):
    """Build the standalone runtime-option parser fixture."""
    root = Path(__file__).resolve().parents[3]
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    binary = tmp_path_factory.mktemp("driver_options") / "driver_options_test"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{root.parent}",
            f"-I{root / 'extension/llm/tokenizers/third-party/json/single_include'}",
            str(
                root
                / "examples/qualcomm/oss_scripts/llama/model_validation/benchmarks/test_driver_options.cpp"
            ),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize(
    "flags,updates",
    [
        ([], {}),
        (
            ["--repetition-penalty", "1.0", "--repeat-last-n", "64"],
            {"repetition_penalty": 1.0, "repeat_last_n": 64},
        ),
        (
            ["--repetition_penalty=1.2", "--repeat_last_n=128"],
            {"repetition_penalty": 1.2, "repeat_last_n": 128},
        ),
        (["--temp", "0.6"], {"temperature": 0.6}),
        (["--temperature=0"], {"temperature": 0}),
        (["--repeat-last-n", "128", "--repeat-last-n=64"], {"repeat_last_n": 64}),
        (["--repair-think-eos=1"], {"repair_think_eos": True}),
        (["--enable-think-budget=false"], {"enable_think_budget": False}),
        (
            ["--repair-think-eos=0", "--enable-think-budget=1", "--think-budget=512"],
            {
                "repair_think_eos": False,
                "enable_think_budget": True,
                "think_budget": 512,
            },
        ),
        (
            ["--repair_think_eos=true", "--enable_think_budget=true"],
            {"repair_think_eos": True, "enable_think_budget": True},
        ),
    ],
)
def test_sampling_overrides_only_requested_fields(driver_options, flags, updates):
    """Keep unrequested JSON fields unchanged when applying CLI overrides."""
    config = {
        "model": "/unchanged/model.pte",
        "temperature": 0,
        "repetition_penalty": 1.0,
        "repeat_last_n": 256,
        "seq_len": 4096,
        "ignore_eos": False,
        "thinking_disabled": False,
    }
    result = subprocess.run(
        [str(driver_options), *flags],
        input=json.dumps(config),
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == dict(config, **updates)


def test_legacy_json_keeps_default_window(driver_options):
    """Preserve the legacy repetition-window default."""
    result = subprocess.run(
        [str(driver_options)],
        input='{"temperature": 0, "repetition_penalty": 1.2}',
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["repeat_last_n"] == 256


@pytest.mark.parametrize(
    "flags",
    [
        ["--repeat-last-n"],
        ["--repeat-last-n=0"],
        ["--repeat-last-n=-1"],
        ["--repeat-last-n=64.5"],
        ["--repeat-last-n=2147483648"],
        ["--repeat-last-n=true"],
        ["--repeat-last-n=NaN"],
        ["--repetition-penalty=0.9"],
        ["--repetition-penalty=nan"],
        ["--repetition-penalty=1e40"],
        ["--temp=-1"],
        ["--temp=inf"],
        ["--temp=1e999"],
        ["--temp="],
        ["--temp", "--repeat-last-n", "64"],
        ["--typo=64"],
        ["64"],
        ["--repair-think-eos=2"],
        ["--enable-think-budget=0.5"],
        ["--repair-think-eos"],
        ["--think-budget=0"],
        ["--think-budget=-1"],
        ["--think-budget=true"],
        ["--think-budget=2147483648"],
    ],
)
def test_invalid_options_fail_before_inference(driver_options, flags):
    """Reject invalid runtime controls before loading a model."""
    result = subprocess.run(
        [str(driver_options), *flags],
        input='{"temperature": 0, "repetition_penalty": 1.2}',
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr
    assert not result.stdout
