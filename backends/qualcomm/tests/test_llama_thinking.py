# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_thinking_controller(tmp_path):
    """Check controller state transitions with address and undefined sanitizers."""
    root = Path(__file__).resolve().parents[3]
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    binary = tmp_path / "thinking_test"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            f"-I{root.parent}",
            str(
                root
                / "examples/qualcomm/oss_scripts/llama/runner/test/thinking_controller_test.cpp"
            ),
            "-o",
            str(binary),
        ],
        check=True,
    )
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "four combinations and boundaries passed" in result.stdout


@pytest.mark.parametrize("mode", ["kv", "hybrid"])
@pytest.mark.parametrize(
    "repair,budget", [(False, False), (True, False), (False, True), (True, True)]
)
def test_export_script_forwards_runtime_controls(
    mode, repair, budget, monkeypatch, tmp_path
):
    """Forward all enabled combinations from the Python CLI to the runner."""
    from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
        TEXT_DECODER,
        TOK_EMBEDDING,
        VISION_ENCODER,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.evaluator.device_evaluator import (
        DefaultEval,
        EvalBase,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.llama import _build_parser

    args = _build_parser().parse_args(
        [
            "--decoder_model",
            "lincal-1_7b",
            "--prompt",
            "hello",
            "--model_mode",
            mode,
            "-a",
            str(tmp_path),
            "--think_budget",
            "256",
            "--repair_think_eos" if repair else "--no-repair_think_eos",
            "--enable_think_budget" if budget else "--no-enable_think_budget",
        ]
    )
    monkeypatch.setattr(
        EvalBase,
        "_get_adb",
        lambda self: SimpleNamespace(
            output_folder="/data/local/tmp/think-test/outputs"
        ),
    )
    evaluator = DefaultEval(
        args,
        None,
        {
            TEXT_DECODER: "decoder.pte",
            TOK_EMBEDDING: "embedding.pte",
            VISION_ENCODER: "vision.pte",
        },
        "tokenizer.json",
        False,
        None,
    )
    if repair or budget:
        assert f"--repair_think_eos={str(repair).lower()}" in evaluator.runner_cmd
        assert f"--enable_think_budget={str(budget).lower()}" in evaluator.runner_cmd
        assert "--think_budget=256" in evaluator.runner_cmd
    else:
        assert "--repair_think_eos" not in evaluator.runner_cmd
