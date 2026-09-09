# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
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


def test_native_repetition_penalty(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    root = Path(__file__).resolve().parents[3]
    binary = tmp_path / "repetition_penalty_test"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root.parent),
            str(
                root
                / "examples/qualcomm/oss_scripts/llama/runner/test/repetition_penalty_test.cpp"
            ),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize("value", ["0", "-1", "0.9", "nan", "inf", "-inf"])
def test_invalid_repetition_penalty(value):
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            [
                "--decoder_model",
                "lincal-1_7b",
                "--prompt",
                "hello",
                f"--repetition_penalty={value}",
            ]
        )


@pytest.mark.parametrize("penalty", [None, 1.0, 1.1])
@pytest.mark.parametrize("mode", ["kv", "hybrid", "lookahead"])
@pytest.mark.parametrize("multimodal", [False, True])
def test_penalty_forwarded(penalty, mode, multimodal, monkeypatch, tmp_path):
    flags = [] if penalty is None else ["--repetition_penalty", str(penalty)]
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
            *flags,
        ]
    )
    monkeypatch.setattr(
        EvalBase, "_get_adb", lambda self: SimpleNamespace(output_folder="outputs")
    )
    evaluator = DefaultEval(
        args=args,
        decoder_model_config=None,
        pte_paths={
            TEXT_DECODER: "d.pte",
            TOK_EMBEDDING: "e.pte",
            VISION_ENCODER: "v.pte",
        },
        runtime_tokenizer_path="tokenizer.json",
        is_multimodal=multimodal,
        dataset_builder=None,
    )
    command = shlex.split(evaluator.runner_cmd)
    assert float(command[command.index("--repetition_penalty") + 1]) == (penalty or 1.0)
    assert "--ignore_eos=false" in command
