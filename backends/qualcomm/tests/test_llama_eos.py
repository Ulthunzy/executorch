# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import shlex
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


@pytest.mark.parametrize(
    "flags,expected",
    [([], False), (["--ignore_eos"], True), (["--no-ignore_eos"], False)],
)
@pytest.mark.parametrize("model_mode", ["kv", "hybrid", "lookahead"])
@pytest.mark.parametrize("is_multimodal", [False, True])
def test_ignore_eos_forwarded_to_runner(
    flags, expected, model_mode, is_multimodal, monkeypatch, tmp_path
):
    args = _build_parser().parse_args(
        [
            "--decoder_model",
            "lincal-1_7b",
            "--prompt",
            "hello",
            "--model_mode",
            model_mode,
            "-a",
            str(tmp_path),
            *flags,
        ]
    )
    assert args.ignore_eos is expected
    monkeypatch.setattr(
        EvalBase,
        "_get_adb",
        lambda self: SimpleNamespace(output_folder="/data/local/tmp/eos-test/outputs"),
    )
    evaluator = DefaultEval(
        args=args,
        decoder_model_config=None,
        pte_paths={
            TEXT_DECODER: "decoder.pte",
            TOK_EMBEDDING: "embedding.pte",
            VISION_ENCODER: "vision.pte",
        },
        runtime_tokenizer_path="tokenizer.json",
        is_multimodal=is_multimodal,
        dataset_builder=None,
    )
    command = shlex.split(evaluator.runner_cmd)
    runner = "qnn_multimodal_runner" if is_multimodal else "qnn_llama_runner"
    assert f"./{runner}" in command
    assert f"--ignore_eos={str(expected).lower()}" in command
    assert command[command.index("--seq_len") + 1] == str(args.max_seq_len)
