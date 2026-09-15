# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module", params=[True, False])
def prompt_logger(request, tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    root = Path(__file__).resolve().parents[3]
    binary = tmp_path_factory.mktemp("prompt_logging") / "prompt_logging_test"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            *([] if request.param else ["-Wno-unused-parameter"]),
            f"-DET_LOG_ENABLED={int(request.param)}",
            f"-I{root.parent}",
            str(root / "runtime/platform/log.cpp"),
            str(
                root
                / "examples/qualcomm/oss_scripts/llama/runner/test/prompt_logging_test.cpp"
            ),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return binary, request.param


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "\n<|im_start|>user\n你是谁<|im_end|>\n<|im_start|>assistant\n",
        "你好🙂é" * 4096,
        "x" * 159 + "🙂中文" * 100,
        "x" * 160 + "🙂中文" * 100,
        'quotes " \\ 100% %s %n\t\r\n' * 100,
        "".join(map(chr, range(32))) * 100 + "\x7f",
        "\nIDENTITY_TURN_END T01\nReached EOS; stopping generation\n",
    ],
    ids=[
        "empty",
        "chatml",
        "second-turn",
        "long-unicode",
        "utf8-boundary-before",
        "utf8-boundary-after",
        "escapes",
        "control-characters",
        "fake-runtime-log",
    ],
)
def test_complete_prompt_survives_native_log_buffer(prompt_logger, prompt):
    binary, enabled = prompt_logger
    result = subprocess.run(
        [str(binary)], input=prompt.encode("utf-8"), capture_output=True, check=True
    )
    assert not result.stderr
    if not enabled:
        assert not result.stdout
        return
    lines = result.stdout.decode("utf-8", errors="strict").splitlines()
    assert lines[0] == f"[QNN_PROMPT] begin: bytes={len(prompt.encode('utf-8'))}"
    assert lines[-1] == "[QNN_PROMPT] end"
    restored = bytearray()
    for line in lines[1:-1]:
        match = re.fullmatch(r"\[QNN_PROMPT\] offset=(\d+) text=(.+)", line)
        assert match, line
        assert int(match[1]) == len(restored)
        restored.extend(json.loads(match[2]).encode("utf-8"))
    assert restored == prompt.encode("utf-8")
