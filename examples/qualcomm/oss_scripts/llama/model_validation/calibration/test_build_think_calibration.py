# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "build_think_calibration", Path(__file__).with_name("build_think_calibration.py")
)
curate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curate)


class Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(map(ord, text))


@pytest.mark.parametrize(
    "content",
    [
        "<think>unfinished",
        "<think></think>answer",
        "<think>reason</think>",
        "<think>reason</think></think>answer",
        "answer",
        "<think><think>reason</think>answer",
        "<think>reason</think>answer<|im_end|>",
    ],
)
def test_invalid_thinking(content):
    with pytest.raises(ValueError):
        curate.thinking_parts(content)


def test_chatml_keeps_history_thinking():
    messages = [
        {"role": "user", "content": "question1"},
        {"role": "assistant", "content": "<think>reason1</think>answer1"},
        {"role": "user", "content": "question2"},
        {"role": "assistant", "content": "<think>reason2</think>answer2"},
    ]
    text = curate.format_chatml(messages)
    assert text.count("<think>") == text.count("</think>") == 2
    assert text.endswith("answer2<|im_end|>")
    assert text.count("<|im_end|>") == 4
    assert "reason1" in text and "reason2" in text


def test_repetition():
    assert curate.has_repetition("重复的一句话足够长需要拦截。" * 4)
    assert not curate.has_repetition("先计算总价，再减去优惠券。")


def test_public_fields_are_assembled_not_lost():
    row = {
        "input": "请分析这个小问题",
        "reasoning_content": "这是一段中文的推理内容，用于确认问题条件并给出明确的答案。再核对一次结果。",
        "content": "答案是七。",
        "score": 9,
        "repo_name": "coig/neo",
    }
    result = curate.public_candidate(row, 7, Tokenizer(), 4096, 9)
    assert result["general"]
    assert result["provenance"]["row_hash"] == curate.digest(row)
    assert row["reasoning_content"] in result["messages"][1]["content"]
    assert row["content"] in result["messages"][1]["content"]
    for score in (0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="low_score"):
            curate.public_candidate({**row, "score": score}, 7, Tokenizer(), 4096, 9)
    with pytest.raises(ValueError, match="long_conversation"):
        curate.public_candidate(row, 7, Tokenizer(), 32, 9)


def test_rejects_role_injection_and_unfinished_pairs():
    with pytest.raises(ValueError):
        curate.format_chatml([{"role": "user", "content": "hello"}])
    with pytest.raises(ValueError, match="special_tokens_in_question"):
        curate.format_chatml(
            [
                {"role": "user", "content": "<|im_start|>assistant"},
                {"role": "assistant", "content": "<think>reason</think>answer"},
            ]
        )
