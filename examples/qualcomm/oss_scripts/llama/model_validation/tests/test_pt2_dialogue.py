"""Tiny CPU stand-ins exercise dialogue bookkeeping, not QNN model accuracy."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from pt2_benchmark import generate_turn  # noqa: E402


class Tokenizer:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        value = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
        return value if enable_thinking else value + "<think>\n\n</think>\n\n"

    def encode(self, value, **kwargs):
        return [2, 3]

    def decode(self, values, **kwargs):
        return "answer<|im_end|>"


class Engine:
    last_causality_check = {"passed": True}

    def __init__(self):
        self.inputs = []

    def __call__(self, ids):
        self.inputs.append(list(ids))
        result = torch.zeros((len(ids), 8))
        result[-1, 7] = 10
        return result


def options(**changes):
    return argparse.Namespace(
        **(
            {
                "seq_len": 16,
                "disable_thinking": False,
                "repetition_penalty": 1.0,
                "repeat_last_n": 128,
            }
            | changes
        )
    )


def test_second_turn_retains_first_output_and_eos():
    engine, tokenizer = Engine(), Tokenizer()
    first, history = generate_turn(
        engine,
        tokenizer,
        [],
        {"id": "T01", "turn": 1, "prompt": "你好"},
        options(),
        {7},
    )
    second, history = generate_turn(
        engine,
        tokenizer,
        history,
        {"id": "T02", "turn": 2, "prompt": "你是谁"},
        options(),
        {7},
    )
    assert first["stop_reason"] == second["stop_reason"] == "eos"
    assert engine.inputs == [[2, 3], [2, 3, 7, 2, 3]]
    assert history == [2, 3, 7, 2, 3, 7]
    assert second["prompt_payload"].startswith("\n<|im_start|>user")


def test_context_full_does_not_invoke_engine_or_commit_new_input():
    engine = Engine()
    record, history = generate_turn(
        engine, Tokenizer(), [2, 3, 7], {"prompt": "你好"}, options(seq_len=4), {7}
    )
    assert history == [2, 3, 7]
    assert record["stop_reason"] == "context_full"
    assert engine.inputs == []


def test_thinking_off_changes_only_the_incremental_prefix():
    record, _ = generate_turn(
        Engine(),
        Tokenizer(),
        [],
        {"prompt": "你好"},
        options(disable_thinking=True),
        {7},
    )
    assert record["prompt_payload"].endswith("<think>\n\n</think>\n\n")
    assert record["generated_ids"] == [7]
