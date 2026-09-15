# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in portable-PTE tests through the real Android QNN text runner."""

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def device_fixture(tmp_path_factory):
    """Deploy deterministic models and a runner into an isolated device directory."""
    device = os.environ.get("ET_THINKING_DEVICE")
    driver = os.environ.get("ET_THINKING_DRIVER")
    if not device or not driver:
        pytest.skip("Set ET_THINKING_DEVICE and ET_THINKING_DRIVER to run on Android")
    import torch
    from executorch.exir import ExecutorchBackendConfig, to_edge
    from executorch.exir.passes import MemoryPlanningPass
    from transformers import AutoTokenizer

    root = tmp_path_factory.mktemp("thinking-device")
    tokenizer_dir = Path(os.environ["ET_THINKING_TOKENIZER_DIR"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    opening, closing, eos, answer, repeat = [
        tokenizer.encode(s, add_special_tokens=False)[0]
        for s in ("<think>", "</think>", "<|im_end|>", "OK", "a")
    ]
    vocab = max(tokenizer.get_vocab().values()) + 1

    class Fixture(torch.nn.Module):
        def __init__(self, loop):
            super().__init__()
            self.loop = loop
            self.register_buffer("vocabulary", torch.arange(vocab))

        def forward(self, tokens, mask, window_mask, positions, k, v):
            next_token = torch.full_like(tokens, opening)
            next_token = torch.where(
                tokens == opening, repeat if self.loop else eos, next_token
            )
            if self.loop:
                next_token = torch.where(tokens == repeat, repeat, next_token)
            next_token = torch.where(tokens == closing, answer, next_token)
            next_token = torch.where(tokens == answer, eos, next_token)
            logits = torch.where(
                self.vocabulary == next_token.unsqueeze(-1), 10.0, -10.0
            )
            values = tokens.to(torch.float32).unsqueeze(1).unsqueeze(-1)
            values = values.expand(1, 1, tokens.shape[1], 2).clone()
            return logits, values.permute(0, 1, 3, 2).contiguous(), values

    def adb(*args):
        return subprocess.run(
            ["adb", "-s", device, *map(str, args)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout

    remote = adb("shell", "mktemp -d /data/local/tmp/et-thinking-test.XXXXXX").strip()
    assert remote.startswith("/data/local/tmp/et-thinking-test.")
    adb("push", driver, remote + "/driver")
    adb("push", tokenizer_dir / "tokenizer.json", remote + "/tokenizer.json")
    adb("shell", f"chmod 755 {shlex.quote(remote)}/driver")
    for name, loop in (("early", False), ("loop", True)):
        methods = {}
        for method, ar in (("prefill_forward", 4), ("kv_forward", 1)):
            inputs = (
                torch.zeros(1, ar, dtype=torch.int64),
                torch.zeros(1, 1, ar, 96),
                torch.zeros(1, 1, ar, 7 + ar),
                torch.zeros(1, ar, dtype=torch.int32),
                torch.zeros(1, 1, 2, 7),
                torch.zeros(1, 1, 7, 2),
            )
            methods[method] = torch.export.export(Fixture(loop), inputs)
        program = to_edge(
            methods, constant_methods={"get_n_layers": 1, "get_sliding_window": 8}
        ).to_executorch(
            config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                )
            )
        )
        path = root / f"{name}.pte"
        path.write_bytes(program.buffer)
        adb("push", path, remote + f"/{name}.pte")

    def run(mode, kind, repair, budget, prefix=False, max_new=20):
        case = f"{mode}-{kind}-{repair}-{budget}-{prefix}-{max_new}"
        settings = dict(
            model=remote + f"/{kind}.pte",
            tokenizer=remote + "/tokenizer.json",
            decoder="lincal",
            mode=mode,
            seq_len=96,
            max_new_tokens=max_new,
            temperature=0,
            repetition_penalty=1.0,
            repeat_last_n=128,
            repair_think_eos=repair,
            enable_think_budget=budget,
            think_budget=2,
        )
        prompt = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        if prefix:
            prompt += "<think>"
        session = dict(
            id=case,
            turns=[
                dict(id=f"T{i}", turn=i, prompt="你好", formatted_prompt=prompt)
                for i in (1, 2)
            ],
        )
        for filename, data in (("config", settings), ("session", session)):
            path = root / f"{case}.{filename}.json"
            path.write_text(json.dumps(data, ensure_ascii=False))
            adb("push", path, remote + f"/{filename}.json")
        libdir = shlex.quote(os.environ["ET_THINKING_LIBDIR"])
        command = (
            f"mkdir {remote}/{case} && LD_LIBRARY_PATH={libdir} ADSP_LIBRARY_PATH={libdir} "
            f"{remote}/driver {remote}/config.json {remote}/session.json {remote}/{case} 2>&1"
        )
        logs = adb("shell", command)
        (root / f"{case}.log").write_text(logs)
        records = [
            json.loads(adb("shell", f"cat {remote}/{case}/T{i}.json")) for i in (1, 2)
        ]
        assert all(r["error_code"] == 0 for r in records), (logs, records)
        assert "True SWA enabled" in logs
        return records, logs

    return run


@pytest.mark.parametrize("mode", ["kv", "hybrid"])
@pytest.mark.parametrize(
    "repair,budget", [(False, False), (True, False), (False, True), (True, True)]
)
def test_early_eos(device_fixture, mode, repair, budget):
    """Consume a repaired closing token before producing the answer."""
    for prefix in (False, True):
        records, logs = device_fixture(mode, "early", repair, budget, prefix)
        for record in records:
            assert ("</think>" in record["response"]) is repair
            assert ("OK" in record["response"]) is repair
            assert record["response"].endswith("<|im_end|>")
        assert ("reason=early_eos" in logs) is repair


@pytest.mark.parametrize("mode", ["kv", "hybrid"])
@pytest.mark.parametrize(
    "repair,budget", [(False, False), (True, False), (False, True), (True, True)]
)
def test_budget_and_swa_continuation(device_fixture, mode, repair, budget):
    """Apply independent controls across two turns and SWA cache wraparound."""
    records, logs = device_fixture(mode, "loop", repair, budget)
    for record in records:
        if budget:
            assert record["response"] == "<think>aa</think>OK<|im_end|>"
        else:
            assert record["response"] == "<think>" + "a" * 19
    assert ("reason=think_budget" in logs) is budget


@pytest.mark.parametrize("mode", ["kv", "hybrid"])
def test_total_limit_remains_authoritative(device_fixture, mode):
    """Reserve closure space without exceeding the caller's output limit."""
    records, _ = device_fixture(mode, "loop", True, True, max_new=3)
    assert all(r["response"] == "<think></think>OK" for r in records)
