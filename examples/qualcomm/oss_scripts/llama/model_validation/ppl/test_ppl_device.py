"""Opt-in real Android teacher-forcing checks against analytic next-token losses."""

import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture(scope="module")
def ppl_device(tmp_path_factory):
    serial = os.environ.get("ET_PPL_DEVICE")
    driver = os.environ.get("ET_PPL_DRIVER")
    libdir = os.environ.get("ET_PPL_LIBDIR")
    if not all((serial, driver, libdir)):
        pytest.skip("Set ET_PPL_DEVICE, ET_PPL_DRIVER and ET_PPL_LIBDIR")

    import torch
    from executorch.exir import ExecutorchBackendConfig, to_edge
    from executorch.exir.passes import MemoryPlanningPass
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast
    from model_validation.ppl import corpus

    root = tmp_path_factory.mktemp("ppl-device")
    names = ["<unk>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"]
    names += list("abcdefghijk")
    native = Tokenizer(BPE(dict(zip(names, range(16))), [], unk_token="<unk>"))
    native.pre_tokenizer = ByteLevel(add_prefix_space=False)
    native.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=native, unk_token="<unk>", additional_special_tokens=names[1:5]
    )
    tokenizer.save_pretrained(root / "tokenizer")
    text = root / "text.txt"
    text.write_text("".join(names[5 + i % 11] for i in range(167)))
    corpus.prepare(root / "tokenizer", text, 96, 64, root / "corpus")

    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("vocabulary", torch.arange(16))

        def forward(self, tokens, mask, window_mask, positions, k, v):
            next_token = (tokens + 1) % 16
            logits = torch.where(
                self.vocabulary == next_token.unsqueeze(-1), 10.0, -10.0
            )
            values = tokens.to(torch.float32).unsqueeze(1).unsqueeze(-1)
            values = values.expand(1, 1, tokens.shape[1], 2).clone()
            return logits, values.permute(0, 1, 3, 2).contiguous(), values

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
        methods[method] = torch.export.export(Fixture(), inputs)
    program = to_edge(
        methods, constant_methods={"get_n_layers": 1, "get_sliding_window": 8}
    ).to_executorch(
        config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False, alloc_graph_output=False
            )
        )
    )
    (root / "fixture.pte").write_bytes(program.buffer)
    return root, serial, driver, libdir


@pytest.mark.parametrize("mode", ["kv", "hybrid"])
def test_teacher_forced_losses_and_resume(ppl_device, mode, monkeypatch):
    from model_validation.common.audit import audit_run
    from model_validation.engines.pte import PTEEngine
    from model_validation.ppl import corpus, evaluate
    from model_validation.ppl.artifacts import read_json

    root, serial, driver, libdir = ppl_device
    output = root / mode
    args = evaluate.parser().parse_args(
        [
            "run",
            "--engine",
            "pte",
            "--corpus",
            str(root / "corpus"),
            "--model",
            str(root / "fixture.pte"),
            "--tokenizer",
            str(root / "tokenizer/tokenizer.json"),
            "--driver",
            str(Path(driver).resolve()),
            "--serial",
            serial,
            "--device-lib-dir",
            libdir,
            "--decoder",
            "lincal",
            "--mode",
            mode,
            "--output",
            str(output),
        ]
    )
    evaluate.run(args)
    _, tokens, windows = corpus.load_corpus(root / "corpus")
    baseline = math.log1p(15 * math.exp(-20))
    expected = []
    for window in windows:
        record = read_json(output / (window["id"] + ".json"))
        losses = [
            baseline + (0 if tokens[p] == (tokens[p - 1] + 1) % 16 else 20)
            for p in range(window["target_begin"], window["end"])
        ]
        assert record["token_nll"] == pytest.approx(losses, abs=1e-12)
        expected.extend(losses)
    summary = read_json(output / "summary.json")
    assert summary["count"] == len(tokens) - 1 == 166
    assert summary["nll_sum"] == pytest.approx(math.fsum(expected), abs=1e-10)
    assert summary["ppl"] == pytest.approx(math.exp(math.fsum(expected) / 166))
    audit_run(output)
    before = (output / "summary.json").read_bytes()
    monkeypatch.setattr(
        PTEEngine, "run", lambda *a: pytest.fail("Resume ran inference")
    )
    args.resume = True
    evaluate.run(args)
    assert (output / "summary.json").read_bytes() == before
