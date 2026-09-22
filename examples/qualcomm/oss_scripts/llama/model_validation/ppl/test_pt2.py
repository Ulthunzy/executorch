"""Validate a tiny exported PT2 on CPU, including deliberate future leakage."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_validation.engines.pt2 import PT2Engine


class TinySequence(torch.nn.Module):
    """Produce vocab logits using supplied causal masks or deliberate leakage."""

    def __init__(self, leak=False):
        super().__init__()
        self.leak = leak

    def forward(self, tokens, causal, sliding, positions, key, value):
        """Keep a small known input layout without any learned weights."""
        embedded = (tokens[..., None] == torch.arange(5, device=tokens.device)).float()
        logits = torch.matmul(
            (causal[0, 0, :, : tokens.shape[1]] == 0).float(), embedded
        )
        if self.leak:
            logits = logits + embedded.sum(1, keepdim=True)
        return logits, key, value


class PT2Test(unittest.TestCase):
    """Require inferred capacity, checked layout and causal logits on every call."""

    def setUp(self):
        """Use only tiny CPU graphs with an AR unrelated to existing models."""
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def exported(self, *, size=8, leak=False, bad_mask=False):
        """Export a flat Lincal-shaped interface without loading external assets."""
        args = (
            torch.zeros((1, size), dtype=torch.int32),
            torch.zeros(1, 1, size, size + int(bad_mask)),
            torch.zeros(1, 1, size, size + 2),
            torch.arange(size, dtype=torch.int32)[None],
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2),
        )
        ep = torch.export.export(TinySequence(leak), args)
        path = self.root / f"tiny-{size}-{leak}-{bad_mask}.pt2"
        torch.export.save(ep, path)
        return path

    def test_dynamic_capacity_and_prefix_logits(self):
        """Infer AR and sliding width and return the complete original prefix."""
        for size in (8, 12):
            for offload in (False, True):
                engine = PT2Engine(
                    self.exported(size=size),
                    device="cpu",
                    causality_offload=offload,
                )
                self.assertEqual(engine.max_input_length, size)
                self.assertEqual(engine.sliding_window, 3)
                ids = [1, 2, 3, 1]
                logits = engine(ids)
                expected = (
                    torch.nn.functional.one_hot(torch.tensor(ids), num_classes=5)
                    .float()
                    .cumsum(0)
                )
                self.assertTrue(torch.equal(logits, expected))
                self.assertTrue(engine.last_causality_check["passed"])
                self.assertEqual(len(engine.last_causality_check["checks"]), 6)
                self.assertTrue(engine.validate_causality([1])["passed"])

    def test_future_leakage_is_rejected_without_patch(self):
        """Refuse leakage both into padding and from within the supplied prefix."""
        engine = PT2Engine(self.exported(leak=True), device="cpu")
        for ids in ([1, 2, 3], [1] * engine.max_input_length):
            with self.assertRaisesRegex(
                ValueError, "masked-future causality check failed"
            ):
                engine(ids)
            self.assertFalse(engine.last_causality_check["passed"])

    def test_unsupported_shape_dtype_and_length(self):
        """Reject unsupported interfaces rather than silently adapting graphs."""
        with self.assertRaisesRegex(ValueError, "shapes or dtypes"):
            PT2Engine(self.exported(bad_mask=True), device="cpu")
        path = self.exported()
        with self.assertRaisesRegex(ValueError, "float32 only"):
            PT2Engine(path, device="cpu", dtype="bfloat16")
        engine = PT2Engine(path, device="cpu")
        for ids in ([], [1] * 9, [-1], [2**32], [1.5], [[1]]):
            with self.assertRaises(ValueError):
                engine(ids)


if __name__ == "__main__":
    unittest.main()
