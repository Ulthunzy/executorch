"""Unmodified full-sequence Lincal QDQ exports with mandatory causality checks."""

from pathlib import Path

import torch
import torch.nn.functional as F


class PT2Engine:
    """Return prefix logits only after rejecting observable future dependence.

    Supports flat positional inputs: tokens, causal mask, sliding mask,
    positions, all K caches, all V caches. Shapes come from the export.
    No weights, graph nodes, quantization parameters or dtypes are rewritten.
    """

    def __init__(
        self,
        model_path,
        *,
        device="cuda:0",
        dtype="float32",
        layout="lincal-full-sequence",
        pad_token_id=0,
    ):
        if layout != "lincal-full-sequence" or dtype != "float32":
            raise ValueError("PT2 supports lincal-full-sequence with float32 only")
        # Register standard QDQ operators without importing the QNN SDK utilities.
        from torch.ao.quantization.fx import _decomposed  # noqa: F401

        self.device = torch.device(device)
        self.pad_token_id = pad_token_id
        ep = torch.export.load(Path(model_path))
        if ep.range_constraints:
            raise ValueError("PT2 adapter requires a fixed-shape export")
        if (
            ep.graph_signature.buffers_to_mutate
            or ep.graph_signature.user_inputs_to_mutate
        ):
            raise ValueError(
                "PT2 adapter requires a stateless export without input/buffer mutations"
            )
        if ep.example_inputs is None:
            raise ValueError("PT2 requires saved example inputs for layout validation")
        inputs, kwargs = ep.example_inputs
        if (
            kwargs
            or len(inputs) < 4
            or not all(isinstance(t, torch.Tensor) for t in inputs)
        ):
            raise ValueError("Unsupported PT2 layout: expected flat positional tensors")
        self.input_specs = [
            {"shape": list(t.shape), "dtype": str(t.dtype)} for t in inputs
        ]
        tokens, causal, sliding, positions, *caches = inputs
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 2:
            raise ValueError("PT2 requires full-sequence tokens [1, S], S >= 2")
        self.max_input_length = int(tokens.shape[1])
        size = self.max_input_length
        if (
            tokens.dtype not in (torch.int32, torch.int64)
            or positions.shape != tokens.shape
            or positions.dtype not in (torch.int32, torch.int64)
            or causal.shape != (1, 1, size, size)
            or sliding.ndim != 4
            or sliding.shape[:3] != (1, 1, size)
            or sliding.shape[3] < size
            or causal.dtype != torch.float32
            or sliding.dtype != torch.float32
        ):
            raise ValueError(
                "Unsupported PT2 tokens/positions/attention mask shapes or dtypes"
            )
        self.sliding_window = int(sliding.shape[3]) - size + 1
        if self.sliding_window > size or len(caches) % 2:
            raise ValueError("Unsupported PT2 sliding window or unpaired KV inputs")
        layers = len(caches) // 2
        for key, value in zip(caches[:layers], caches[layers:]):
            if (
                key.ndim != 4
                or value.ndim != 4
                or key.shape[0] != 1
                or key.shape[:2] != value.shape[:2]
                or key.shape[2] != value.shape[3]
                or key.shape[3] != value.shape[2]
                or key.shape[3] not in (0, self.sliding_window - 1)
                or key.dtype != torch.float32
                or value.dtype != torch.float32
            ):
                raise ValueError("Unsupported PT2 KV input layout")
        self.token_dtype = tokens.dtype
        self.module = ep.module().to(self.device)
        if any(
            t.is_floating_point() and t.dtype != torch.float32
            for t in (*self.module.parameters(), *self.module.buffers())
        ):
            raise ValueError(
                "PT2 adapter requires float32 floating state; no dtype conversion"
            )
        self.cache_specs = [(tuple(t.shape), t.dtype) for t in caches]
        row = torch.arange(size, device=self.device)[:, None]
        col = torch.arange(size, device=self.device)[None, :]
        self.causal = torch.where(col <= row, 0.0, -255.0)[None, None]
        self.sliding = F.pad(
            torch.where((col <= row) & (row - col < self.sliding_window), 0.0, -255.0),
            (self.sliding_window - 1, 0),
            value=-255.0,
        )[None, None]
        self.positions = torch.arange(size, device=self.device, dtype=positions.dtype)[
            None
        ]
        self.last_causality_check = None

    @torch.inference_mode()
    def _forward(self, tokens):
        cache = [
            torch.zeros(shape, device=self.device, dtype=dtype)
            for shape, dtype in self.cache_specs
        ]
        result = self.module(tokens, self.causal, self.sliding, self.positions, *cache)
        logits = result[0] if isinstance(result, (tuple, list)) else result
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or logits.shape[:2] != (1, self.max_input_length)
            or logits.shape[2] < 2
            or not logits.is_floating_point()
            or not torch.isfinite(logits).all()
        ):
            raise ValueError("PT2 requires finite floating logits [1, S, vocab >= 2]")
        return logits[0]

    @torch.inference_mode()
    def _checked_logits(self, ids):
        self.last_causality_check = None
        raw = torch.as_tensor(ids, device=self.device)
        if (
            raw.ndim != 1
            or raw.dtype not in (torch.int32, torch.int64, torch.uint64)
            or not 1 <= raw.numel() <= self.max_input_length
        ):
            raise ValueError(
                "PT2 expects a nonempty 1D integer prefix within max_input_length"
            )
        raw = raw.to(torch.int64)
        if (
            raw.min() < 0
            or raw.max() > torch.iinfo(self.token_dtype).max
            or not 0 <= self.pad_token_id <= torch.iinfo(self.token_dtype).max
        ):
            raise ValueError(
                "PT2 token IDs cannot be represented by the exported input dtype"
            )
        length = raw.numel()
        tokens = torch.full(
            (1, self.max_input_length),
            self.pad_token_id,
            device=self.device,
            dtype=self.token_dtype,
        )
        tokens[0, :length] = raw
        baseline = self._forward(tokens)
        if max(raw.max().item(), self.pad_token_id) >= baseline.shape[-1]:
            raise ValueError("PT2 input token ID outside logits vocabulary")
        checks = []
        for cut in sorted({1, max(1, length // 2), length}):
            if cut == self.max_input_length:
                continue
            for fill in ("ones", "alternating_zero_one"):
                variant = tokens.clone()
                tail = variant[0, cut:]
                if fill == "ones":
                    tail.fill_(1)
                else:
                    tail.copy_(torch.arange(tail.numel(), device=self.device) % 2)
                comparison = self._forward(variant)[:cut]
                difference = (baseline[:cut] - comparison).abs().max().item()
                equal = torch.equal(baseline[:cut], comparison)
                del comparison
                checks.append(
                    dict(cut=cut, fill=fill, bit_exact=equal, max_abs=difference)
                )
                if not equal:
                    self.last_causality_check = dict(passed=False, checks=checks)
                    raise ValueError(
                        f"PT2 masked-future causality check failed at cut={cut}, "
                        f"fill={fill}, max_abs={difference}; graph was not patched"
                    )
        self.last_causality_check = dict(
            passed=True,
            policy="bit_exact_prefix_logits",
            prefix_length=length,
            checks=checks,
        )
        return baseline[:length]

    def validate_causality(self, ids):
        """Check several prefix cuts and two future fills, raising on any change."""
        self._checked_logits(ids)
        return self.last_causality_check

    def __call__(self, ids):
        """Return [len(ids), vocab] logits after mandatory causality validation."""
        return self._checked_logits(ids)
