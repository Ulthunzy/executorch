# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from executorch.examples.qualcomm.custom_op.deltanet_support.chunk import ChunkGDN
from executorch.examples.qualcomm.custom_op.deltanet_support.state_update import (
    chunk_state_update,
)


def test_preserves_half_feedback():
    state = torch.full((1, 1, 32, 64), 1 + 2**-10, dtype=torch.float16)
    decay = torch.full((1, 1, 1, 1), 1 + 2**-10, dtype=torch.float16)
    key = torch.zeros(1, 1, 32, 32, dtype=torch.float16)
    key[..., 0] = 1
    correction = torch.zeros(1, 1, 32, 64, dtype=torch.float16)
    correction[:, :, 0] = -(1 + 2**-9)
    weight = torch.ones(1, 1, 32, 1, dtype=torch.float16)
    result = chunk_state_update(state, key, correction, weight, decay)
    assert torch.count_nonzero(result) == 0
    assert torch.count_nonzero(decay.float() * state.float() - (1 + 2**-9)) > 0


@pytest.mark.parametrize("nonzero_state", [False, True])
def test_chunk_feedback_and_export(nonzero_state):
    generator = torch.Generator().manual_seed(47)
    shape = (1, 128, 2, 32)
    inputs = (
        torch.nn.functional.normalize(torch.randn(shape, generator=generator), dim=-1)
        / 32**0.5,
        torch.nn.functional.normalize(torch.randn(shape, generator=generator), dim=-1),
        torch.randn(1, 128, 2, 64, generator=generator) * 0.1,
        torch.rand(1, 128, 2, generator=generator) * 0.08 + 0.9,
        torch.rand(1, 128, 2, generator=generator) * 0.5 + 0.1,
        torch.randn(1, 2, 32, 64, generator=generator) * nonzero_state,
    )
    baseline = ChunkGDN()
    fused = ChunkGDN(fused_state_update=True)
    with torch.no_grad():
        expected = baseline(*inputs)
        actual = fused(*inputs)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    exported = torch.export.export(fused, inputs, strict=True)
    assert sum(n.target == chunk_state_update for n in exported.graph.nodes) == 2


def test_export_rejects_unsupported_shape():
    class Unsupported(torch.nn.Module):
        def forward(self, state, key, correction, weight, decay):
            return chunk_state_update(state, key, correction, weight, decay)

    inputs = (
        torch.zeros(1, 1, 31, 64, dtype=torch.float16),
        torch.zeros(1, 1, 31, 64, dtype=torch.float16),
        torch.zeros(1, 1, 64, 64, dtype=torch.float16),
        torch.ones(1, 1, 64, 1, dtype=torch.float16),
        torch.ones(1, 1, 1, 1, dtype=torch.float16),
    )
    with pytest.raises((RuntimeError, torch._dynamo.exc.TorchRuntimeError)):
        torch.export.export(Unsupported(), inputs, strict=True)
