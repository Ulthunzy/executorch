# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from executorch.examples.qualcomm.custom_op.deltanet_support.chunk import (
    _chunk_fake,
    chunk,
    ChunkGDN,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def inputs(tokens=128, width=128):
    generator = torch.Generator().manual_seed(61)
    shape = (1, tokens, 2, width)
    return (
        torch.nn.functional.normalize(torch.randn(shape, generator=generator), dim=-1)
        / width**0.5,
        torch.nn.functional.normalize(torch.randn(shape, generator=generator), dim=-1),
        torch.randn(shape, generator=generator) * 0.1,
        0.95 + torch.rand(shape[:-1], generator=generator) * 0.05,
        0.1 + torch.rand(shape[:-1], generator=generator) * 0.5,
        torch.randn(1, 2, width, width, generator=generator) * 0.1,
    )


def sequential(query, key, value, decay, beta, state):
    query, key, value, decay, beta, state = (
        x.half().double() for x in (query, key, value, decay, beta, state)
    )
    output = []
    for index in range(query.shape[1]):
        k = key[:, index]
        state = state * decay[:, index, :, None, None]
        correction = beta[:, index, :, None] * (
            value[:, index] - (k.unsqueeze(-2) @ state).squeeze(-2)
        )
        state = state + k.unsqueeze(-1) * correction.unsqueeze(-2)
        output.append((query[:, index].unsqueeze(-2) @ state).squeeze(-2))
    return torch.stack(output, 1), state


def test_ar128_exports_one_chunk():
    program = torch.export.export(ChunkGDN(fused_chunk=True), inputs(), strict=True)
    custom = [
        n.target for n in program.graph.nodes if "qnn_custom_ops." in str(n.target)
    ]
    assert custom == [chunk]
    assert not list(program.module().named_buffers())


@pytest.mark.parametrize("tokens,width", [(64, 128), (256, 128), (128, 32)])
def test_other_shapes_retain_portable_path(tokens, width):
    args = inputs(tokens, width)
    expected = ChunkGDN()(*args)
    actual = ChunkGDN(fused_chunk=True)(*args)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    program = torch.export.export(ChunkGDN(fused_chunk=True), args, strict=True)
    assert all(n.target != chunk for n in program.graph.nodes)


def test_nonzero_state_feedback_against_sequential():
    args = list(inputs())
    state = args[-1]
    reference_state = state
    module = ChunkGDN(fused_chunk=True)
    for _ in range(4):
        actual, state = module(*args[:-1], state)
        expected, reference_state = sequential(*args[:-1], reference_state)
        for output, reference in ((actual, expected), (state, reference_state)):
            relative = torch.linalg.vector_norm(output.double() - reference)
            relative /= torch.linalg.vector_norm(reference)
            assert relative < 0.005
            assert torch.isfinite(output).all()


def test_padding_preserves_state():
    query, key, value, decay, beta, state = inputs()
    query.zero_()
    beta.zero_()
    decay.fill_(1)
    output, updated = ChunkGDN(fused_chunk=True)(query, key, value, decay, beta, state)
    assert torch.count_nonzero(output) == 0
    assert torch.equal(updated, state.half().float())


@pytest.mark.parametrize("index", range(7))
@pytest.mark.parametrize("invalid", ["shape", "dtype"])
def test_fake_rejects_unsupported_inputs(index, invalid):
    query, key, value, decay, beta, state = inputs()
    args = [
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        decay.transpose(1, 2).unsqueeze(-2),
        beta.transpose(1, 2).unsqueeze(-2),
        state,
        key.transpose(1, 2).transpose(-1, -2),
    ]
    args = [x.half() for x in args]
    args[index] = args[index][..., :64] if invalid == "shape" else args[index].float()
    with pytest.raises(RuntimeError):
        _chunk_fake(*args)
