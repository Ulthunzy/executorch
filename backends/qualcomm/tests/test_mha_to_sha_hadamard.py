# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.qualcomm._passes import (
    ConvertBmmToMatmul,
    ConvertMhaToSha,
    RemoveRedundancy,
)
from executorch.backends.qualcomm.utils.utils import convert_linear_to_conv2d
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops


class HadamardAttention(torch.nn.Module):
    def __init__(self, kv_heads):
        super().__init__()
        self.kv_heads = kv_heads
        self.q = torch.nn.Linear(128, 128, bias=False)
        self.k = torch.nn.Linear(128, kv_heads * 32, bias=False)
        self.v = torch.nn.Linear(128, kv_heads * 32, bias=False)

    def forward(self, x):
        batch, seq, _ = x.shape
        q = self.q(x).view(batch, seq, 4, 32).transpose(1, 2)
        k = self.k(x).view(batch, seq, self.kv_heads, 32).transpose(1, 2)
        v = self.v(x).view(batch, seq, self.kv_heads, 32).transpose(1, 2)
        q = torch.ops.qnn_custom.hadamard_transform.default(q, 1.0)
        k = torch.ops.qnn_custom.hadamard_transform.default(k, 1.0)
        repeat = 4 // self.kv_heads
        keys = k[:, :, None].expand(batch, self.kv_heads, repeat, seq, 32)
        values = v[:, :, None].expand(batch, self.kv_heads, repeat, seq, 32)
        keys = keys.reshape(batch, 4, seq, 32)
        values = values.reshape(batch, 4, seq, 32)
        weights = torch.softmax((q @ keys.transpose(-2, -1)) / (32**0.5), -1)
        output = (weights @ values).transpose(1, 2).reshape(batch, seq, 128)
        # Direct rotation outputs must remain valid, without a slice/concat cycle.
        return output, q, k


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("ar", [1, 4, 8, 128])
def test_mha_to_sha_keeps_hadamard_batched(kv_heads, ar):
    torch.manual_seed(42)
    module = convert_linear_to_conv2d(HadamardAttention(kv_heads)).eval()
    inputs = (torch.randn(1, ar, 128),)
    with torch.no_grad():
        reference = module(*inputs)
    program = to_edge(
        torch.export.export(module, inputs, strict=True)
    ).exported_program()
    graph_module = RemoveRedundancy(quantization_capture=False)(
        program.graph_module
    ).graph_module
    graph_module = ConvertBmmToMatmul()(graph_module).graph_module
    result = ConvertMhaToSha(program)(graph_module)
    assert result.modified
    rotations = [
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.edge.qnn_custom.hadamard_transform.default
    ]
    assert len(rotations) == 2
    assert sorted(node.meta["val"].shape[1] for node in rotations) == sorted(
        [4, kv_heads]
    )
    softmaxes = [
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.edge.aten._softmax.default
    ]
    assert len(softmaxes) == 4
    assert all(node.meta["val"].shape[1] == 1 for node in softmaxes)
    with torch.no_grad():
        actual = program.module()(*inputs)
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
