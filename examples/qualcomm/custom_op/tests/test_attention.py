# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.examples.qualcomm.custom_op.attention import (
    attention_reference,
    attention_quant_annotations,
)
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.mark.parametrize(
    "batch,heads,kv_heads,dimension,queries,valid,first,history",
    [
        (1, 16, 2, 256, 128, 7, 0, 17),
        (2, 6, 2, 63, 37, 17, 3, 128),
        (1, 3, 3, 65, 128, 127, 5, 259),
        (1, 8, 1, 128, 1, 1, 0, 1024),
    ],
)
def test_reference(batch, heads, kv_heads, dimension, queries, valid, first, history):
    generator = torch.Generator().manual_seed(17)

    def rand(*shape):
        return torch.randn(*shape, generator=generator)

    capacity = first + history + 128
    q = rand(batch, heads, queries, dimension)
    k, v = rand(batch, kv_heads, dimension, capacity), rand(
        batch, kv_heads, capacity, dimension
    )
    nk, nv = rand(batch, kv_heads, dimension, queries), rand(
        batch, kv_heads, queries, dimension
    )
    lengths = torch.tensor([valid, first, history], dtype=torch.int32)
    result = attention_reference(q, k, nk, v, nv, lengths)
    for row in range(valid):
        keys = torch.cat(
            (k[..., first : first + history], nk[..., : row + 1]), -1
        ).transpose(-1, -2)
        values = torch.cat(
            (v[..., first : first + history, :], nv[..., : row + 1, :]), -2
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            q[..., row : row + 1, :], keys, values, enable_gqa=True
        )
        torch.testing.assert_close(
            result[..., row : row + 1, :], expected, atol=2e-6, rtol=1e-5
        )
    assert torch.count_nonzero(result[..., valid:, :]) == 0
    k[..., :first] = float("nan")
    k[..., first + history :] = float("nan")
    v[..., :first, :] = float("nan")
    v[..., first + history :, :] = float("nan")
    nk[..., valid:] = float("nan")
    nv[..., valid:, :] = float("nan")
    torch.testing.assert_close(attention_reference(q, k, nk, v, nv, lengths), result)


def test_export_descriptor_boundary():
    class Attention(torch.nn.Module):
        def forward(self, q, kd, nk, vd, nv, lengths, grids):
            return torch.ops.qnn_custom_ops.attention_valid(
                q, kd, nk, vd, nv, lengths, grids
            )

    args = (
        torch.empty(1, 16, 128, 256),
        torch.zeros(1, 1, 1, 3, dtype=torch.int32),
        torch.empty(1, 2, 256, 128),
        torch.zeros(1, 1, 1, 3, dtype=torch.int32),
        torch.empty(1, 2, 128, 256),
        torch.zeros(1, 1, 1, 3, dtype=torch.int32),
        torch.ones(1, 1, 1, 13),
    )
    program = torch.export.export(Attention(), args, strict=True)
    assert any(
        node.target == torch.ops.qnn_custom_ops.attention_valid.default
        for node in program.graph.nodes
    )
    with pytest.raises(RuntimeError, match="DSP FD descriptors"):
        Attention()(*args)
    with FakeTensorMode():
        bad = [torch.empty(t.shape, dtype=t.dtype) for t in args]
        bad[2] = torch.empty(1, 3, 256, 128)
        with pytest.raises(RuntimeError):
            Attention()(*bad)


def test_quantization_configuration():
    from executorch.backends.qualcomm.quantizer.qconfig import (
        get_16a8w_qnn_ptq_config,
        get_8a8w_qnn_ptq_config,
    )

    annotations = attention_quant_annotations(get_16a8w_qnn_ptq_config())
    assert len(annotations) == 1 and callable(annotations[0])
    with pytest.raises(ValueError, match="U16"):
        attention_quant_annotations(get_8a8w_qnn_ptq_config())


def test_shared_package_registration():
    from executorch.examples.qualcomm.custom_op.deltanet_support.package import (
        deltanet_package,
    )

    from executorch.examples.qualcomm.custom_op.attention_support.package import (
        attention_package,
    )

    package = deltanet_package("/tmp/host.so", "libet_qnnop.so", valid_length=True)
    attention = attention_package("/tmp/host.so", "libet_qnnop.so")
    assert len(attention.op_package_infos) == 2
    package.op_package_infos.extend(attention.op_package_infos)
    names = [info.qnn_op_type_name for info in package.op_package_infos]
    assert names.count("AttentionValid") == 2
    assert names.count("DeltaNetCoreValid") == 2
    assert names.count("DeltaNetCore") == 2
