# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP16 ChunkGDN state update with fused HMX matmul and decay epilogue."""

import torch

from executorch.examples.qualcomm.custom_op.htp_op_package.package import op_package


_library = torch.library.Library("qnn_custom_ops", "FRAGMENT")
_library.define(
    "chunk_state_update(Tensor state, Tensor key, Tensor correction, "
    "Tensor weight, Tensor decay) -> Tensor"
)


@torch.library.impl(_library, "chunk_state_update", "CompositeExplicitAutograd")
def _reference(state, key, correction, weight, decay):
    return decay * state + key @ (weight * correction)


@torch.library.register_fake("qnn_custom_ops::chunk_state_update")
def _fake(state, key, correction, weight, decay):
    torch._check(state.ndim == 4)
    torch._check(key.ndim == 4)
    batch, heads, rows, columns = state.shape
    inner = key.shape[-1]
    for size in (batch, heads, rows, columns, inner):
        torch._check(size > 0)
    torch._check(state.numel() <= 2**31 - 1)
    torch._check(key.shape == (batch, heads, rows, inner))
    torch._check(correction.shape == (batch, heads, inner, columns))
    torch._check(weight.shape == (batch, heads, inner, 1))
    torch._check(decay.shape == (batch, heads, 1, 1))
    for value in (state, key, correction, weight, decay):
        torch._check(value.dtype == torch.float16)
    torch._check(rows % 32 == 0)
    torch._check(columns % 64 == 0)
    torch._check(inner % 32 == 0)
    torch._check(rows <= 256)
    torch._check(columns <= 256)
    torch._check(inner <= 128)
    return torch.empty_like(state)


chunk_state_update = torch.ops.qnn_custom_ops.chunk_state_update.default


def state_update_package(host, device):
    return op_package(host, device, [(chunk_state_update, "ChunkStateUpdate")])
