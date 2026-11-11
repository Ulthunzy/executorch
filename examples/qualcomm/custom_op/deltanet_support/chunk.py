# Copyright (c) Ulthunzy
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GDN chunk recurrence using a block triangular inverse."""

import torch

from executorch.examples.qualcomm.custom_op.htp_op_package.package import op_package

from .state_update import chunk_state_update


_library = torch.library.Library("qnn_custom_ops", "FRAGMENT")
_library.define(
    "chunk(Tensor query, Tensor key, Tensor value, Tensor decay, Tensor beta, "
    "Tensor state, Tensor key_transposed) -> (Tensor, Tensor)"
)


@torch.library.impl(_library, "chunk", "CompositeExplicitAutograd")
def _chunk_reference(query, key, value, decay, beta, state, key_transposed):
    decay = decay.clamp_min(2**-14)
    beta = beta.transpose(-1, -2)
    gram = (key * beta) @ key_transposed
    qk = query @ key_transposed
    weights = torch.zeros_like(decay[..., 0, :])
    cumulative = torch.ones_like(decay[..., :1, :1], dtype=torch.float32)
    columns = torch.arange(128, device=query.device)
    lower, attention, prefix = [], [], []
    for row in range(128):
        d = decay[..., 0, row : row + 1]
        cumulative = (cumulative * d.unsqueeze(-1).float()).clamp_min(
            1.1253517471925912e-7
        )
        prefix.append(cumulative.half())
        weights = (weights * d).clamp_min(2**-23)
        weights = torch.where(columns == row, 1, torch.where(columns < row, weights, 0))
        lower.append(torch.where(columns < row, -weights * gram[..., row, :], 0))
        attention.append(weights * qk[..., row, :])
    lower = torch.stack(lower, -2)
    attention = torch.stack(attention, -2)
    prefix = torch.cat(prefix, -2)
    blocks = []
    for start in (0, 64):
        leaves = []
        for offset in (start, start + 32):
            inverse = lower[..., offset : offset + 32, offset : offset + 32].clone()
            for row in range(32):
                coefficients = inverse[..., row, :].clone()
                for column in range(1, row):
                    inverse[..., row, :] += (
                        coefficients[..., column : column + 1] * inverse[..., column, :]
                    )
            inverse += torch.eye(32, dtype=query.dtype, device=query.device)
            leaves.append(inverse)
        left, right = leaves
        cross = right @ lower[..., start + 32 : start + 64, start : start + 32] @ left
        blocks.append(
            torch.cat(
                (
                    torch.cat((left, torch.zeros_like(left)), -1),
                    torch.cat((cross, right), -1),
                ),
                -2,
            )
        )
    source = beta * (value - prefix * (key @ state))
    first = blocks[0] @ source[..., :64, :]
    second = blocks[1] @ (source[..., 64:, :] + lower[..., 64:, :64] @ first)
    correction = torch.cat((first, second), -2)
    return (
        prefix * (query @ state) + attention @ correction,
        prefix[..., -1:, :] * state
        + key_transposed @ (weights.unsqueeze(-1) * correction),
    )


@torch.library.register_fake("qnn_custom_ops::chunk")
def _chunk_fake(query, key, value, decay, beta, state, key_transposed):
    torch._check(query.ndim == 4)
    batch, heads = query.shape[:2]
    torch._check(batch > 0)
    torch._check(heads > 0)
    torch._check(batch * heads <= 256)
    for tensor in (query, key, value, state, key_transposed):
        torch._check(tensor.shape == (batch, heads, 128, 128))
        torch._check(tensor.dtype == torch.float16)
    for tensor in (decay, beta):
        torch._check(tensor.shape == (batch, heads, 1, 128))
        torch._check(tensor.dtype == torch.float16)
    return torch.empty_like(query), torch.empty_like(state)


chunk = torch.ops.qnn_custom_ops.chunk.default


def chunk_package(host, device):
    """Register the fused AR128 operator with the host and device packages."""
    return op_package(host, device, [(chunk, "Chunk")])


class ChunkGDN(torch.nn.Module):
    def __init__(self, fused_state_update=False, fused_chunk=False):
        super().__init__()
        self.fused_state_update = fused_state_update
        self.fused_chunk = fused_chunk
        chunk_size = 64
        dtype = torch.float16
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.register_buffer(
            "lower", torch.tril(torch.ones(chunk_size, chunk_size, dtype=dtype))
        )
        self.register_buffer(
            "negative_strict",
            -torch.tril(torch.ones(chunk_size, chunk_size, dtype=dtype), diagonal=-1),
        )
        self.inverse_leaf = chunk_size // 2
        self.register_buffer("leaf_eye", torch.eye(self.inverse_leaf, dtype=dtype))
        row = torch.arange(self.inverse_leaf)[:, None]
        col = torch.arange(self.inverse_leaf)[None, :]
        self.register_buffer("leaf_diagonal", (row // 8 == col // 8).to(dtype))
        self.register_buffer(
            "merge_8", ((row // 16 == col // 16) & (row // 8 == col // 8 + 1)).to(dtype)
        )
        self.register_buffer("merge_16", ((row >= 16) & (col < 16)).to(dtype))

    def triangular_inverse(self, lower):
        batch, heads, _, _ = lower.shape
        width = self.inverse_leaf
        blocks = torch.stack(
            (lower[:, :, :width, :width], lower[:, :, width:, width:]), dim=2
        ).reshape(batch, heads * 2, width, width)
        # Embed 8x8 leaves in 32x32 operands. High-order dense triangular powers
        # overflow FP16 for aligned keys, even when the inverse is bounded.
        power = blocks * self.leaf_diagonal
        inverse = self.leaf_eye + power
        for _ in range(2):
            power = power @ power
            inverse = inverse @ (self.leaf_eye + power)
        for mask in (self.merge_8, self.merge_16):
            inverse = inverse + inverse @ (blocks * mask) @ inverse
        a = inverse[:, 0::2]
        # Materialize the odd slice: direct QNN Concat otherwise reads the wrong
        # stride. Retain the device-validated FP16 subnormal workaround.
        b = inverse[:, 1::2] + 2**-24
        # Inverse of [[A, 0], [-T, B]] has lower block B^-1 T A^-1.
        cross = b @ lower[:, :, width:, :width] @ a
        return torch.cat(
            (torch.cat((a, torch.zeros_like(a)), -1), torch.cat((cross, b), -1)), -2
        )

    def forward(self, query, key, value, decay, beta, state):
        query = query.to(self.dtype).transpose(1, 2)
        key = key.to(self.dtype).transpose(1, 2)
        value = value.to(self.dtype).transpose(1, 2)
        decay = decay.to(self.dtype).transpose(1, 2)
        beta = beta.to(self.dtype).transpose(1, 2)
        state = state.to(self.dtype)
        if (
            self.fused_chunk
            and query.shape[2:] == (128, 128)
            and value.shape[-1] == 128
        ):
            output, updated = chunk(
                query,
                key,
                value,
                decay.unsqueeze(-2),
                beta.unsqueeze(-2),
                state,
                key.transpose(-1, -2),
            )
            return output.transpose(1, 2).float(), updated.float()
        outputs = []
        c = self.chunk_size
        for start in range(0, query.shape[2], c):
            q = query[:, :, start : start + c]
            k = key[:, :, start : start + c]
            v = value[:, :, start : start + c]
            b = beta[:, :, start : start + c].unsqueeze(-1)
            d = decay[:, :, start : start + c].clamp_min(2**-14)
            # A matrix prefix sum avoids an unsupported scan in some backends.
            g = (d.log().unsqueeze(-2) @ self.lower.transpose(0, 1)).squeeze(-2)
            weights = (g.unsqueeze(-1) - g.unsqueeze(-2)).clamp(
                min=-16, max=0
            ).exp() * self.lower
            prefix = g.clamp(min=-16, max=0).exp().unsqueeze(-1)
            lower = ((k * b) @ k.transpose(-1, -2)) * (weights * self.negative_strict)
            inverse = self.triangular_inverse(lower)
            correction = inverse @ (b * (v - prefix * (k @ state)))
            y = (
                prefix * (q @ state)
                + ((q @ k.transpose(-1, -2)) * weights) @ correction
            )
            if self.fused_state_update:
                state = chunk_state_update(
                    state,
                    k.transpose(-1, -2),
                    correction,
                    weights[:, :, -1:].transpose(-1, -2),
                    prefix[:, :, -1:],
                )
            else:
                state = prefix[:, :, -1:] * state + k.transpose(-1, -2) @ (
                    weights[:, :, -1:].transpose(-1, -2) * correction
                )
            outputs.append(y)
        return torch.cat(outputs, dim=2).transpose(1, 2).float(), state.float()
