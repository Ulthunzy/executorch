# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GDN chunk recurrence using a block triangular inverse."""

import torch


class ChunkGDN(torch.nn.Module):
    def __init__(self):
        super().__init__()
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
            state = prefix[:, :, -1:] * state + k.transpose(-1, -2) @ (
                weights[:, :, -1:].transpose(-1, -2) * correction
            )
            outputs.append(y)
        return torch.cat(outputs, dim=2).transpose(1, 2).float(), state.float()
