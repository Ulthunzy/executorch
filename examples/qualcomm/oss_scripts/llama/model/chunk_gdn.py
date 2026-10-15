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
            "strict",
            torch.tril(torch.ones(chunk_size, chunk_size, dtype=dtype), diagonal=-1),
        )
        self.inverse_leaf = 8
        self.register_buffer("leaf_eye", torch.eye(self.inverse_leaf, dtype=dtype))
        block = torch.arange(chunk_size // self.inverse_leaf)[:, None, None]
        row = block * self.inverse_leaf + torch.arange(self.inverse_leaf)[None, :, None]
        col = block * self.inverse_leaf + torch.arange(self.inverse_leaf)[None, None, :]
        self.register_buffer("leaf_indices", (row * chunk_size + col).flatten().int())
        indices = []
        for level in range(chunk_size.bit_length() - 1):
            width = 1 << level
            block = torch.arange(chunk_size // (2 * width))[:, None, None]
            row = block * (2 * width) + width + torch.arange(width)[None, :, None]
            col = block * (2 * width) + torch.arange(width)[None, None, :]
            indices.append((row * chunk_size + col).flatten())
        self.inverse_sizes = tuple(index.numel() for index in indices)
        self.register_buffer("inverse_indices", torch.cat(indices).to(torch.int32))

    def triangular_inverse(self, lower):
        batch, heads, count, _ = lower.shape
        flat = lower.flatten(-2)
        leaf = self.inverse_leaf
        power = flat.index_select(-1, self.leaf_indices).reshape(
            batch, heads * count // leaf, leaf, leaf
        )
        inverse = self.leaf_eye + power
        for _ in range(leaf.bit_length() - 2):
            power = power @ power
            inverse = inverse @ (self.leaf_eye + power)
        indices = self.inverse_indices.split(self.inverse_sizes)
        for level, index in enumerate(indices):
            width = 1 << level
            if width < leaf:
                continue
            blocks = heads * count // (2 * width)
            even = torch.arange(
                0, blocks * 2, 2, device=lower.device, dtype=torch.int32
            )
            odd = torch.arange(1, blocks * 2, 2, device=lower.device, dtype=torch.int32)
            a = inverse.index_select(1, even)
            # Materialize the odd gather: direct QNN Concat reads the wrong
            # stride in this graph. The smallest FP16 subnormal prevents that
            # view optimization; its numerical effect is checked on device.
            b = inverse.index_select(1, odd) + 2**-24
            off_diagonal = flat.index_select(-1, index).reshape(
                batch, blocks, width, width
            )
            # Inverse of [[A, 0], [-T, B]] has lower block B^-1 T A^-1.
            cross = b @ off_diagonal @ a
            inverse = torch.cat(
                (torch.cat((a, torch.zeros_like(a)), -1), torch.cat((cross, b), -1)),
                -2,
            )
        return inverse.reshape(batch, heads, count, count)

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
            weights = ((g.unsqueeze(-1) - g.unsqueeze(-2)) * self.lower).clamp(
                min=-16, max=0
            ).exp() * self.lower
            prefix = g.clamp(min=-16, max=0).exp().unsqueeze(-1)
            lower = -((k * b) @ k.transpose(-1, -2)) * weights * self.strict
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
