# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors

import math
from typing import List, Optional, Tuple

import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import (
    hf_precompute_freqs_cis,
    precompute_freqs_cis,
)
from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import (
    AttentionMask,
    CausalAttentionMask,
    SlidingWindowAttentionMask,
)
from executorch.examples.qualcomm.oss_scripts.llama.model import (
    FeedForward_REGISTRY,
    NORM_REGISTRY,
    Qwen35RMSNorm,
    Qwen35RMSNormGated,
    ROTARY_EMB_REGISTRY,
)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def repeat_heads(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, seq_len, num_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(
        batch, seq_len, num_heads, n_rep, head_dim
    )
    return hidden_states.reshape(batch, seq_len, num_heads * n_rep, head_dim)


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _qnn_softplus(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) + torch.log1p(torch.exp(-torch.abs(x)))


class AttentionSinkRope(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        sink_size: int,
        eviction_batch_size: int,
        ar_len: int,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.sink_size = sink_size
        self.eviction_batch_size = eviction_batch_size
        self.n_layers = config.n_layers
        self.original_position = eviction_batch_size + sink_size
        self.new_position = sink_size
        self.num_to_keep = (
            config.max_context_len - sink_size - eviction_batch_size - ar_len
        )
        self.evict_k_cache_shape = [
            config.max_batch_size,
            config.n_kv_heads,
            config.head_dim,
            eviction_batch_size,
        ]
        self.evict_v_cache_shape = [
            config.max_batch_size,
            config.n_kv_heads,
            eviction_batch_size,
            config.head_dim,
        ]
        self.kv_cache_shape = {
            # single head, k input
            "k": (
                config.max_batch_size,
                config.n_kv_heads,
                config.head_dim,
                config.max_context_len - ar_len,
            ),
            # single head, v input
            "v": (
                config.max_batch_size,
                config.n_kv_heads,
                config.max_context_len - ar_len,
                config.head_dim,
            ),
        }

        if getattr(config, "enable_r3", False):
            self.register_buffer(
                "r3_weight",
                torch.tensor(
                    scipy.linalg.hadamard(config.head_dim, dtype=float)
                    / math.sqrt(config.head_dim),
                    dtype=torch.float32,
                    device="cpu",
                ),
                persistent=False,
            )

        if config.partial_rotary_factor < 1:
            self.apply_rope_emb = ROTARY_EMB_REGISTRY["partial"]
        else:
            self.apply_rope_emb = ROTARY_EMB_REGISTRY["default"]

        if config.use_hf_rope:
            freqs_cos, freqs_sin = hf_precompute_freqs_cis(
                config.head_dim,
                config.max_context_len,
                config.rope_freq_base,
                config.partial_rotary_factor,
            )
            freqs_cos = freqs_cos[:, : freqs_cos.shape[-1] // 2]
            freqs_sin = freqs_sin[:, : freqs_sin.shape[-1] // 2]
        else:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                config.head_dim,
                config.max_context_len,
                config.rope_freq_base,
                config.use_scaled_rope,
                config.rope_scale_factor,
            )
        original_freqs_cos = freqs_cos.narrow(
            0, self.original_position, self.num_to_keep
        )
        original_freqs_sin = freqs_sin.narrow(
            0, self.original_position, self.num_to_keep
        )
        new_freqs_cos = freqs_cos.narrow(0, self.new_position, self.num_to_keep)
        new_freqs_sin = freqs_sin.narrow(0, self.new_position, self.num_to_keep)
        rerotation_cos = (
            new_freqs_cos * original_freqs_cos + new_freqs_sin * original_freqs_sin
        )
        rerotation_sin = (
            new_freqs_sin * original_freqs_cos - new_freqs_cos * original_freqs_sin
        )
        self.register_buffer("rerotation_cos", rerotation_cos, persistent=False)
        self.register_buffer("rerotation_sin", rerotation_sin, persistent=False)

        self.sliding_window = kwargs.get("sliding_window", False)
        if self.sliding_window:
            # Get attention type for each layer
            self.layer_types = kwargs["layer_types"]
            # Get local freq base for sliding attention
            rope_freq_base = kwargs["rope_local_base_freq"]
            local_freqs_cos, local_freqs_sin = hf_precompute_freqs_cis(
                config.head_dim,
                config.max_context_len,
                rope_freq_base,
                config.partial_rotary_factor,
            )
            local_freqs_cos = local_freqs_cos[:, : local_freqs_cos.shape[-1] // 2]
            local_freqs_sin = local_freqs_sin[:, : local_freqs_sin.shape[-1] // 2]
            local_original_freqs_cos = local_freqs_cos.narrow(
                0, self.original_position, self.num_to_keep
            )
            local_original_freqs_sin = local_freqs_sin.narrow(
                0, self.original_position, self.num_to_keep
            )
            local_new_freqs_cos = local_freqs_cos.narrow(
                0, self.new_position, self.num_to_keep
            )
            local_new_freqs_sin = local_freqs_sin.narrow(
                0, self.new_position, self.num_to_keep
            )
            local_rerotation_cos = (
                local_new_freqs_cos * local_original_freqs_cos
                + local_new_freqs_sin * local_original_freqs_sin
            )
            local_rerotation_sin = (
                local_new_freqs_sin * local_original_freqs_cos
                - local_new_freqs_cos * local_original_freqs_sin
            )
            self.register_buffer(
                "local_rerotation_cos", local_rerotation_cos, persistent=False
            )
            self.register_buffer(
                "local_rerotation_sin", local_rerotation_sin, persistent=False
            )

    def forward(self, k_caches: List[torch.Tensor], v_caches: List[torch.Tensor]):
        """
        Rerotate k_cache from original_position to new_position, and return the kv cache after eviction. This is done by rerotating
        k_cache with (new_position * theta - original_position * theta) with the following matrix:
        (cos(delta), -sin(delta)
         sin(delta), cos(delta))
         where delta = new_position * theta - original_position * theta

         Based on https://github.com/huggingface/transformers/pull/26681
        """

        output_k_caches, output_v_caches = [], []
        for ind, (k_cache, v_cache) in enumerate(zip(k_caches, v_caches)):
            # k_cache: (batch_size, n_kv_heads, head_dim, seq_len)
            # v_cache: (batch_size, n_kv_heads, seq_len, head_dim)
            k_dim_to_slice = 3
            v_dim_to_slice = 2

            k_to_keep = k_cache.narrow(
                k_dim_to_slice,
                self.original_position,
                self.num_to_keep,
            )
            k_to_keep = k_to_keep.transpose(2, 3)
            if getattr(self.config, "enable_r3", False):
                # We need to revert the key from spin quant before applying RoPE
                k_to_keep = torch.matmul(k_to_keep, self.r3_weight.T)

            if self.sliding_window and self.layer_types[ind] == "sliding_attention":
                k_to_keep = self.apply_rope_emb(
                    k_to_keep, self.local_rerotation_cos, self.local_rerotation_sin
                )
            else:
                k_to_keep = self.apply_rope_emb(
                    k_to_keep, self.rerotation_cos, self.rerotation_sin
                )
            if getattr(self.config, "enable_r3", False):
                k_to_keep = torch.matmul(k_to_keep, self.r3_weight)
            k_to_keep = k_to_keep.transpose(2, 3)
            new_k_cache = torch.cat(
                [
                    k_cache.narrow(k_dim_to_slice, 0, self.sink_size),
                    k_to_keep,
                    torch.zeros(self.evict_k_cache_shape),
                ],
                dim=k_dim_to_slice,
            )

            new_v_cache = torch.cat(
                [
                    v_cache.narrow(v_dim_to_slice, 0, self.sink_size),
                    v_cache.narrow(
                        v_dim_to_slice,
                        self.original_position,
                        self.num_to_keep,
                    ),
                    torch.zeros(self.evict_v_cache_shape),
                ],
                dim=v_dim_to_slice,
            )

            output_k_caches.append(new_k_cache)
            output_v_caches.append(new_v_cache)

        return output_k_caches, output_v_caches

    def get_example_inputs(self):
        k_cache, v_cache = [], []

        for _ in range(self.n_layers):
            k_cache.append(torch.zeros(self.kv_cache_shape["k"]))
            v_cache.append(torch.zeros(self.kv_cache_shape["v"]))
        return k_cache, v_cache

    def get_metadata(self):
        return {
            "get_eviction_batch_size": self.eviction_batch_size,
            "get_max_context_len": self.config.max_context_len,
            "get_sink_size": self.sink_size,
        }


class LlamaAttention(nn.Module):
    def __init__(self, layer_idx: int, config: ModelArgs, output_new_cache_only=False):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.n_kv_heads = config.n_kv_heads
        self.num_key_value_groups = config.n_heads // self.n_kv_heads
        self.max_seq_len = config.max_seq_len
        self.max_context_len = config.max_context_len
        self.use_kv_cache = config.use_kv_cache
        self.output_new_cache_only = output_new_cache_only
        self.enable_masked_softmax = getattr(config, "enable_masked_softmax", False)
        self.use_qk_norm = config.use_qk_norm
        self.qk_norm_before_rope = config.qk_norm_before_rope
        self.use_q_gate = config.use_q_gate
        q_out_dim = self.n_heads * self.head_dim * (2 if self.use_q_gate else 1)
        # If None, assume each layer uses rope
        self.use_rope = (
            config.no_rope_layer_interval is None
            or (layer_idx + 1) % config.no_rope_layer_interval
        )

        if self.use_qk_norm:
            q_norm_dim = self.head_dim
            k_norm_dim = self.head_dim
            norm_cls = (
                Qwen35RMSNorm
                if config.rms_norm_add_unit_offset
                else torch.nn.RMSNorm
            )
            self.q_norm_fn = norm_cls(q_norm_dim, eps=config.norm_eps)
            self.k_norm_fn = norm_cls(k_norm_dim, eps=config.norm_eps)

        if config.partial_rotary_factor < 1:
            self.apply_rope_emb = ROTARY_EMB_REGISTRY["partial"]
        else:
            self.apply_rope_emb = ROTARY_EMB_REGISTRY["default"]

        self.wq = nn.Linear(
            self.dim,
            q_out_dim,
            bias=getattr(config, "attention_qkv_bias", False),
        )
        self.wk = nn.Linear(
            self.dim,
            self.n_kv_heads * self.head_dim,
            bias=getattr(config, "attention_qkv_bias", False),
        )
        self.wv = nn.Linear(
            self.dim,
            self.n_kv_heads * self.head_dim,
            bias=getattr(config, "attention_qkv_bias", False),
        )
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.attn_softmax = torch.nn.Softmax(dim=-1)

        self.scale = (
            float(self.head_dim) ** 0.5
            if config.attention_multiplier is None
            else 1.0 / config.attention_multiplier
        )

        # gemma 2 uses soft-capping on attention and logits
        self.attn_logit_softcapping = config.attn_logit_softcapping

        if getattr(config, "enable_r3", False):
            self.register_buffer(
                "r3_weight",
                torch.tensor(
                    scipy.linalg.hadamard(self.head_dim, dtype=float)
                    / math.sqrt(self.head_dim),
                    dtype=torch.float32,
                    device="cpu",
                ),
                persistent=False,
            )

    def prepare_attention_conv(self):
        self.wq_conv = nn.Conv2d(
            self.dim,
            self.n_heads * self.head_dim * (2 if self.use_q_gate else 1),
            1,
            bias=getattr(self.config, "attention_qkv_bias", False),
        )
        self.wk_conv = nn.Conv2d(
            self.dim,
            self.n_kv_heads * self.head_dim,
            1,
            bias=getattr(self.config, "attention_qkv_bias", False),
        )
        self.wv_conv = nn.Conv2d(
            self.dim,
            self.n_kv_heads * self.head_dim,
            1,
            bias=getattr(self.config, "attention_qkv_bias", False),
        )
        self.wo_conv = nn.Conv2d(self.n_heads * self.head_dim, self.dim, 1, bias=False)

        self.forward_no_conv = self.forward
        self.forward = self.forward_attention_conv

        self.wq_conv.weight.data.copy_(self.wq.weight[:, :, None, None])
        if self.wq_conv.bias is not None:
            self.wq_conv.bias.data.copy_(self.wq.bias)
        self.wk_conv.weight.data.copy_(self.wk.weight[:, :, None, None])
        if self.wk_conv.bias is not None:
            self.wk_conv.bias.data.copy_(self.wk.bias)
        self.wv_conv.weight.data.copy_(self.wv.weight[:, :, None, None])
        if self.wv_conv.bias is not None:
            self.wv_conv.bias.data.copy_(self.wv.bias)
        self.wo_conv.weight.data.copy_(self.wo.weight[:, :, None, None])

        del self.wq
        del self.wk
        del self.wv
        del self.wo

    def forward_attention_conv(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        hidden_states = torch.reshape(
            hidden_states, (bsz, seq_len, 1, self.dim)
        ).transpose(1, 3)

        q = self.wq_conv(hidden_states)
        k = self.wk_conv(hidden_states)
        v = self.wv_conv(hidden_states)
        q = q.permute(0, 3, 1, 2).squeeze(-1)
        k = k.permute(0, 3, 1, 2).squeeze(-1)
        v = v.permute(0, 3, 1, 2).squeeze(-1)
        if self.use_q_gate:
            q, gate = torch.chunk(
                q.view(bsz, seq_len, self.n_heads, self.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(bsz, seq_len, -1)
        else:
            q = q.view(bsz, seq_len, self.n_heads, self.head_dim)
            gate = None
        q = q.transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm and self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)

        if self.use_rope:
            q = self.apply_rope_emb(q, freqs_cos, freqs_sin)
            k = self.apply_rope_emb(k, freqs_cos, freqs_sin)

        if self.use_qk_norm and not self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)
        if getattr(self.config, "enable_r3", False):
            q = torch.matmul(q, self.r3_weight)
            k = torch.matmul(k, self.r3_weight)
        k = k.transpose(2, 3)

        kh, vh = None, None
        # kv cache mode
        if self.use_kv_cache:
            kh = torch.cat([k_caches, k], dim=-1)
            vh = torch.cat([v_caches, v], dim=2)
        # batch_prefill mode
        else:
            kh = k
            vh = v

        kh = repeat_kv(kh, self.num_key_value_groups)
        vh = repeat_kv(vh, self.num_key_value_groups)

        attn = q @ kh
        attn = attn / self.scale
        if self.enable_masked_softmax:
            attn_min = torch.amin(attn, dim=-1, keepdim=True)
            minus_value = -20
            attn = torch.where(atten_mask == 0, attn, attn_min + minus_value)
        else:
            attn = attn + atten_mask
        attn = self.attn_softmax(attn)
        y = attn @ vh
        y = y.transpose(1, 2).reshape(bsz, seq_len, -1)
        if gate is not None:
            y = y * torch.sigmoid(gate)
        y = y.reshape(bsz, seq_len, 1, -1).transpose(1, 3)
        y = self.wo_conv(y)
        y = y.transpose(1, 3)
        y = y.reshape(bsz, seq_len, -1)

        if self.output_new_cache_only:
            return y, [k], [v]

        return y, [kh], [vh]

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape

        q = self.wq(hidden_states)
        k = self.wk(hidden_states)
        v = self.wv(hidden_states)
        if self.use_q_gate:
            q, gate = torch.chunk(
                q.view(bsz, seq_len, self.n_heads, self.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(bsz, seq_len, -1)
        else:
            q = q.view(bsz, seq_len, self.n_heads, self.head_dim)
            gate = None
        q = q.transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm and self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)

        if self.use_rope:
            q = self.apply_rope_emb(q, freqs_cos, freqs_sin)
            k = self.apply_rope_emb(k, freqs_cos, freqs_sin)

        if self.use_qk_norm and not self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)
        if getattr(self.config, "enable_r3", False):
            q = torch.matmul(q, self.r3_weight)
            k = torch.matmul(k, self.r3_weight)
        k = k.transpose(2, 3)

        kh, vh = None, None
        # kv cache mode
        if self.use_kv_cache:
            kh = torch.cat([k_caches, k], dim=-1)
            vh = torch.cat([v_caches, v], dim=2)
        # batch_prefill mode
        else:
            kh = k
            vh = v

        kh = repeat_kv(kh, self.num_key_value_groups)
        vh = repeat_kv(vh, self.num_key_value_groups)

        attn = q @ kh
        # gemma2-2b
        if self.attn_logit_softcapping is not None:
            attn = attn / self.attn_logit_softcapping
            attn = torch.tanh(attn)
            attn = attn * self.attn_logit_softcapping
        attn = attn / self.scale
        if self.enable_masked_softmax:
            attn_min = torch.amin(attn, dim=-1, keepdim=True)
            minus_value = -20
            attn = torch.where(atten_mask == 0, attn, attn_min + minus_value)
        else:
            attn = attn + atten_mask
        attn = self.attn_softmax(attn)
        y = attn @ vh
        y = y.transpose(1, 2).reshape(bsz, seq_len, -1)
        if gate is not None:
            y = y * torch.sigmoid(gate)
        y = self.wo(y)

        if self.output_new_cache_only:
            return y, [k], [v]

        return y, [kh], [vh]


class Qwen35DeltaNet(nn.Module):
    def __init__(self, layer_idx: int, config: ModelArgs, output_new_cache_only=False):
        super().__init__()
        del layer_idx
        del output_new_cache_only
        self.hidden_size = config.dim
        self.max_batch_size = config.max_batch_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.scaling = self.head_k_dim**-0.5
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        assert (
            self.num_v_heads % self.num_k_heads == 0
        ), "linear_num_value_heads must be divisible by linear_num_key_heads."

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
            padding=0,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        a = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(a))
        self.norm = Qwen35RMSNormGated(self.head_v_dim, eps=config.norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def _apply_causal_conv(
        self, mixed_qkv: torch.Tensor, conv_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = mixed_qkv.shape
        mixed_qkv = mixed_qkv.transpose(1, 2)
        conv_state = conv_state[:batch_size]
        conv_weight = self.conv1d.weight.squeeze(1).to(mixed_qkv.dtype)

        if seq_len == 1:
            mixed_qkv_last = mixed_qkv.squeeze(-1)
            new_conv_state = torch.cat([conv_state[:, :, 1:], mixed_qkv], dim=-1)
            out = (conv_state[:, :, 1:] * conv_weight[:, :-1].unsqueeze(0)).sum(
                dim=-1
            ) + mixed_qkv_last * conv_weight[:, -1].unsqueeze(0)
            out = F.silu(out.unsqueeze(-1)).to(mixed_qkv.dtype)
            return out.transpose(1, 2).contiguous(), new_conv_state

        hidden_states_new = torch.cat([conv_state, mixed_qkv], dim=-1)
        new_conv_state = hidden_states_new[:, :, -self.conv_kernel_size :]
        conv_outputs = []
        for i in range(seq_len):
            window = hidden_states_new[:, :, i + 1 : i + 1 + self.conv_kernel_size]
            conv_outputs.append((window * conv_weight.unsqueeze(0)).sum(dim=-1))
        out = torch.stack(conv_outputs, dim=-1)
        out = F.silu(out).to(mixed_qkv.dtype)
        return out.transpose(1, 2).contiguous(), new_conv_state

    def _recurrent_gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        initial_dtype = query.dtype
        query = _l2norm(query.float()) * self.scaling
        key = _l2norm(key.float())
        value = value.float()
        beta = beta.float()
        last_recurrent_state = recurrent_state[: query.shape[0]].float()
        outputs = []

        for i in range(query.shape[1]):
            q_t = query[:, i]
            k_t = key[:, i]
            v_t = value[:, i]
            g_t = g[:, i].exp().unsqueeze(-1).unsqueeze(-1)
            g_t = g_t.expand_as(last_recurrent_state)
            beta_t = beta[:, i].unsqueeze(-1).expand_as(v_t)

            last_recurrent_state = last_recurrent_state * g_t
            k_for_state = k_t.unsqueeze(-1).expand_as(last_recurrent_state)
            kv_mem = (last_recurrent_state * k_for_state).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            state_update = k_t.unsqueeze(-1).expand(
                k_t.shape[0], k_t.shape[1], k_t.shape[2], delta.shape[-1]
            ) * delta.unsqueeze(-2).expand(
                delta.shape[0], delta.shape[1], k_t.shape[-1], delta.shape[-1]
            )
            last_recurrent_state = last_recurrent_state + state_update
            q_for_state = q_t.unsqueeze(-1).expand_as(last_recurrent_state)
            outputs.append((last_recurrent_state * q_for_state).sum(dim=-2))

        core_attn_out = torch.stack(outputs, dim=1)
        if last_recurrent_state.shape[0] != recurrent_state.shape[0]:
            last_recurrent_state = torch.cat(
                [
                    last_recurrent_state,
                    recurrent_state[last_recurrent_state.shape[0] :].float(),
                ],
                dim=0,
            )
        return core_attn_out.to(initial_dtype), last_recurrent_state.to(initial_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: torch.Tensor,
        v_caches: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del freqs_cos
        del freqs_sin
        del atten_mask
        del input_pos
        batch_size, seq_len, _ = hidden_states.shape
        conv_state = k_caches
        recurrent_state = v_caches
        if conv_state is None:
            conv_state = torch.zeros(
                batch_size,
                self.conv_dim,
                self.conv_kernel_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        if recurrent_state is None:
            recurrent_state = torch.zeros(
                batch_size,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(
            batch_size, seq_len, -1, self.head_v_dim
        )
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        mixed_qkv, new_conv_state = self._apply_causal_conv(mixed_qkv, conv_state)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        if self.num_v_heads // self.num_k_heads > 1:
            head_repeat = self.num_v_heads // self.num_k_heads
            query = repeat_heads(query, head_repeat)
            key = repeat_heads(key, head_repeat)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * _qnn_softplus(a.float() + self.dt_bias)
        core_attn_out, new_recurrent_state = self._recurrent_gated_delta_rule(
            query, key, value, g, beta, recurrent_state
        )
        core_attn_out = self.norm(
            core_attn_out.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        )
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out), [new_conv_state], [new_recurrent_state]


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.hidden_dim is not None
        self.hidden_dim: int = args.hidden_dim
        self.dim: int = args.dim
        self.w1 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.act_fn = args.act_fn.get_function()

    def prepare_feedfoward_conv(self):
        self.w1_conv = nn.Conv2d(self.dim, self.hidden_dim, 1, bias=False)
        self.w2_conv = nn.Conv2d(self.hidden_dim, self.dim, 1, bias=False)
        self.w3_conv = nn.Conv2d(self.dim, self.hidden_dim, 1, bias=False)

        self.forward_no_conv = self.forward
        self.forward = self.forward_feedfoward_conv
        self.w1_conv.weight.data.copy_(self.w1.weight[:, :, None, None])
        self.w2_conv.weight.data.copy_(self.w2.weight[:, :, None, None])
        self.w3_conv.weight.data.copy_(self.w3.weight[:, :, None, None])

        del self.w1
        del self.w2
        del self.w3

    def forward_feedfoward_conv(self, x):
        bsz, _, _ = x.size()
        x = torch.reshape(x, (bsz, -1, 1, self.dim))
        x = x.transpose(1, 3)  # Transpose right before and after Conv
        x = self.w2_conv(self.act_fn(self.w1_conv(x)) * self.w3_conv(x))
        x = x.transpose(1, 3)
        x = torch.reshape(x, (bsz, -1, self.dim))
        return x

    def forward(self, x):
        return self.w2(self.act_fn(self.w1(x)) * self.w3(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, layer_idx: int, config: ModelArgs, output_new_cache_only=False):
        super().__init__()
        self.dim = config.dim
        self.attention = LlamaAttention(
            layer_idx=layer_idx,
            config=config,
            output_new_cache_only=output_new_cache_only,
        )

        self.feed_forward = FeedForward_REGISTRY.get(
            config.model_architecture, FeedForward
        )(config)
        self.attention_norm = NORM_REGISTRY[config.norm_type](
            config.dim, eps=config.norm_eps
        )
        self.ffn_norm = (
            NORM_REGISTRY[config.norm_type](config.dim, eps=config.norm_eps)
            if config.use_ffn_norm
            else None
        )

        self.post_attention_norm = (
            torch.nn.RMSNorm(config.dim, eps=config.norm_eps)
            if config.post_attention_norm
            else None
        )
        self.residual_multiplier = config.residual_multiplier
        self.post_ffn_norm = (
            torch.nn.RMSNorm(config.dim, eps=config.norm_eps)
            if config.post_ffn_norm
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        hidden_states = self.attention_norm(x)
        h, k_cache, v_cache = self.attention(
            hidden_states=hidden_states,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            atten_mask=atten_mask,
            k_caches=k_caches,
            v_caches=v_caches,
        )
        if self.post_attention_norm:
            h = self.post_attention_norm(h)
        h = (
            x + h * self.residual_multiplier
            if self.residual_multiplier is not None
            else x + h
        )
        hidden_states = hidden_states if self.ffn_norm is None else self.ffn_norm(h)
        out = self.feed_forward(hidden_states)
        if self.post_ffn_norm:
            out = self.post_ffn_norm(out)
        output = (
            h + out * self.residual_multiplier
            if self.residual_multiplier is not None
            else h + out
        )

        return output, k_cache, v_cache


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, layer_idx: int, config: ModelArgs, output_new_cache_only=False):
        super().__init__()
        layer_type = (
            config.layer_types[layer_idx]
            if config.layer_types is not None
            else "full_attention"
        )
        if layer_type == "linear_attention":
            self.attention = Qwen35DeltaNet(
                layer_idx=layer_idx,
                config=config,
                output_new_cache_only=output_new_cache_only,
            )
        elif layer_type == "full_attention":
            self.attention = LlamaAttention(
                layer_idx=layer_idx,
                config=config,
                output_new_cache_only=output_new_cache_only,
            )
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {layer_type}")

        self.feed_forward = FeedForward_REGISTRY.get(
            config.model_architecture, FeedForward
        )(config)
        self.attention_norm = Qwen35RMSNorm(
            config.dim,
            eps=config.norm_eps,
            add_unit_offset=config.rms_norm_add_unit_offset,
        )
        self.ffn_norm = (
            Qwen35RMSNorm(
                config.dim,
                eps=config.norm_eps,
                add_unit_offset=config.rms_norm_add_unit_offset,
            )
            if config.use_ffn_norm
            else None
        )
        self.post_attention_norm = (
            Qwen35RMSNorm(
                config.dim,
                eps=config.norm_eps,
                add_unit_offset=config.rms_norm_add_unit_offset,
            )
            if config.post_attention_norm
            else None
        )
        self.residual_multiplier = config.residual_multiplier
        self.post_ffn_norm = (
            Qwen35RMSNorm(
                config.dim,
                eps=config.norm_eps,
                add_unit_offset=config.rms_norm_add_unit_offset,
            )
            if config.post_ffn_norm
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.attention_norm(x)
        h, k_cache, v_cache = self.attention(
            hidden_states=hidden_states,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            atten_mask=atten_mask,
            k_caches=k_caches,
            v_caches=v_caches,
        )
        if self.post_attention_norm:
            h = self.post_attention_norm(h)
        h = (
            x + h * self.residual_multiplier
            if self.residual_multiplier is not None
            else x + h
        )
        hidden_states = hidden_states if self.ffn_norm is None else self.ffn_norm(h)
        out = self.feed_forward(hidden_states)
        if self.post_ffn_norm:
            out = self.post_ffn_norm(out)
        output = (
            h + out * self.residual_multiplier
            if self.residual_multiplier is not None
            else h + out
        )
        return output, k_cache, v_cache


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        ar_len=1,
        output_new_cache_only=True,
        output_cache=True,
        use_i64_token=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = config.dim
        self.head_dim = config.head_dim
        self.max_batch_size = config.max_batch_size
        self.max_seq_len = config.max_seq_len
        self.max_context_len = config.max_context_len
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_layers = config.n_layers
        self.vocab_size = config.vocab_size
        self.rope_freq_base = config.rope_freq_base
        self.use_kv_cache = config.use_kv_cache
        self.embedding_scale_factor = config.embedding_scale_factor
        self.ar_len = ar_len
        self.output_new_cache_only = output_new_cache_only
        self.use_i64_token = use_i64_token
        self.output_cache = output_cache
        self.kv_io_bit_width = config.kv_io_bit_width
        self.logits_scaling = config.logits_scaling

        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(i, config, self.output_new_cache_only)
                for i in range(config.n_layers)
            ]
        )
        self.norm = NORM_REGISTRY[config.norm_type](config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=config.output_bias)

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        if config.use_hf_rope:
            freqs_cos, freqs_sin = hf_precompute_freqs_cis(
                config.head_dim,
                config.max_context_len,
                config.rope_freq_base,
                config.partial_rotary_factor,
            )
            freqs_cos = freqs_cos[:, : freqs_cos.shape[-1] // 2]
            freqs_sin = freqs_sin[:, : freqs_sin.shape[-1] // 2]
        else:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                config.head_dim,
                config.max_context_len,
                config.rope_freq_base,
                config.use_scaled_rope,
                config.rope_scale_factor,
            )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def prepare_output_conv(self):
        def forward_output_conv(x):
            bsz, _, _ = x.size()
            x = torch.reshape(x, (bsz, -1, 1, self.dim))
            x = x.transpose(1, 3)  # Transpose right before and after Conv
            x = self.output_conv(x)
            x = x.transpose(1, 3)
            x = torch.reshape(x, (bsz, -1, self.vocab_size))
            return x

        self.output_conv = nn.Conv2d(self.dim, self.vocab_size, 1, bias=False)
        self.output_conv.weight.data.copy_(self.output.weight[:, :, None, None])

        del self.output
        self.output = forward_output_conv

    def forward(
        self,
        tokens: torch.Tensor,
        atten_mask: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        *args,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        output_k_cache = []
        output_v_cache = []
        # following tensors should be invariant across batches
        freqs_cos = (
            self.freqs_cos[input_pos][0] if self.use_kv_cache else self.freqs_cos
        )
        freqs_sin = (
            self.freqs_sin[input_pos][0] if self.use_kv_cache else self.freqs_sin
        )

        hidden_states = self.embedding_scale_factor * self.tok_embeddings(tokens)

        for ind, decoder_layer in enumerate(self.layers):
            k_caches = None
            v_caches = None
            if self.use_kv_cache:
                offset_k = ind
                offset_v = self.n_layers + offset_k
                k_caches = args[offset_k]
                v_caches = args[offset_v]

            hidden_states, k, v = decoder_layer(
                hidden_states,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                atten_mask=atten_mask,
                k_caches=k_caches,
                v_caches=v_caches,
            )
            output_k_cache.extend(k)
            output_v_cache.extend(v)

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)

        if self.logits_scaling:
            logits = logits / self.logits_scaling

        if self.output_cache:
            return logits, output_k_cache, output_v_cache
        return logits

    def get_example_inputs(self):
        dtype = torch.int64 if self.use_i64_token else torch.int32
        tokens = torch.randint(
            self.vocab_size, (self.max_batch_size, self.ar_len), dtype=dtype
        )
        atten_mask = AttentionMask(
            CausalAttentionMask(self.max_batch_size, self.ar_len, self.max_context_len)
        )
        if self.use_kv_cache:
            pos_ids = torch.zeros((self.max_batch_size, self.ar_len), dtype=torch.int32)
            k_cache, v_cache = [], []
            for _ in range(self.n_layers):
                # transpose first to decrease the runtime efforts
                k_cache.append(
                    torch.zeros(
                        self.max_batch_size,
                        self.n_kv_heads,
                        self.head_dim,
                        self.max_context_len - self.ar_len,
                    )
                )
                v_cache.append(
                    torch.zeros(
                        self.max_batch_size,
                        self.n_kv_heads,
                        self.max_context_len - self.ar_len,
                        self.head_dim,
                    )
                )
            return (
                tokens,
                atten_mask,
                pos_ids,
                k_cache,
                v_cache,
            )

        return (
            tokens,
            atten_mask,
        )

    def get_metadata(self):
        return {
            "get_ar_len": self.ar_len,
            "get_bos_id": 1,
            "get_eos_id": 2,
            "get_dim": self.dim,
            "get_head_dim": self.head_dim,
            "get_max_batch_size": self.max_batch_size,
            "get_max_seq_len": self.max_seq_len,
            "get_max_context_len": self.max_context_len,
            "get_n_bos": 1,
            "get_n_eos": 1,
            "get_n_kv_heads": self.n_kv_heads,
            "get_n_layers": self.n_layers,
            "get_vocab_size": self.vocab_size,
            "get_use_kv_cache": self.use_kv_cache,
            "get_kv_io_bit_width": self.kv_io_bit_width,
        }


class Qwen35Model(LlamaModel):
    def __init__(
        self,
        config: ModelArgs,
        ar_len=1,
        output_new_cache_only=True,
        output_cache=True,
        use_i64_token=False,
        **kwargs,
    ):
        nn.Module.__init__(self)
        del kwargs
        self.dim = config.dim
        self.head_dim = config.head_dim
        self.max_batch_size = config.max_batch_size
        self.max_seq_len = config.max_seq_len
        self.max_context_len = config.max_context_len
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_layers = config.n_layers
        self.vocab_size = config.vocab_size
        self.rope_freq_base = config.rope_freq_base
        self.use_kv_cache = config.use_kv_cache
        self.embedding_scale_factor = config.embedding_scale_factor
        self.ar_len = ar_len
        self.output_new_cache_only = output_new_cache_only
        self.use_i64_token = use_i64_token
        self.output_cache = output_cache
        self.kv_io_bit_width = config.kv_io_bit_width
        self.logits_scaling = config.logits_scaling
        self.layer_types = config.layer_types

        assert self.layer_types is not None, "Qwen3.5 requires layer_types."
        assert len(self.layer_types) == self.n_layers, (
            f"Length of layer_types ({len(self.layer_types)}) must match "
            f"n_layers ({self.n_layers})"
        )

        self.layers = nn.ModuleList(
            [
                Qwen35DecoderLayer(i, config, self.output_new_cache_only)
                for i in range(config.n_layers)
            ]
        )
        self.norm = Qwen35RMSNorm(
            config.dim,
            eps=config.norm_eps,
            add_unit_offset=config.rms_norm_add_unit_offset,
        )
        self.output = nn.Linear(config.dim, config.vocab_size, bias=config.output_bias)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        if config.use_hf_rope:
            freqs_cos, freqs_sin = hf_precompute_freqs_cis(
                config.head_dim,
                config.max_context_len,
                config.rope_freq_base,
                config.partial_rotary_factor,
            )
            freqs_cos = freqs_cos[:, : freqs_cos.shape[-1] // 2].contiguous()
            freqs_sin = freqs_sin[:, : freqs_sin.shape[-1] // 2].contiguous()
        else:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                config.head_dim,
                config.max_context_len,
                config.rope_freq_base,
                config.use_scaled_rope,
                config.rope_scale_factor,
            )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        self.mrope_section = getattr(config, "mrope_section", [11, 11, 10])
        self.rotary_freq_dim = freqs_cos.shape[-1]

        self.linear_conv_kernel_dim = config.linear_conv_kernel_dim
        self.linear_key_head_dim = config.linear_key_head_dim
        self.linear_value_head_dim = config.linear_value_head_dim
        self.linear_num_key_heads = config.linear_num_key_heads
        self.linear_num_value_heads = config.linear_num_value_heads
        self.linear_key_dim = self.linear_key_head_dim * self.linear_num_key_heads
        self.linear_value_dim = (
            self.linear_value_head_dim * self.linear_num_value_heads
        )
        self.linear_conv_dim = self.linear_key_dim * 2 + self.linear_value_dim

    def _apply_interleaved_mrope(
        self,
        temporal_freqs: torch.Tensor,
        height_freqs: torch.Tensor,
        width_freqs: torch.Tensor,
    ) -> torch.Tensor:
        h_end = min(self.mrope_section[1] * 3, self.rotary_freq_dim)
        w_end = min(self.mrope_section[2] * 3, self.rotary_freq_dim)
        freqs = []
        for idx in range(self.rotary_freq_dim):
            if idx % 3 == 1 and idx < h_end:
                freqs.append(height_freqs[:, idx : idx + 1])
            elif idx % 3 == 2 and idx < w_end:
                freqs.append(width_freqs[:, idx : idx + 1])
            else:
                freqs.append(temporal_freqs[:, idx : idx + 1])
        return torch.cat(freqs, dim=-1)

    def _get_qwen35_freqs(
        self, input_pos: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_kv_cache:
            return self.freqs_cos, self.freqs_sin
        if input_pos is not None and input_pos.dim() == 3:
            input_pos = input_pos.to(torch.long)
            return (
                self._apply_interleaved_mrope(
                    self.freqs_cos[input_pos[0]][0],
                    self.freqs_cos[input_pos[1]][0],
                    self.freqs_cos[input_pos[2]][0],
                ),
                self._apply_interleaved_mrope(
                    self.freqs_sin[input_pos[0]][0],
                    self.freqs_sin[input_pos[1]][0],
                    self.freqs_sin[input_pos[2]][0],
                ),
            )
        return self.freqs_cos[input_pos][0], self.freqs_sin[input_pos][0]

    def get_example_inputs(self):
        dtype = torch.int64 if self.use_i64_token else torch.int32
        tokens = torch.randint(
            self.vocab_size, (self.max_batch_size, self.ar_len), dtype=dtype
        )
        atten_mask = AttentionMask(
            CausalAttentionMask(self.max_batch_size, self.ar_len, self.max_context_len)
        )
        if self.use_kv_cache:
            pos_ids = torch.zeros((self.max_batch_size, self.ar_len), dtype=torch.int32)
            k_cache, v_cache = [], []
            for layer_type in self.layer_types:
                if layer_type == "linear_attention":
                    k_cache.append(
                        torch.zeros(
                            self.max_batch_size,
                            self.linear_conv_dim,
                            self.linear_conv_kernel_dim,
                        )
                    )
                    v_cache.append(
                        torch.zeros(
                            self.max_batch_size,
                            self.linear_num_value_heads,
                            self.linear_key_head_dim,
                            self.linear_value_head_dim,
                        )
                    )
                else:
                    k_cache.append(
                        torch.zeros(
                            self.max_batch_size,
                            self.n_kv_heads,
                            self.head_dim,
                            self.max_context_len - self.ar_len,
                        )
                    )
                    v_cache.append(
                        torch.zeros(
                            self.max_batch_size,
                            self.n_kv_heads,
                            self.max_context_len - self.ar_len,
                            self.head_dim,
                        )
                    )
            return (
                tokens,
                atten_mask,
                pos_ids,
                k_cache,
                v_cache,
            )

        return (
            tokens,
            atten_mask,
        )

    def get_metadata(self):
        meta_data = super().get_metadata()
        meta_data["get_bos_id"] = 248045
        meta_data["get_eos_id"] = 248046
        meta_data["get_eos_ids"] = [248046, 248044]
        meta_data["get_n_eos"] = 2
        return meta_data


class Qwen35ModelWithoutEmbedding(Qwen35Model):
    def __init__(
        self,
        config: ModelArgs,
        ar_len=1,
        output_new_cache_only=True,
        output_cache=True,
        use_i64_token=False,
        **kwargs,
    ):
        super().__init__(
            config=config,
            ar_len=ar_len,
            output_new_cache_only=output_new_cache_only,
            output_cache=output_cache,
            use_i64_token=use_i64_token,
        )
        self.audio_token_id = kwargs.get("audio_token_id", None)
        self.image_token_id = kwargs.get("image_token_id", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        atten_mask: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        *args,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        output_k_cache = []
        output_v_cache = []
        freqs_cos, freqs_sin = self._get_qwen35_freqs(input_pos)

        hidden_states = self.embedding_scale_factor * hidden_states

        for ind, decoder_layer in enumerate(self.layers):
            k_caches = None
            v_caches = None
            if self.use_kv_cache:
                offset_k = ind
                offset_v = self.n_layers + offset_k
                k_caches = args[offset_k]
                v_caches = args[offset_v]

            hidden_states, k, v = decoder_layer(
                hidden_states,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                atten_mask=atten_mask,
                k_caches=k_caches,
                v_caches=v_caches,
            )
            output_k_cache.extend(k)
            output_v_cache.extend(v)

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)

        if self.logits_scaling:
            logits = logits / self.logits_scaling

        if self.output_cache:
            return logits, output_k_cache, output_v_cache
        return logits

    def get_example_inputs(self):
        hidden_states = torch.randn(
            (self.max_batch_size, self.ar_len, self.dim), dtype=torch.float32
        )
        atten_mask = AttentionMask(
            CausalAttentionMask(self.max_batch_size, self.ar_len, self.max_context_len)
        )
        if self.use_kv_cache:
            pos_ids = torch.zeros(
                (3, self.max_batch_size, self.ar_len), dtype=torch.int32
            )
            k_cache, v_cache = [], []
            for layer_type in self.layer_types:
                if layer_type == "linear_attention":
                    k_cache.append(
                        torch.zeros(
                            self.max_batch_size,
                            self.linear_conv_dim,
                            self.linear_conv_kernel_dim,
                        )
                    )
                    v_cache.append(
                        torch.zeros(
                            self.max_batch_size,
                            self.linear_num_value_heads,
                            self.linear_key_head_dim,
                            self.linear_value_head_dim,
                        )
                    )
                else:
                    k_cache.append(
                        torch.zeros(
                            self.max_batch_size,
                            self.n_kv_heads,
                            self.head_dim,
                            self.max_context_len - self.ar_len,
                        )
                    )
                    v_cache.append(
                        torch.zeros(
                            self.max_batch_size,
                            self.n_kv_heads,
                            self.max_context_len - self.ar_len,
                            self.head_dim,
                        )
                    )
            return (
                hidden_states,
                atten_mask,
                pos_ids,
                k_cache,
                v_cache,
            )

        return (
            hidden_states,
            atten_mask,
        )

    def get_metadata(self):
        meta_data = super().get_metadata()
        if self.audio_token_id:
            meta_data["audio_token_id"] = self.audio_token_id
        if self.image_token_id:
            meta_data["image_token_id"] = self.image_token_id
        return meta_data


class LlamaModelWithoutEmbedding(LlamaModel):
    def __init__(
        self,
        config: ModelArgs,
        ar_len=1,
        output_new_cache_only=True,
        output_cache=True,
        use_i64_token=False,
        **kwargs,
    ):
        super().__init__(
            config=config,
            ar_len=ar_len,
            output_new_cache_only=output_new_cache_only,
            output_cache=output_cache,
            use_i64_token=use_i64_token,
            **kwargs,
        )

        # Set the audio/image token ID from keyword arguments. It defaults to None if not provided.
        # If an ID is provided, it will be stored in the model's metadata.
        self.audio_token_id = kwargs.get("audio_token_id", None)
        self.image_token_id = kwargs.get("image_token_id", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        atten_mask: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        *args,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        output_k_cache = []
        output_v_cache = []
        # following tensors should be invariant across batches
        freqs_cos = (
            self.freqs_cos[input_pos][0] if self.use_kv_cache else self.freqs_cos
        )
        freqs_sin = (
            self.freqs_sin[input_pos][0] if self.use_kv_cache else self.freqs_sin
        )

        hidden_states = self.embedding_scale_factor * hidden_states

        for ind, decoder_layer in enumerate(self.layers):
            k_caches = None
            v_caches = None
            if self.use_kv_cache:
                offset_k = ind
                offset_v = self.n_layers + offset_k
                k_caches = args[offset_k]
                v_caches = args[offset_v]

            hidden_states, k, v = decoder_layer(
                hidden_states,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                atten_mask=atten_mask,
                k_caches=k_caches,
                v_caches=v_caches,
            )
            output_k_cache.extend(k)
            output_v_cache.extend(v)

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)

        if self.logits_scaling:
            logits = logits / self.logits_scaling

        if self.output_cache:
            return logits, output_k_cache, output_v_cache
        return logits

    def get_example_inputs(self):
        hidden_states = torch.randn(
            (self.max_batch_size, self.ar_len, self.dim), dtype=torch.float32
        )
        atten_mask = AttentionMask(
            CausalAttentionMask(self.max_batch_size, self.ar_len, self.max_seq_len)
        )
        if self.use_kv_cache:
            pos_ids = torch.zeros((self.max_batch_size, self.ar_len), dtype=torch.int32)
            k_cache, v_cache = [], []

            for _ in range(self.n_layers):
                # transpose first to decrease the runtime efforts
                k_cache.append(
                    torch.zeros(
                        self.max_batch_size,
                        self.n_kv_heads,
                        self.head_dim,
                        self.max_seq_len - self.ar_len,
                    )
                )
                v_cache.append(
                    torch.zeros(
                        self.max_batch_size,
                        self.n_kv_heads,
                        self.max_seq_len - self.ar_len,
                        self.head_dim,
                    )
                )
            return (
                hidden_states,
                atten_mask,
                pos_ids,
                k_cache,
                v_cache,
            )

        return (
            hidden_states,
            atten_mask,
        )

    def get_metadata(self):
        meta_data = super().get_metadata()
        if self.audio_token_id:
            meta_data["audio_token_id"] = self.audio_token_id
        if self.image_token_id:
            meta_data["image_token_id"] = self.image_token_id
        return meta_data


class MultiScopeAwareLlamaModelWithoutEmbedding(LlamaModelWithoutEmbedding):
    """
    Multimodal decoder variant that supports per-layer full vs sliding-window attention.

    This mirrors MultiScopeAwareLlamaModel but accepts hidden states instead of
    token IDs because multimodal embeddings are prepared outside the decoder.
    """

    def __init__(
        self,
        config: ModelArgs,
        ar_len=1,
        output_new_cache_only=True,
        output_cache=True,
        use_i64_token=False,
        **kwargs,
    ):
        super().__init__(
            config=config,
            ar_len=ar_len,
            output_new_cache_only=output_new_cache_only,
            output_cache=output_cache,
            use_i64_token=use_i64_token,
            **kwargs,
        )
        self.final_logit_softcapping = config.final_logit_softcapping

        if config.layer_types is None:
            raise ValueError(
                "MultiScopeAwareLlamaModelWithoutEmbedding requires `layer_types` in ModelArgs."
            )

        self.layer_types = list(config.layer_types)[: config.n_layers]
        if len(self.layer_types) != config.n_layers:
            raise ValueError(
                f"Length of layer_types ({len(self.layer_types)}) must match n_layers ({config.n_layers})"
            )
        if config.local_rope_theta is None:
            raise ValueError(
                "local_rope_theta should not be None, please set it explicitly in config."
            )

        self.sliding_window = config.sliding_window
        local_freqs_cos, local_freqs_sin = hf_precompute_freqs_cis(
            config.head_dim,
            config.max_context_len,
            config.local_rope_theta,
            config.partial_rotary_factor,
        )
        local_freqs_cos = local_freqs_cos[:, : local_freqs_cos.shape[-1] // 2]
        local_freqs_sin = local_freqs_sin[:, : local_freqs_sin.shape[-1] // 2]
        self.register_buffer("local_freqs_cos", local_freqs_cos, persistent=False)
        self.register_buffer("local_freqs_sin", local_freqs_sin, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        atten_mask: torch.Tensor,
        window_atten_mask: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        *args,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        output_k_cache = []
        output_v_cache = []
        freqs_cos = (
            self.freqs_cos[input_pos][0] if self.use_kv_cache else self.freqs_cos
        )
        freqs_sin = (
            self.freqs_sin[input_pos][0] if self.use_kv_cache else self.freqs_sin
        )
        local_freqs_cos = (
            self.local_freqs_cos[input_pos][0]
            if self.use_kv_cache
            else self.local_freqs_cos
        )
        local_freqs_sin = (
            self.local_freqs_sin[input_pos][0]
            if self.use_kv_cache
            else self.local_freqs_sin
        )

        for ind, decoder_layer in enumerate(self.layers):
            k_caches = None
            v_caches = None
            if self.use_kv_cache:
                offset_k = ind
                offset_v = self.n_layers + offset_k
                k_caches = args[offset_k]
                v_caches = args[offset_v]

            if self.layer_types[ind] == "sliding_attention":
                hidden_states, k, v = decoder_layer(
                    hidden_states,
                    freqs_cos=local_freqs_cos,
                    freqs_sin=local_freqs_sin,
                    atten_mask=window_atten_mask,
                    k_caches=k_caches,
                    v_caches=v_caches,
                )
            else:
                hidden_states, k, v = decoder_layer(
                    hidden_states,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    atten_mask=atten_mask,
                    k_caches=k_caches,
                    v_caches=v_caches,
                )

            output_k_cache.extend(k)
            output_v_cache.extend(v)

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)
        if self.final_logit_softcapping:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping

        if self.output_cache:
            return logits, output_k_cache, output_v_cache
        return logits

    def get_example_inputs(self):
        inputs = list(super().get_example_inputs())
        causal_mask = CausalAttentionMask(
            self.max_batch_size, self.ar_len, self.max_context_len
        )
        sliding_window_mask = SlidingWindowAttentionMask(
            self.max_batch_size,
            self.ar_len,
            self.max_context_len,
            sliding_window=self.sliding_window,
        )
        inputs[1] = AttentionMask([causal_mask, sliding_window_mask])
        return tuple(inputs)

    def get_metadata(self):
        meta_data = super().get_metadata()
        meta_data["get_sliding_window"] = self.sliding_window
        return meta_data


class MultiScopeAwareLlamaModelWithoutEmbedding_swa(
    MultiScopeAwareLlamaModelWithoutEmbedding
):
    def _get_window_context_len_swa(self) -> int:
        # Keep the decode KV cache length at (sliding_window - 1) while
        # expanding the prompt-side attention span by (ar_len - 1) so the
        # earliest token in a chunk still sees the full sliding window.
        return min(self.max_seq_len, self.sliding_window + self.ar_len - 1)

    def get_example_inputs(self):
        hidden_states = torch.randn(
            (self.max_batch_size, self.ar_len, self.dim), dtype=torch.float32
        )

        causal_mask = CausalAttentionMask(
            self.max_batch_size, self.ar_len, self.max_seq_len
        )
        window_context_len = self._get_window_context_len_swa()
        sliding_window_mask = SlidingWindowAttentionMask(
            self.max_batch_size,
            self.ar_len,
            window_context_len,
            sliding_window=self.sliding_window,
        )

        atten_mask = AttentionMask([causal_mask, sliding_window_mask])

        if not self.use_kv_cache:
            return (
                hidden_states,
                atten_mask,
            )

        pos_ids = torch.zeros((self.max_batch_size, self.ar_len), dtype=torch.int32)

        full_cache_len = self.max_seq_len - self.ar_len
        window_cache_len = max(0, window_context_len - self.ar_len)

        k_cache, v_cache = [], []
        for layer_idx in range(self.n_layers):
            cache_len = (
                window_cache_len
                if self.layer_types[layer_idx] == "sliding_attention"
                else full_cache_len
            )
            k_cache.append(
                torch.zeros(
                    self.max_batch_size,
                    self.n_kv_heads,
                    self.head_dim,
                    cache_len,
                )
            )
            v_cache.append(
                torch.zeros(
                    self.max_batch_size,
                    self.n_kv_heads,
                    cache_len,
                    self.head_dim,
                )
            )

        return (
            hidden_states,
            atten_mask,
            pos_ids,
            k_cache,
            v_cache,
        )


class MultiScopeAwareLlamaModel(LlamaModel):
    def __init__(
        self,
        config: ModelArgs,
        ar_len=1,
        output_new_cache_only=True,
        output_cache=True,
        use_i64_token=False,
        **kwargs,
    ):
        super().__init__(
            config=config,
            ar_len=ar_len,
            output_new_cache_only=output_new_cache_only,
            output_cache=output_cache,
            use_i64_token=use_i64_token,
            **kwargs,
        )
        # Parameter final_logit_softcapping is not necessary for all
        self.final_logit_softcapping = config.final_logit_softcapping

        # Gemma2/Gemma3 requires additional configuration parameters:
        # - layer_types: Specifies the type of each layer (e.g., full vs. sliding attention)
        # - local_rope_theta: Base frequency for local RoPE
        # - sliding_window: Size of the sliding window for local attention
        self.layer_types = config.layer_types
        if self.layer_types is not None:
            assert len(self.layer_types) == self.n_layers, (
                f"Length of layer_types ({len(self.layer_types)}) must match "
                f"n_layers ({self.n_layers})"
            )
        assert (
            config.local_rope_theta is not None
        ), "local_rope_theta should not be None, please set it explicitly in config."

        self.sliding_window = config.sliding_window
        local_freqs_cos, local_freqs_sin = hf_precompute_freqs_cis(
            config.head_dim,
            config.max_context_len,
            config.local_rope_theta,
            config.partial_rotary_factor,
        )
        local_freqs_cos = local_freqs_cos[:, : local_freqs_cos.shape[-1] // 2]
        local_freqs_sin = local_freqs_sin[:, : local_freqs_sin.shape[-1] // 2]
        self.register_buffer("local_freqs_cos", local_freqs_cos, persistent=False)
        self.register_buffer("local_freqs_sin", local_freqs_sin, persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        atten_mask: torch.Tensor,
        window_atten_mask: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        *args,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:

        output_k_cache = []
        output_v_cache = []
        # following tensors should be invariant across batches
        freqs_cos = (
            self.freqs_cos[input_pos][0] if self.use_kv_cache else self.freqs_cos
        )
        freqs_sin = (
            self.freqs_sin[input_pos][0] if self.use_kv_cache else self.freqs_sin
        )
        local_freqs_cos = (
            self.local_freqs_cos[input_pos][0]
            if self.use_kv_cache
            else self.local_freqs_cos
        )
        local_freqs_sin = (
            self.local_freqs_sin[input_pos][0]
            if self.use_kv_cache
            else self.local_freqs_sin
        )

        hidden_states = self.embedding_scale_factor * self.tok_embeddings(tokens)
        for ind, decoder_layer in enumerate(self.layers):
            k_caches = None
            v_caches = None
            if self.use_kv_cache:
                offset_k = ind
                offset_v = self.n_layers + offset_k
                k_caches = args[offset_k]
                v_caches = args[offset_v]

            if self.layer_types[ind] == "sliding_attention":
                hidden_states, k, v = decoder_layer(
                    hidden_states,
                    freqs_cos=local_freqs_cos,
                    freqs_sin=local_freqs_sin,
                    atten_mask=window_atten_mask,
                    k_caches=k_caches,
                    v_caches=v_caches,
                )
            else:
                hidden_states, k, v = decoder_layer(
                    hidden_states,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    atten_mask=atten_mask,
                    k_caches=k_caches,
                    v_caches=v_caches,
                )

            output_k_cache.extend(k)
            output_v_cache.extend(v)

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)
        if self.final_logit_softcapping:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping

        if self.output_cache:
            return logits, output_k_cache, output_v_cache
        return logits

    def get_example_inputs(self):
        inputs = list(super().get_example_inputs())
        causal_mask = CausalAttentionMask(
            self.max_batch_size, self.ar_len, self.max_context_len
        )
        sliding_window_mask = SlidingWindowAttentionMask(
            self.max_batch_size,
            self.ar_len,
            self.max_context_len,
            sliding_window=self.sliding_window,
        )
        # Don't reverse the order of attention mask
        inputs[1] = AttentionMask([causal_mask, sliding_window_mask])
        return tuple(inputs)

    def get_metadata(self):
        meta_data = super().get_metadata()
        meta_data["get_sliding_window"] = self.sliding_window
        return meta_data


class MultiScopeAwareLlamaModel_swa(MultiScopeAwareLlamaModel):
    def _get_window_context_len_swa(self) -> int:
        # Keep the decode KV cache length at (sliding_window - 1) while
        # expanding the prompt-side attention span by (ar_len - 1) so the
        # earliest token in a chunk still sees the full sliding window.
        return min(self.max_seq_len, self.sliding_window + self.ar_len - 1)

    def get_example_inputs(self, use_kv_cache=True):
        dtype = torch.int64 if self.use_i64_token else torch.int32
        tokens = torch.randint(
            self.vocab_size, (self.max_batch_size, self.ar_len), dtype=dtype
        )

        causal_mask = CausalAttentionMask(
            self.max_batch_size, self.ar_len, self.max_seq_len
        )
        window_context_len = self._get_window_context_len_swa()
        sliding_window_mask = SlidingWindowAttentionMask(
            self.max_batch_size,
            self.ar_len,
            window_context_len,
            sliding_window=self.sliding_window,
        )

        atten_mask = AttentionMask([causal_mask, sliding_window_mask])

        if not use_kv_cache:
            return (
                tokens,
                atten_mask,
            )

        pos_ids = torch.zeros((self.max_batch_size, self.ar_len), dtype=torch.int32)

        full_cache_len = self.max_seq_len - self.ar_len
        window_cache_len = max(0, window_context_len - self.ar_len)

        k_cache, v_cache = [], []
        for layer_idx in range(self.n_layers):
            cache_len = (
                window_cache_len
                if self.layer_types[layer_idx] == "sliding_attention"
                else full_cache_len
            )
            k_cache.append(
                torch.zeros(
                    self.max_batch_size,
                    self.n_kv_heads,
                    self.head_dim,
                    cache_len,
                )
            )
            v_cache.append(
                torch.zeros(
                    self.max_batch_size,
                    self.n_kv_heads,
                    cache_len,
                    self.head_dim,
                )
            )

        return (
            tokens,
            atten_mask,
            pos_ids,
            k_cache,
            v_cache,
        )
