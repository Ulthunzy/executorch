# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Valid-length causal GQA in the shared HTP operator package.

FD descriptors are a device-only boundary. Calibration should use real cache
values and ``attention_reference`` before replacing cache inputs with descriptors.
"""

import math

import torch
from torch.library import Library

attention_lib = Library("qnn_custom_ops", "FRAGMENT")
attention_lib.define(
    "attention_valid(Tensor query, Tensor key_descriptor, Tensor new_key, "
    "Tensor value_descriptor, Tensor new_value, Tensor lengths, Tensor grids) -> Tensor"
)


@torch.library.impl(attention_lib, "attention_valid", "CompositeExplicitAutograd")
def attention_valid(
    query, key_descriptor, new_key, value_descriptor, new_value, lengths, grids
):
    raise RuntimeError(
        "attention_valid uses DSP FD descriptors and must be delegated to QNN; "
        "use attention_reference with real cache tensors for host evaluation"
    )


@torch.library.register_fake("qnn_custom_ops::attention_valid")
def attention_valid_fake(
    query, key_descriptor, new_key, value_descriptor, new_value, lengths, grids
):
    torch._check(query.ndim == 4)
    torch._check(new_key.ndim == 4)
    torch._check(new_value.ndim == 4)
    batch, heads, queries, dimension = query.shape
    kv_heads = new_key.shape[1]
    for size in (batch, heads, queries, dimension, kv_heads):
        torch._check(size > 0)
    torch._check(dimension <= 4096)
    torch._check(query.numel() <= 2**31 - 1)
    torch._check(heads % kv_heads == 0)
    torch._check(new_key.shape == (batch, kv_heads, dimension, queries))
    torch._check(new_value.shape == (batch, kv_heads, queries, dimension))
    for descriptor in (key_descriptor, value_descriptor, lengths):
        torch._check(descriptor.dtype == torch.int32)
        torch._check(descriptor.shape == (1, 1, 1, 3))
    torch._check(grids.dtype == torch.float32)
    torch._check(grids.shape == (1, 1, 1, 13))
    torch._check(query.dtype in (torch.float16, torch.float32))
    torch._check(query.dtype == new_key.dtype)
    torch._check(query.dtype == new_value.dtype)
    return torch.empty_like(query)


def attention_quant_annotations(quant_config):
    from executorch.backends.qualcomm.custom_op.annotator import (
        CustomOpsQuantAnnotator,
        IOQuantConfig,
    )

    for spec in (quant_config.input_activation, quant_config.output_activation):
        if (
            spec is None
            or spec.quant_min != 0
            or spec.quant_max != 65535
            or spec.qscheme not in (torch.per_tensor_affine, torch.per_tensor_symmetric)
        ):
            raise ValueError(
                "AttentionValid requires per-tensor U16 activation quantization"
            )
    annotator = CustomOpsQuantAnnotator()
    annotator.register_annotation(
        torch.ops.qnn_custom_ops.attention_valid.default,
        IOQuantConfig(
            input_quant_specs={
                index: quant_config.input_activation for index in (0, 2, 4)
            },
            output_quant_specs={0: quant_config.output_activation},
        ),
    )
    return (annotator.build_annotation_fn(),)


def attention_reference(
    query, key_cache, new_key, value_cache, new_value, lengths, *, scale=None
):
    """FP32 causal reference, using only the visible history and valid queries.

    Historical K is [B,Hkv,D,C], V is [B,Hkv,C,D]; new K/V use Q instead
    of C. lengths is [valid, first, history], shared across the batch.
    This reference has no quantization or descriptor side effects.
    """
    if query.ndim != 4 or new_key.ndim != 4:
        raise ValueError("expected rank-four query and new_key")
    batch, heads, queries, dimension = query.shape
    kv_heads = new_key.shape[1]
    if min(batch, heads, queries, dimension, kv_heads) <= 0 or heads % kv_heads:
        raise ValueError("query heads must be a positive multiple of KV heads")
    if key_cache.ndim != 4:
        raise ValueError("expected rank-four key cache")
    capacity = key_cache.shape[-1]
    if (
        key_cache.shape != (batch, kv_heads, dimension, capacity)
        or value_cache.shape != (batch, kv_heads, capacity, dimension)
        or new_key.shape != (batch, kv_heads, dimension, queries)
        or new_value.shape != (batch, kv_heads, queries, dimension)
    ):
        raise ValueError("incompatible query, key or value layout")
    if lengths.numel() != 3 or lengths.dtype != torch.int32:
        raise ValueError("lengths must contain three int32 values")
    valid, first, history = lengths.flatten().tolist()
    if not 1 <= valid <= queries or first < 0 or not 0 <= history <= capacity - first:
        raise ValueError("invalid valid/first/history range")
    scale = dimension**-0.5 if scale is None else scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    keys = torch.cat(
        (key_cache[..., first : first + history], new_key[..., :valid]), dim=-1
    )
    values = torch.cat(
        (value_cache[..., first : first + history, :], new_value[..., :valid, :]),
        dim=-2,
    )
    groups = heads // kv_heads
    keys = keys.repeat_interleave(groups, dim=1).float()
    values = values.repeat_interleave(groups, dim=1).float()
    scores = torch.matmul(query[..., :valid, :].float(), keys) * scale
    visible = torch.arange(history + valid, device=query.device)[None, :] <= (
        history + torch.arange(valid, device=query.device)[:, None]
    )
    scores.masked_fill_(~visible, float("-inf"))
    output = torch.zeros_like(query)
    output[..., :valid, :] = torch.matmul(torch.softmax(scores, dim=-1), values).to(
        query.dtype
    )
    return output
