# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark a generic DeltaNet custom op against the decomposed DeltaNet graph."""

import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from multiprocessing.connection import Client
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from executorch.backends.qualcomm.custom_op.annotator import (
    CustomOpsQuantAnnotator,
    IOQuantConfig,
)
from executorch.backends.qualcomm.export_utils import (
    build_executorch_binary,
    get_backend_type,
    make_quantizer,
    QnnConfig,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.backends.qualcomm.quantizer.qconfig import (
    get_16a16w_qnn_ptq_config,
    get_16a4w_qnn_ptq_config,
    get_16a8w_qnn_ptq_config,
    get_8a4w_qnn_ptq_config,
    get_8a8w_qnn_ptq_config,
    get_fp16a8w_qnn_ptq_config,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    _soc_info_table,
    QcomChipset,
    QnnExecuTorchOpPackageInfo,
    QnnExecuTorchOpPackageOptions,
    QnnExecuTorchOpPackagePlatform,
    QnnExecuTorchOpPackageTarget,
)
from torch.library import impl, Library
from torchao.quantization.pt2e.quantizer import SharedQuantizationSpec
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


_DELTA_NET_SCALING = 0.08838834764831845
_DELTA_NET_EPS = 1e-6

delta_net_lib = Library("qnn_custom_ops", "DEF")
delta_net_lib.define(
    "deltanet("
    "Tensor query, "
    "Tensor key, "
    "Tensor value, "
    "Tensor g, "
    "Tensor beta, "
    "Tensor recurrent_state, "
    "float scaling, "
    "float epsilon"
    ") -> (Tensor, Tensor)"
)
delta_net_lib.define(
    "deltanet.out("
    "Tensor query, "
    "Tensor key, "
    "Tensor value, "
    "Tensor g, "
    "Tensor beta, "
    "Tensor recurrent_state, "
    "float scaling, "
    "float epsilon, "
    "*, Tensor(a!) core_attn_out, Tensor(b!) new_recurrent_state"
    ") -> (Tensor(a!), Tensor(b!))"
)
delta_net_lib.define(
    "deltanet_packed("
    "Tensor query, "
    "Tensor key, "
    "Tensor value, "
    "Tensor g, "
    "Tensor beta, "
    "Tensor recurrent_state"
    ") -> Tensor"
)
delta_net_lib.define(
    "deltanet_packed.out("
    "Tensor query, "
    "Tensor key, "
    "Tensor value, "
    "Tensor g, "
    "Tensor beta, "
    "Tensor recurrent_state, "
    "*, Tensor(a!) packed_out"
    ") -> Tensor(a!)"
)
delta_net_lib.define(
    "deltanet_core("
    "Tensor query, "
    "Tensor key, "
    "Tensor value, "
    "Tensor decay, "
    "Tensor beta, "
    "Tensor recurrent_state"
    ") -> (Tensor, Tensor)"
)
delta_net_lib.define(
    "deltanet_core.out("
    "Tensor query, "
    "Tensor key, "
    "Tensor value, "
    "Tensor decay, "
    "Tensor beta, "
    "Tensor recurrent_state, "
    "*, Tensor(a!) core_attn_out, Tensor(b!) new_recurrent_state"
    ") -> (Tensor(a!), Tensor(b!))"
)
delta_net_lib.define(
    "deltanet_core_stream("
    "Tensor query, Tensor key, Tensor value, Tensor decay, Tensor beta, "
    "Tensor recurrent_state"
    ") -> (Tensor, Tensor)"
)
delta_net_lib.define(
    "deltanet_core_fp16_state("
    "Tensor query, Tensor key, Tensor value, Tensor decay, Tensor beta, "
    "Tensor recurrent_state) -> (Tensor, Tensor)"
)
delta_net_lib.define(
    "deltanet_core_fp16_state.out("
    "Tensor query, Tensor key, Tensor value, Tensor decay, Tensor beta, "
    "Tensor recurrent_state, *, Tensor(a!) core_attn_out, "
    "Tensor(b!) new_recurrent_state) -> (Tensor(a!), Tensor(b!))"
)
delta_net_lib.define(
    "deltanet_core_fp16_accum("
    "Tensor query, "
    "Tensor key, "
    "Tensor value, "
    "Tensor decay, "
    "Tensor beta, "
    "Tensor recurrent_state"
    ") -> (Tensor, Tensor)"
)
delta_net_lib.define(
    "deltanet_core_fp16_accum.out("
    "Tensor query, "
    "Tensor key, "
    "Tensor value, "
    "Tensor decay, "
    "Tensor beta, "
    "Tensor recurrent_state, "
    "*, Tensor(a!) core_attn_out, Tensor(b!) new_recurrent_state"
    ") -> (Tensor(a!), Tensor(b!))"
)
delta_net_lib.define("deltanet_identity(Tensor input) -> Tensor")
delta_net_lib.define(
    "deltanet_identity.out(Tensor input, *, Tensor(a!) output) -> Tensor(a!)"
)
delta_net_lib.define("deltanet_decay_state(Tensor recurrent_state, Tensor g) -> Tensor")
delta_net_lib.define(
    "deltanet_decay_state.out("
    "Tensor recurrent_state, "
    "Tensor g, "
    "*, Tensor(a!) output"
    ") -> Tensor(a!)"
)
delta_net_lib.define(
    "deltanet_state_passthrough(Tensor recurrent_state, Tensor g) -> Tensor"
)
delta_net_lib.define(
    "deltanet_state_passthrough.out("
    "Tensor recurrent_state, "
    "Tensor g, "
    "*, Tensor(a!) output"
    ") -> Tensor(a!)"
)


def _l2norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _scalar_attr(value) -> float:
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


def _deltanet_core_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    promote_float: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    initial_dtype = query.dtype
    if promote_float:
        query = query.float()
        key = key.float()
        value = value.float()
        beta = beta.float()
        recurrent_state = recurrent_state.float()
        g = g.float()
    last_recurrent_state = recurrent_state[: query.shape[0]]
    outputs = []

    for i in range(query.shape[1]):
        q_t = query[:, i]
        k_t = key[:, i]
        v_t = value[:, i]
        g_t = g[:, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(
            -1
        ) * delta.unsqueeze(-2)
        outputs.append((last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2))

    return torch.stack(outputs, dim=1).to(initial_dtype), last_recurrent_state.to(
        initial_dtype
    )


def _deltanet_core_decay_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    promote_float: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    initial_dtype = query.dtype
    if promote_float:
        query = query.float()
        key = key.float()
        value = value.float()
        decay = decay.float()
        beta = beta.float()
        recurrent_state = recurrent_state.float()
    last_recurrent_state = recurrent_state[: query.shape[0]]
    outputs = []

    for i in range(query.shape[1]):
        q_t = query[:, i]
        k_t = key[:, i]
        v_t = value[:, i]
        decay_t = decay[:, i].unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * decay_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(
            -1
        ) * delta.unsqueeze(-2)
        outputs.append((last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2))

    return torch.stack(outputs, dim=1).to(initial_dtype), last_recurrent_state.to(
        initial_dtype
    )


def _deltanet_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    scaling: float,
    epsilon: float,
    promote_float: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scaling = _scalar_attr(scaling)
    epsilon = _scalar_attr(epsilon)
    norm_query = query.float() if promote_float else query
    norm_key = key.float() if promote_float else key
    return _deltanet_core_reference(
        _l2norm(norm_query, epsilon) * scaling,
        _l2norm(norm_key, epsilon),
        value,
        g,
        beta,
        recurrent_state,
        promote_float=promote_float,
    )


def _pack_deltanet_outputs(
    core_attn_out: torch.Tensor,
    new_recurrent_state: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (core_attn_out.reshape(-1), new_recurrent_state.reshape(-1)),
        dim=0,
    )


def _unpack_deltanet_outputs(
    packed: torch.Tensor,
    output_shape: Sequence[int],
    state_shape: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    output_numel = math.prod(output_shape)
    packed_flat = packed.reshape(-1)
    return (
        packed_flat[:output_numel].reshape(output_shape),
        packed_flat[output_numel : output_numel + math.prod(state_shape)].reshape(
            state_shape
        ),
    )


@impl(delta_net_lib, "deltanet", dispatch_key="CompositeExplicitAutograd")
def deltanet_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    scaling: float,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _deltanet_reference(
        query, key, value, g, beta, recurrent_state, scaling, epsilon
    )


@impl(delta_net_lib, "deltanet.out", dispatch_key="CompositeExplicitAutograd")
def deltanet_out_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    scaling: float,
    epsilon: float,
    *,
    core_attn_out: torch.Tensor,
    new_recurrent_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ref_out, ref_state = _deltanet_reference(
        query, key, value, g, beta, recurrent_state, scaling, epsilon
    )
    core_attn_out.copy_(ref_out)
    new_recurrent_state.copy_(ref_state)
    return core_attn_out, new_recurrent_state


@impl(delta_net_lib, "deltanet_packed", dispatch_key="CompositeExplicitAutograd")
def deltanet_packed_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
) -> torch.Tensor:
    ref_out, ref_state = _deltanet_core_reference(
        query,
        key,
        value,
        g,
        beta,
        recurrent_state,
    )
    return _pack_deltanet_outputs(ref_out, ref_state)


@impl(delta_net_lib, "deltanet_packed.out", dispatch_key="CompositeExplicitAutograd")
def deltanet_packed_out_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    *,
    packed_out: torch.Tensor,
) -> torch.Tensor:
    packed_out.copy_(deltanet_packed_impl(query, key, value, g, beta, recurrent_state))
    return packed_out


@impl(delta_net_lib, "deltanet_core", dispatch_key="CompositeExplicitAutograd")
def deltanet_core_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _deltanet_core_decay_reference(
        query, key, value, decay, beta, recurrent_state
    )


@impl(delta_net_lib, "deltanet_core.out", dispatch_key="CompositeExplicitAutograd")
def deltanet_core_out_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    *,
    core_attn_out: torch.Tensor,
    new_recurrent_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ref_out, ref_state = _deltanet_core_decay_reference(
        query, key, value, decay, beta, recurrent_state
    )
    core_attn_out.copy_(ref_out)
    new_recurrent_state.copy_(ref_state)
    return core_attn_out, new_recurrent_state


@impl(
    delta_net_lib,
    "deltanet_core_stream",
    dispatch_key="CompositeExplicitAutograd",
)
def deltanet_core_stream_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ref_out, ref_state = _deltanet_core_decay_reference(
        query,
        key,
        value,
        decay,
        beta,
        recurrent_state.transpose(-1, -2),
    )
    return ref_out, ref_state.transpose(-1, -2)


@impl(
    delta_net_lib,
    "deltanet_core_fp16_state",
    dispatch_key="CompositeExplicitAutograd",
)
def deltanet_core_fp16_state_impl(query, key, value, decay, beta, recurrent_state):
    if recurrent_state.dtype != torch.float16:
        raise ValueError("DeltaNetCoreFp16State requires a float16 recurrent state")
    query, key, value, decay, beta = (
        tensor.float() for tensor in (query, key, value, decay, beta)
    )
    state = recurrent_state.float()
    outputs = []
    for i in range(query.shape[1]):
        q, k, v = query[:, i], key[:, i], value[:, i]
        d, b = decay[:, i].unsqueeze(-1), beta[:, i].unsqueeze(-1)
        delta = (v - (state * k.unsqueeze(-1)).sum(-2) * d) * b
        outputs.append(
            (state * q.unsqueeze(-1)).sum(-2) * d
            + (q * k).sum(-1, keepdim=True) * delta
        )
        state = (state * d.unsqueeze(-1) + k.unsqueeze(-1) * delta.unsqueeze(-2))
        state = state.half().float()
    return torch.stack(outputs, dim=1), state.half()


@impl(
    delta_net_lib,
    "deltanet_core_fp16_state.out",
    dispatch_key="CompositeExplicitAutograd",
)
def deltanet_core_fp16_state_out_impl(
    query, key, value, decay, beta, recurrent_state, *,
    core_attn_out, new_recurrent_state,
):
    ref_out, ref_state = deltanet_core_fp16_state_impl(
        query, key, value, decay, beta, recurrent_state
    )
    core_attn_out.copy_(ref_out)
    new_recurrent_state.copy_(ref_state)
    return core_attn_out, new_recurrent_state


@impl(
    delta_net_lib, "deltanet_core_fp16_accum", dispatch_key="CompositeExplicitAutograd"
)
def deltanet_core_fp16_accum_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _deltanet_core_decay_reference(
        query, key, value, decay, beta, recurrent_state
    )


@impl(
    delta_net_lib,
    "deltanet_core_fp16_accum.out",
    dispatch_key="CompositeExplicitAutograd",
)
def deltanet_core_fp16_accum_out_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    *,
    core_attn_out: torch.Tensor,
    new_recurrent_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ref_out, ref_state = _deltanet_core_decay_reference(
        query, key, value, decay, beta, recurrent_state
    )
    core_attn_out.copy_(ref_out)
    new_recurrent_state.copy_(ref_state)
    return core_attn_out, new_recurrent_state


@impl(delta_net_lib, "deltanet_identity", dispatch_key="CompositeExplicitAutograd")
def deltanet_identity_impl(input: torch.Tensor) -> torch.Tensor:
    return input.clone()


@impl(delta_net_lib, "deltanet_identity.out", dispatch_key="CompositeExplicitAutograd")
def deltanet_identity_out_impl(
    input: torch.Tensor,
    *,
    output: torch.Tensor,
) -> torch.Tensor:
    output.copy_(input)
    return output


@impl(delta_net_lib, "deltanet_decay_state", dispatch_key="CompositeExplicitAutograd")
def deltanet_decay_state_impl(
    recurrent_state: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    return recurrent_state * g[:, 0].float().exp().unsqueeze(-1).unsqueeze(-1)


@impl(
    delta_net_lib, "deltanet_decay_state.out", dispatch_key="CompositeExplicitAutograd"
)
def deltanet_decay_state_out_impl(
    recurrent_state: torch.Tensor,
    g: torch.Tensor,
    *,
    output: torch.Tensor,
) -> torch.Tensor:
    output.copy_(deltanet_decay_state_impl(recurrent_state, g))
    return output


@impl(
    delta_net_lib,
    "deltanet_state_passthrough",
    dispatch_key="CompositeExplicitAutograd",
)
def deltanet_state_passthrough_impl(
    recurrent_state: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    return recurrent_state.clone()


@impl(
    delta_net_lib,
    "deltanet_state_passthrough.out",
    dispatch_key="CompositeExplicitAutograd",
)
def deltanet_state_passthrough_out_impl(
    recurrent_state: torch.Tensor,
    g: torch.Tensor,
    *,
    output: torch.Tensor,
) -> torch.Tensor:
    output.copy_(recurrent_state)
    return output


class BaselineDeltaNet(torch.nn.Module):
    def __init__(
        self,
        scaling: float = _DELTA_NET_SCALING,
        epsilon: float = _DELTA_NET_EPS,
        promote_float: bool = True,
    ):
        super().__init__()
        self.scaling = scaling
        self.epsilon = epsilon
        self.promote_float = promote_float

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _deltanet_reference(
            query,
            key,
            value,
            g,
            beta,
            recurrent_state,
            self.scaling,
            self.epsilon,
            self.promote_float,
        )


class CustomOpDeltaNet(BaselineDeltaNet):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.qnn_custom_ops.deltanet.default(
            query,
            key,
            value,
            g,
            beta,
            recurrent_state,
            self.scaling,
            self.epsilon,
        )


class CustomOpDeltaNetPacked(BaselineDeltaNet):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        packed = torch.ops.qnn_custom_ops.deltanet_packed.default(
            _l2norm(query, self.epsilon) * self.scaling,
            _l2norm(key, self.epsilon),
            value,
            g,
            beta,
            recurrent_state,
        )
        return _unpack_deltanet_outputs(
            packed,
            (query.shape[0], query.shape[1], query.shape[2], value.shape[3]),
            recurrent_state.shape,
        )


class CustomOpDeltaNetCore(BaselineDeltaNet):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.qnn_custom_ops.deltanet_core.default(
            _l2norm(query, self.epsilon) * self.scaling,
            _l2norm(key, self.epsilon),
            value,
            torch.exp(g),
            beta,
            recurrent_state,
        )


class CustomOpDeltaNetCoreOnly(torch.nn.Module):
    def forward(self, query, key, value, decay, beta, recurrent_state):
        return torch.ops.qnn_custom_ops.deltanet_core.default(
            query, key, value, decay, beta, recurrent_state
        )


class CustomOpDeltaNetCoreFp16State(torch.nn.Module):
    def forward(self, query, key, value, decay, beta, recurrent_state):
        return torch.ops.qnn_custom_ops.deltanet_core_fp16_state.default(
            query, key, value, decay, beta, recurrent_state
        )


class CustomOpDeltaNetCoreFp16Accum(BaselineDeltaNet):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.qnn_custom_ops.deltanet_core_fp16_accum.default(
            _l2norm(query, self.epsilon) * self.scaling,
            _l2norm(key, self.epsilon),
            value,
            torch.exp(g),
            beta,
            recurrent_state,
        )


class CustomOpDeltaNetCoreStream(BaselineDeltaNet):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.qnn_custom_ops.deltanet_core_stream.default(
            _l2norm(query, self.epsilon) * self.scaling,
            _l2norm(key, self.epsilon),
            value,
            torch.exp(g),
            beta,
            recurrent_state,
        )


class CustomOpDeltaNetIdentity(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.ops.qnn_custom_ops.deltanet_identity.default(value),
            torch.ops.qnn_custom_ops.deltanet_identity.default(recurrent_state),
        )


class CustomOpDeltaNetDecayState(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            value,
            torch.ops.qnn_custom_ops.deltanet_decay_state.default(
                recurrent_state,
                g,
            ),
        )


class CustomOpDeltaNetStatePassthrough(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            value,
            torch.ops.qnn_custom_ops.deltanet_state_passthrough.default(
                recurrent_state,
                g,
            ),
        )


def _run(cmd: Sequence[str], cwd: Optional[str] = None) -> None:
    subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, cwd=cwd, check=True)


def make_inputs(
    ar_len: int,
    seed: int,
    batch: int,
    heads: int,
    head_k_dim: int,
    head_v_dim: int,
    input_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    return (
        (torch.randn(batch, ar_len, heads, head_k_dim) * 0.1).to(input_dtype),
        (torch.randn(batch, ar_len, heads, head_k_dim) * 0.1).to(input_dtype),
        (torch.randn(batch, ar_len, heads, head_v_dim) * 0.1).to(input_dtype),
        (-torch.rand(batch, ar_len, heads) * 0.1).to(input_dtype),
        torch.rand(batch, ar_len, heads).to(input_dtype),
        (torch.randn(batch, heads, head_k_dim, head_v_dim) * 0.1).to(input_dtype),
    )


def parse_quant_dtypes(raw: str) -> List[Optional[QuantDtype]]:
    quant_dtypes = []
    for item in raw.split(","):
        name = item.strip()
        if not name:
            continue
        if name in {"none", "fp16", "float"}:
            quant_dtypes.append(None)
        else:
            quant_dtypes.append(getattr(QuantDtype, name))
    return quant_dtypes


def quant_name(quant_dtype: Optional[QuantDtype]) -> str:
    return "fp16" if quant_dtype is None else quant_dtype.name


def make_model(
    variant: str,
    scaling: float,
    epsilon: float,
    input_dtype: str = "fp32",
) -> torch.nn.Module:
    if variant == "baseline":
        return BaselineDeltaNet(
            scaling, epsilon, promote_float=input_dtype != "fp16"
        ).eval()
    if variant == "custom":
        return CustomOpDeltaNet(scaling, epsilon).eval()
    if variant == "packed":
        return CustomOpDeltaNetPacked(scaling, epsilon).eval()
    if variant == "core":
        return CustomOpDeltaNetCore(scaling, epsilon).eval()
    if variant == "core_only":
        return CustomOpDeltaNetCoreOnly().eval()
    if variant == "core_fp16accum":
        return CustomOpDeltaNetCoreFp16Accum(scaling, epsilon).eval()
    if variant == "core_stream":
        return CustomOpDeltaNetCoreStream(scaling, epsilon).eval()
    if variant == "probe_identity":
        return CustomOpDeltaNetIdentity().eval()
    if variant == "probe_decay":
        return CustomOpDeltaNetDecayState().eval()
    if variant == "probe_state_passthrough":
        return CustomOpDeltaNetStatePassthrough().eval()
    raise RuntimeError(f"Unknown variant: {variant}")


def quant_config_for_dtype(quant_dtype: QuantDtype, act_observer=None):
    config_fns = {
        QuantDtype.use_16a16w: get_16a16w_qnn_ptq_config,
        QuantDtype.use_16a8w: get_16a8w_qnn_ptq_config,
        QuantDtype.use_16a4w: get_16a4w_qnn_ptq_config,
        QuantDtype.use_16a4w_block: get_16a4w_qnn_ptq_config,
        QuantDtype.use_8a8w: get_8a8w_qnn_ptq_config,
        QuantDtype.use_8a4w: get_8a4w_qnn_ptq_config,
        QuantDtype.use_fp16a8w: get_fp16a8w_qnn_ptq_config,
    }
    if act_observer is not None:
        return config_fns[quant_dtype](act_observer=act_observer)
    return config_fns[quant_dtype]()


def deltanet_quant_annotations(
    quant_dtype: QuantDtype,
    *,
    act_observer=None,
    share_state_qparams=False,
    fp16_state=False,
    fp32_state=False,
):
    float_state = fp16_state or fp32_state
    core_quant_dtype = QuantDtype.use_fp16a8w if float_state else quant_dtype
    quant_cfg = quant_config_for_dtype(core_quant_dtype, act_observer=act_observer)
    custom_quant_annotator = CustomOpsQuantAnnotator()
    custom_quant_annotator.register_annotation(
        torch.ops.qnn_custom_ops.deltanet.default,
        IOQuantConfig(
            input_quant_specs={index: quant_cfg.input_activation for index in range(6)},
            output_quant_specs={
                0: quant_cfg.output_activation,
                1: quant_cfg.output_activation,
            },
        ),
    )
    custom_quant_annotator.register_annotation(
        torch.ops.qnn_custom_ops.deltanet_packed.default,
        IOQuantConfig(
            input_quant_specs={index: quant_cfg.input_activation for index in range(6)},
            output_quant_specs={0: quant_cfg.output_activation},
        ),
    )
    custom_quant_annotator.register_annotation(
        torch.ops.qnn_custom_ops.deltanet_core.default,
        IOQuantConfig(
            input_quant_specs={index: quant_cfg.input_activation for index in range(6)},
            output_quant_specs={
                0: quant_cfg.output_activation,
                1: quant_cfg.output_activation,
            },
        ),
    )
    custom_quant_annotator.register_annotation(
        torch.ops.qnn_custom_ops.deltanet_core_stream.default,
        IOQuantConfig(
            input_quant_specs={index: quant_cfg.input_activation for index in range(6)},
            output_quant_specs={
                0: quant_cfg.output_activation,
                1: quant_cfg.output_activation,
            },
        ),
    )
    custom_quant_annotator.register_annotation(
        torch.ops.qnn_custom_ops.deltanet_core_fp16_accum.default,
        IOQuantConfig(
            input_quant_specs={index: quant_cfg.input_activation for index in range(6)},
            output_quant_specs={
                0: quant_cfg.output_activation,
                1: quant_cfg.output_activation,
            },
        ),
    )
    custom_quant_annotator.register_annotation(
        torch.ops.qnn_custom_ops.deltanet_identity.default,
        IOQuantConfig(
            input_quant_specs={0: quant_cfg.input_activation},
            output_quant_specs={0: quant_cfg.output_activation},
        ),
    )
    custom_quant_annotator.register_annotation(
        torch.ops.qnn_custom_ops.deltanet_decay_state.default,
        IOQuantConfig(
            input_quant_specs={
                0: quant_cfg.input_activation,
                1: quant_cfg.input_activation,
            },
            output_quant_specs={0: quant_cfg.output_activation},
        ),
    )
    custom_quant_annotator.register_annotation(
        torch.ops.qnn_custom_ops.deltanet_state_passthrough.default,
        IOQuantConfig(
            input_quant_specs={
                0: quant_cfg.input_activation,
                1: quant_cfg.input_activation,
            },
            output_quant_specs={0: quant_cfg.output_activation},
        ),
    )
    annotate_fn = custom_quant_annotator.build_annotation_fn()
    if float_state:
        from torchao.quantization.pt2e.quantizer import QuantizationAnnotation

        cache_cfg = quant_config_for_dtype(quant_dtype, act_observer=act_observer)

        def annotate_conv_state_cat(gm):
            # The causal-conv history is a graph IO tensor.  Annotate the cat
            # edge explicitly so PT2E creates the cache encoding and its
            # calibration qparams; the opaque recurrence core must stay FP16.
            annotated = 0
            cache_spec = cache_cfg.input_activation

            # The cache placeholder is sliced before it reaches cat.  PT2E
            # attaches an input QSpec to the consuming node, so annotating only
            # cat is too late: the placeholder can otherwise remain a raw
            # integer graph input with no quantize node to inspect later.
            for placeholder in gm.graph.nodes:
                if placeholder.op != "placeholder":
                    continue
                value = placeholder.meta.get("val")
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != (
                    1,
                    6144,
                    4,
                ):
                    continue
                for user in tuple(placeholder.users):
                    if user.op != "call_function" or "slice" not in str(user.target):
                        continue
                    user.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                        input_qspec_map={placeholder: cache_spec},
                        output_qspec=cache_spec,
                        _annotated=True,
                    )
                    annotated += 1

            for node in gm.graph.nodes:
                if node.target != torch.ops.aten.cat.default:
                    continue
                inputs = node.args[0] if node.args else ()
                if not isinstance(inputs, (tuple, list)):
                    continue
                cache_inputs = [
                    value
                    for value in inputs
                    if isinstance(value, torch.fx.Node)
                    and tuple(value.meta.get("val", torch.empty(0)).shape)
                    == (1, 6144, 4)
                ]
                if not cache_inputs:
                    continue
                cache_input = cache_inputs[0]
                input_qspec_map = {cache_input: cache_spec}
                shared = SharedQuantizationSpec((cache_input, node))
                for value in inputs:
                    if value is not cache_input and isinstance(value, torch.fx.Node):
                        input_qspec_map[value] = shared
                node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                    input_qspec_map=input_qspec_map,
                    output_qspec=shared,
                    _annotated=True,
                )
                annotated += 1

            # The updated causal-conv history is returned through the model
            # output tuple.  Annotating the cat alone does not claim the final
            # output edge when the graph contains a trailing cast/view, so the
            # exported output can remain FLOAT while its matching input is
            # UINT16.  Claim each cache output explicitly to keep the IO pair
            # in one encoding.
            output_annotated = 0
            for node in gm.graph.nodes:
                if node.op != "output" or not node.args:
                    continue
                outputs = node.args[0]
                if not isinstance(outputs, (tuple, list)):
                    outputs = (outputs,)
                for value in outputs:
                    if not isinstance(value, torch.fx.Node):
                        continue
                    tensor = value.meta.get("val")
                    if (
                        not isinstance(tensor, torch.Tensor)
                        or tuple(tensor.shape)
                        != (
                            1,
                            6144,
                            4,
                        )
                        or tensor.dtype != torch.float32
                    ):
                        continue
                    value.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                        input_qspec_map={},
                        output_qspec=cache_spec,
                        _annotated=True,
                    )
                    output_annotated += 1
            logging.info(
                "Annotated %d GDN causal-conv cache cats and %d outputs",
                annotated,
                output_annotated,
            )

        def annotate_float_core_region(gm):
            conv_targets = {
                torch.ops.aten.conv2d.default,
                torch.ops.aten.convolution.default,
            }

            def add_nodes(value, work):
                if isinstance(value, torch.fx.Node):
                    work.append(value)
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        add_nodes(item, work)
                elif isinstance(value, dict):
                    for item in value.values():
                        add_nodes(item, work)

            # Keep projection convolutions quantized, but remove activation QDQ
            # from the tensor path feeding the recurrent core and out_proj.
            for node in gm.graph.nodes:
                if node.target != torch.ops.qnn_custom_ops.deltanet_core.default:
                    continue
                attention_users = [
                    user
                    for user in node.users
                    if len(user.args) > 1 and user.args[1] == 0
                ]
                work = [node, *attention_users]
                seen = set()
                while work:
                    current = work.pop()
                    if current in seen or current.op in {"placeholder", "output"}:
                        continue
                    seen.add(current)
                    if current.target in conv_targets:
                        continue
                    current.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                        input_qspec_map={}, output_qspec=None, _annotated=True
                    )
                    add_nodes(current.args, work)
                    work.extend(
                        user
                        for user in current.users
                        if user.target not in conv_targets
                        and not (
                            current is node and len(user.args) > 1 and user.args[1] == 1
                        )
                    )

        # Keep this cache annotation last.  The float-core pass walks backward
        # through the graph and may otherwise replace the cache edge annotation
        # with an empty annotation while removing activation QDQ from the core.
        return (annotate_fn, annotate_float_core_region, annotate_conv_state_cat)

    if not share_state_qparams:
        return (annotate_fn,)

    def annotate_shared_state(gm):
        annotate_fn(gm)
        stateful_core_targets = {
            torch.ops.qnn_custom_ops.deltanet_core.default,
            torch.ops.qnn_custom_ops.deltanet_core_stream.default,
            torch.ops.qnn_custom_ops.deltanet_core_fp16_accum.default,
        }
        for node in gm.graph.nodes:
            if node.target not in stateful_core_targets:
                continue
            # Recurrent state must use one encoding at the graph boundary and
            # after the core updates it in place.
            for user in node.users:
                if user.args[1] == 1:
                    user.meta[Q_ANNOTATION_KEY].output_qspec = SharedQuantizationSpec(
                        (node.args[5], node)
                    )

    return (annotate_shared_state,)


def deltanet_fp16_state_quant_annotations(
    quant_dtype=QuantDtype.use_16a4w, *, act_observer=None
):
    if quant_dtype not in {
        QuantDtype.use_16a4w,
        QuantDtype.use_16a4w_block,
        QuantDtype.use_16a8w,
        QuantDtype.use_16a16w,
    }:
        raise ValueError("DeltaNetCoreFp16State requires U16 activation quantization")
    config = quant_config_for_dtype(quant_dtype, act_observer=act_observer)
    annotator = CustomOpsQuantAnnotator()
    annotator.register_annotation(
        torch.ops.qnn_custom_ops.deltanet_core_fp16_state.default,
        IOQuantConfig(
            input_quant_specs={index: config.input_activation for index in range(5)},
            output_quant_specs={0: config.output_activation, 1: None},
        ),
    )
    return (annotator.build_annotation_fn(),)


def make_deltanet_quantizer(args, quant_dtype: Optional[QuantDtype]):
    if quant_dtype is None:
        return None

    return make_quantizer(
        quant_dtype=quant_dtype,
        custom_annotations=deltanet_quant_annotations(quant_dtype),
        backend=get_backend_type(args.backend),
        soc_model=args.soc_model,
    )


def setup_op_package(
    args, *, state_vtcm: bool = False, stream_recurrence: bool = False
):
    op_package_dir = Path(args.op_package_dir).resolve()
    op_package_name = "DeltaNetOpPackage"
    interface_provider = f"{op_package_name}InterfaceProvider"
    lib_name = f"libQnn{op_package_name}"
    soc_info = _soc_info_table[getattr(QcomChipset, args.soc_model)]
    arch = soc_info.htp_info.htp_arch

    if args.build_op_package:
        if "HEXAGON_SDK_ROOT" not in os.environ:
            raise RuntimeError("Environment variable HEXAGON_SDK_ROOT must be set")
        if "ANDROID_NDK_ROOT" not in os.environ:
            raise RuntimeError("Environment variable ANDROID_NDK_ROOT must be set")

        build_dir = op_package_dir / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        _run(
            ["make", "htp_x86", "htp_aarch64", f"htp_v{arch}"], cwd=str(op_package_dir)
        )
        shutil.copyfile(
            op_package_dir / "build" / f"hexagon-v{arch}" / f"{lib_name}.so",
            op_package_dir / "build" / f"hexagon-v{arch}" / f"{lib_name}_HTP.so",
        )

    workspace = args.workspace
    custom_op_map = [
        (
            str(torch.ops.qnn_custom_ops.deltanet_core_fp16_state.default),
            "DeltaNetCoreFp16State",
        ),
        (
            str(torch.ops.qnn_custom_ops.deltanet_core.default),
            "DeltaNetCoreVtcm" if state_vtcm else "DeltaNetCore",
        ),
        (
            str(torch.ops.qnn_custom_ops.deltanet_core_stream.default),
            "DeltaNetCoreStream",
        ),
        (
            str(torch.ops.qnn_custom_ops.deltanet_core_fp16_accum.default),
            "DeltaNetCoreFp16Accum",
        ),
    ]
    op_package_infos = []
    for custom_op_name, qnn_op_type_name in custom_op_map:
        op_package_infos.extend(
            [
                QnnExecuTorchOpPackageInfo(
                    op_package_name=op_package_name,
                    op_package_path=f"{workspace}/{lib_name}_HTP.so",
                    interface_provider=interface_provider,
                    target=QnnExecuTorchOpPackageTarget.HTP,
                    custom_op_name=custom_op_name,
                    qnn_op_type_name=qnn_op_type_name,
                    platform=QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID,
                ),
                QnnExecuTorchOpPackageInfo(
                    op_package_name=op_package_name,
                    op_package_path=f"{workspace}/{lib_name}.so",
                    interface_provider=interface_provider,
                    target=QnnExecuTorchOpPackageTarget.CPU,
                    custom_op_name=custom_op_name,
                    qnn_op_type_name=qnn_op_type_name,
                    platform=QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID,
                ),
                QnnExecuTorchOpPackageInfo(
                    op_package_name=op_package_name,
                    op_package_path=str(
                        op_package_dir
                        / "build"
                        / "x86_64-linux-clang"
                        / f"{lib_name}.so"
                    ),
                    interface_provider=interface_provider,
                    target=QnnExecuTorchOpPackageTarget.CPU,
                    custom_op_name=custom_op_name,
                    qnn_op_type_name=qnn_op_type_name,
                    platform=QnnExecuTorchOpPackagePlatform.X86_64,
                ),
            ]
        )
    op_package_options = QnnExecuTorchOpPackageOptions(
        op_package_infos=op_package_infos
    )
    op_package_paths = [
        str(op_package_dir / "build" / f"hexagon-v{arch}" / f"{lib_name}_HTP.so"),
        str(op_package_dir / "build" / "aarch64-android" / f"{lib_name}.so"),
    ]
    return op_package_options, op_package_paths


def parse_runner_avg_ms(output: str) -> Optional[float]:
    matches = re.findall(
        r"(?<!Total\s)\d+\s+inferences?\s+took\s+[0-9.]+\s+ms,\s+avg\s+([0-9.]+)\s+ms",
        output,
    )
    if not matches:
        matches = re.findall(
            r"Total\s+\d+\s+inferences?\s+took\s+[0-9.]+\s+ms,\s+avg\s+([0-9.]+)\s+ms",
            output,
        )
    return None if not matches else float(matches[-1])


def tensor_sqnr(ref: torch.Tensor, actual: torch.Tensor) -> float:
    ref = ref.detach().cpu().float()
    actual = actual.detach().cpu().float()
    noise = torch.sum((ref - actual) ** 2)
    if noise == 0:
        return float("inf")
    signal = torch.sum(ref**2)
    return float(10 * torch.log10(signal / noise))


def read_outputs(
    output_dir: Path,
    refs: Sequence[torch.Tensor],
) -> List[Dict[str, object]]:
    output_info = []
    output_files = sorted(output_dir.glob("*.raw"))
    if len(output_files) != len(refs):
        raise RuntimeError(
            f"Expected {len(refs)} outputs in {output_dir}, got {len(output_files)}"
        )
    for output_file, ref in zip(output_files, refs):
        expected_numel = ref.numel()
        raw_size = output_file.stat().st_size
        dtype = None
        for candidate in (np.float32, np.float16, np.uint16, np.uint8):
            if raw_size == expected_numel * np.dtype(candidate).itemsize:
                dtype = candidate
                break
        if dtype is None:
            output_info.append(
                {
                    "file": str(output_file),
                    "error": f"unexpected byte size {raw_size}",
                }
            )
            continue

        data = np.fromfile(output_file, dtype=dtype).reshape(tuple(ref.shape))
        actual = torch.from_numpy(data.astype(np.float32))
        ref_float = ref.detach().cpu().float()
        abs_diff = (actual - ref_float).abs()
        output_info.append(
            {
                "file": str(output_file),
                "dtype": np.dtype(dtype).name,
                "shape": list(ref.shape),
                "sqnr_db": tensor_sqnr(ref_float, actual),
                "max_abs_diff": float(abs_diff.max()),
                "mean_abs_diff": float(abs_diff.mean()),
            }
        )
    return output_info


def run_one(
    args,
    qnn_config: QnnConfig,
    variant: str,
    quant_dtype: Optional[QuantDtype],
    dataset: List[Tuple[torch.Tensor, ...]],
    op_package_options,
    op_package_paths: Sequence[str],
) -> Dict[str, object]:
    qname = quant_name(quant_dtype)
    model = make_model(variant, args.scaling, args.epsilon, args.input_dtype)
    if variant == "core_only":
        dataset = [
            (
                _l2norm(inputs[0].float(), args.epsilon)
                .mul(args.scaling)
                .to(inputs[0].dtype),
                _l2norm(inputs[1].float(), args.epsilon).to(inputs[1].dtype),
                inputs[2],
                inputs[3].float().exp().to(inputs[3].dtype),
                *inputs[4:],
            )
            for inputs in dataset
        ]
    if variant == "core_stream":
        dataset = [(*inputs[:5], inputs[5].transpose(-1, -2)) for inputs in dataset]
    sample_input = dataset[0]
    with torch.no_grad():
        refs = model(*sample_input)

    input_dtype_tag = "" if args.input_dtype == "fp32" else f"_in{args.input_dtype}"
    case_name = (
        f"deltanet_{variant}_{qname}{input_dtype_tag}_b{args.batch}_t{args.ar_len}_"
        f"h{args.heads}_k{args.head_k_dim}_v{args.head_v_dim}"
    )
    case_dir = Path(args.artifact) / case_name
    if case_dir.exists() and not qnn_config.pre_gen_pte:
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    pte_file = case_dir / case_name
    logging.info("Building %s quant=%s", variant, qname)
    uses_op_package = variant in {
        "core",
        "core_only",
        "core_stream",
        "core_fp16accum",
    }
    try:
        build_executorch_binary(
            model=model,
            qnn_config=qnn_config,
            file_name=str(pte_file),
            dataset=dataset,
            quant_dtype=quant_dtype,
            custom_quantizer=(
                make_deltanet_quantizer(args, quant_dtype) if uses_op_package else None
            ),
            op_package_options=op_package_options if uses_op_package else None,
        )
    except SystemExit:
        if not args.compile_only:
            raise

    result = {
        "case": case_name,
        "variant": variant,
        "state_vtcm": args.state_vtcm,
        "quant_dtype": qname,
        "batch": args.batch,
        "ar_len": args.ar_len,
        "heads": args.heads,
        "head_k_dim": args.head_k_dim,
        "head_v_dim": args.head_v_dim,
        "input_dtype": args.input_dtype,
        "warm_up": args.warm_up,
    }
    if args.compile_only:
        return result

    pte_path = f"{pte_file}.pte"
    adb = SimpleADB(
        qnn_config=qnn_config,
        pte_path=pte_path,
        workspace=args.workspace,
    )

    runner_output = {"text": ""}

    def capture_output(proc):
        runner_output["text"] = proc.stdout

    adb.push(
        inputs=[sample_input],
        files=op_package_paths if uses_op_package else None,
    )
    adb.extra_cmds += f" --warm_up {args.warm_up}"
    adb.execute(output_callback=capture_output, iteration=args.iteration)
    (case_dir / "runner.log").write_text(runner_output["text"])
    adb.pull(host_output_path=str(case_dir))

    result["runner_avg_ms"] = parse_runner_avg_ms(runner_output["text"])
    result["outputs"] = read_outputs(case_dir / "outputs", refs)
    return result


def main(args) -> None:
    variants = [variant.strip() for variant in args.variants.split(",") if variant]
    supported_variants = {
        "baseline",
        "core",
        "core_only",
        "core_stream",
        "core_fp16accum",
    }
    if not variants or set(variants) - supported_variants:
        raise ValueError(
            "--variants supports baseline,core,core_only,core_stream,core_fp16accum; "
            "the legacy DeltaNet.cpp kernels have been removed"
        )
    qnn_config = QnnConfig.load_config(args.config_file if args.config_file else args)
    if qnn_config.enable_x86_64:
        raise RuntimeError("DeltaNet benchmark currently expects an Android device")

    if (
        os.path.exists(args.artifact)
        and not args.keep_artifact
        and not args.pre_gen_pte
    ):
        shutil.rmtree(args.artifact)
    os.makedirs(args.artifact, exist_ok=True)

    if args.state_vtcm and (
        args.variants not in {"core", "core_only"}
        or any(
            dtype is None or not dtype.name.startswith("use_16a")
            for dtype in parse_quant_dtypes(args.quant_dtypes)
        )
    ):
        raise ValueError("--state_vtcm requires core/core_only and 16-bit activations")
    op_package_options, op_package_paths = setup_op_package(
        args,
        state_vtcm=args.state_vtcm,
        stream_recurrence=args.stream_recurrence,
    )
    input_dtype = torch.float16 if args.input_dtype == "fp16" else torch.float32
    dataset = [
        make_inputs(
            args.ar_len,
            args.seed + i,
            args.batch,
            args.heads,
            args.head_k_dim,
            args.head_v_dim,
            input_dtype,
        )
        for i in range(args.calibration_samples)
    ]

    quant_dtypes = parse_quant_dtypes(args.quant_dtypes)
    results = []
    for quant_dtype in quant_dtypes:
        for variant in variants:
            results.append(
                run_one(
                    args,
                    qnn_config,
                    variant,
                    quant_dtype,
                    dataset,
                    op_package_options,
                    op_package_paths,
                )
            )

    summary_path = Path(args.artifact) / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    logging.info("Saved summary to %s", summary_path)

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps({"results": results}))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = setup_common_args_and_variables()
    default_op_package_dir = (
        Path(__file__).parent / "deltanet_op_package_htp" / "DeltaNetOpPackage"
    )
    parser.add_argument(
        "-a",
        "--artifact",
        default="./deltanet-qnn-benchmark",
        type=str,
        help="Folder for generated DeltaNet QNN benchmark artifacts.",
    )
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--ar_len", default=1, type=int)
    parser.add_argument("--heads", default=16, type=int)
    parser.add_argument("--head_k_dim", default=128, type=int)
    parser.add_argument("--head_v_dim", default=128, type=int)
    parser.add_argument(
        "--input_dtype",
        choices=("fp32", "fp16"),
        default="fp32",
        help="Floating-point dtype for generated inputs when quant_dtype is fp16/none.",
    )
    parser.add_argument("--scaling", default=_DELTA_NET_SCALING, type=float)
    parser.add_argument("--epsilon", default=_DELTA_NET_EPS, type=float)
    parser.add_argument("--iteration", default=50, type=int)
    parser.add_argument("--warm_up", default=5, type=int)
    parser.add_argument("--calibration_samples", default=4, type=int)
    parser.add_argument(
        "--variants",
        default="baseline,core",
        type=str,
        help="Comma-separated variants: baseline,core,core_only,core_stream,core_fp16accum.",
    )
    parser.add_argument(
        "--quant_dtypes",
        default="fp16,use_16a8w,use_16a4w",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--op_package_dir",
        default=str(default_op_package_dir),
        type=str,
    )
    parser.add_argument(
        "--workspace",
        default="/data/local/tmp/executorch/deltanet_benchmark",
        type=str,
    )
    parser.add_argument("--build_op_package", action="store_true")
    parser.add_argument(
        "--stream_recurrence",
        action="store_true",
        help="Register the one-token, one-state-column recurrence op.",
    )
    parser.add_argument(
        "--state_vtcm",
        action="store_true",
        help="Keep the QUInt16 DeltaNetCore state in VTCM across all tokens.",
    )
    parser.add_argument("--keep_artifact", action="store_true")

    args = parser.parse_args()
    if args.seed is None:
        args.seed = 666

    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
