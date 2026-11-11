# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
import torchao.quantization.pt2e.quantize_pt2e  # noqa: F401
from executorch.examples.qualcomm.custom_op.attention_support.graph import (
    replace_attention,
    replace_attention_layers,
)

import executorch.examples.qualcomm.custom_op.deltanet  # noqa: F401


def qdq(value, scale=0.001, zero=32768, bits=16):
    dtype = torch.uint8 if bits == 8 else torch.int32
    encoded = torch.ops.quantized_decomposed.quantize_per_tensor.default(
        value, scale, zero, 0, 2**bits - 1, dtype
    )
    return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        encoded, scale, zero, 0, 2**bits - 1, dtype
    )


class CalibratedAttention(torch.nn.Module):
    def __init__(self, cache_bits=16):
        super().__init__()
        self.cache_bits = cache_bits

    def forward(self, hidden, mask, *caches):
        layers = len(caches) // 2
        keys, values = [], []
        scale, zero = (0.001, 32768) if self.cache_bits == 16 else (0.01, 128)
        for layer in range(layers):
            query = qdq(hidden)
            key = qdq(hidden[:, :2].transpose(-1, -2), scale, zero, self.cache_bits)
            value = qdq(hidden[:, :2], scale, zero, self.cache_bits)
            keys.append(key)
            values.append(value)
            history_k = qdq(caches[layer], scale, zero, self.cache_bits)
            history_v = qdq(caches[layers + layer], scale, zero, self.cache_bits)
            k = qdq(torch.cat((history_k, key), -1), scale, zero, self.cache_bits)
            v = qdq(torch.cat((history_v, value), -2), scale, zero, self.cache_bits)
            k = qdq(k[:, :, None].expand(-1, -1, 4, -1, -1).flatten(1, 2))
            v = qdq(v[:, :, None].expand(-1, -1, 4, -1, -1).flatten(1, 2))
            logits = qdq(qdq(query @ k) / (hidden.shape[-1] ** 0.5))
            probability = qdq(torch.softmax(qdq(logits + mask), -1), 1 / 65535, 0)
            hidden = hidden + qdq(probability @ v)
        return hidden, *keys, *values


@pytest.mark.parametrize("queries", [1, 128])
@pytest.mark.parametrize("layers", [1, 6])
@pytest.mark.parametrize("cache_bits", [8, 16])
def test_replace_calibrated_layers(queries, layers, cache_bits):
    inputs = (
        torch.randn(1, 8, queries, 32) * 0.01,
        torch.zeros(1, 1, queries, 256 + queries),
        *(torch.zeros(1, 2, 32, 256) for _ in range(layers)),
        *(torch.zeros(1, 2, 256, 32) for _ in range(layers)),
    )
    module = torch.export.export(CalibratedAttention(cache_bits), inputs).module()
    original = module(*inputs)
    if layers == 1:
        module, values, record = replace_attention(module, inputs, 256)
        records = [record]
    else:
        module, values, records = replace_attention_layers(module, inputs, 256)
        assert [r["key_input_index"] for r in records] == list(range(2, 2 + layers))
        assert [r["value_input_index"] for r in records] == list(
            range(2 + layers, 2 + 2 * layers)
        )
    assert len(records) == layers
    assert all(r["heads"] == 2 and r["dimension"] == 32 for r in records)
    assert all(r["encodings"][0]["elements"] == 2 * 32 * 256 for r in records)
    assert values[1].flatten().tolist() == [queries, 0, 0]
    assert all(v.shape == (1, 1, 1, 3) and v.dtype == torch.int32 for v in values[1:])
    program = torch.export.export(module, values, strict=False)
    cores = [
        n
        for n in program.graph.nodes
        if n.target == torch.ops.qnn_custom_ops.attention_valid.default
    ]
    assert len(cores) == layers
    assert not any(
        n.target in (torch.ops.aten.matmul.default, torch.ops.aten.softmax.int)
        for n in program.graph.nodes
    )
    for core in cores:
        assert core.args[0].meta["val"].shape == inputs[0].shape
    output = next(n for n in module.graph.nodes if n.op == "output")
    output.args = ((output.args[0][1], output.args[0][1 + layers]),)
    module.graph.set_codegen(torch.fx.graph.CodeGen())
    module._forward_pre_hooks.clear()
    module.graph.eliminate_dead_code()
    module.recompile()
    for actual, expected in zip(module(*values), (original[1], original[1 + layers])):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("position_capacity", [1024, 1128])
def test_generation_export_separates_kv_and_position_capacity(position_capacity):
    from executorch.examples.qualcomm.oss_scripts.llama.quantize.generation import (
        remove_export_checks,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.wrappers.base_component import (
        Mode,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.wrappers.llm_wrappers import (
        Qwen35TextDecoder,
    )

    class Hybrid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = CalibratedAttention(cache_bits=8)
            self.register_buffer("positions", torch.zeros(position_capacity, 256))
            self.register_buffer("output_weight", torch.ones(32, 256, 1, 1) / 256)

        def forward(self, hidden, mask, position, key, value, state):
            hidden = hidden + self.positions[position[0]]
            length = hidden.shape[1]
            attended, key_out, value_out = self.attention(
                hidden.reshape(1, length, 8, 32).transpose(1, 2), mask, key, value
            )
            query = hidden[:, :, None, :32]
            gate = torch.sigmoid(hidden[:, :, :1])
            recurrent, updated = torch.ops.qnn_custom_ops.deltanet_core.default(
                query, query, hidden[:, :, None, :64], gate, gate, state
            )
            hidden = attended.transpose(1, 2).reshape(1, length, 256)
            hidden = hidden + recurrent.squeeze(2).repeat(1, 1, 4)
            logits = (
                torch.nn.functional.conv2d(
                    qdq(hidden.transpose(1, 2).unsqueeze(2)), self.output_weight
                )
                .squeeze(2)
                .transpose(1, 2)
            )
            return logits, key_out, value_out, updated

    wrapper = Qwen35TextDecoder.__new__(Qwen35TextDecoder)
    wrapper.control_args = SimpleNamespace(model_mode="hybrid", max_seq_len=1000)
    wrapper.fused_gdn, wrapper.online_attention = False, True
    module = Hybrid()
    models = []
    for ar, mode in ((1, Mode.DECODE), (128, Mode.PREFILL)):
        inputs = (
            torch.zeros(1, ar, 256),
            torch.zeros(1, 1, ar, position_capacity),
            torch.arange(ar, dtype=torch.int32).reshape(1, ar),
            torch.zeros(1, 2, 32, position_capacity - ar),
            torch.zeros(1, 2, position_capacity - ar, 32),
            torch.zeros(1, 1, 32, 64),
        )
        graph = torch.export.export(module, inputs).module()
        remove_export_checks(graph)
        models.append(
            SimpleNamespace(
                decoder=graph,
                export_input=inputs,
                meta={"get_max_context_len": position_capacity},
                mode=mode,
            )
        )
    wrapper.decode, wrapper.prefill = models
    base = SimpleNamespace(
        meta={"get_max_context_len": position_capacity},
        model_args=SimpleNamespace(layer_types=["full_attention", "linear_attention"]),
        tok_embedding=torch.nn.Embedding(4, 256),
        tok_embedding_export_input=(torch.zeros(1, 128, dtype=torch.int64),),
    )
    wrapper._prepare_generation_graphs(base, wrapper.prefill.export_input)
    assert wrapper.decode.meta["get_prefill_position_capacity"] == position_capacity
    assert wrapper.decode.meta["get_prefill_max_past"] == position_capacity - 128
    for model in models:
        assert model.meta["get_max_context_len"] == 1000
        assert model.meta["get_attention_cache_capacity"] == 1000
        assert (
            sum(
                n.target == torch.ops.qnn_custom_ops.attention_valid.default
                for n in model.decoder.graph.nodes
            )
            == 1
        )
    assert wrapper.attention_cache["DECODE"] == wrapper.attention_cache["PREFILL"]
    assert base.meta["get_max_context_len"] == position_capacity
