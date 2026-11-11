# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
import torchao.quantization.pt2e.quantize_pt2e  # noqa: F401
import torchao.quantization.quant_primitives  # noqa: F401
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)

from executorch.examples.qualcomm.oss_scripts.llama.model.qwen35_output import (
    split_qwen35_output,
    use_split_qwen35_output,
)


def scoped_inputs():
    control = SimpleNamespace(
        decoder_model="qwen3_5-0_8b",
        model_mode="hybrid",
        max_seq_len=8192,
        max_context_len=8320,
        prefill_ar_len=128,
    )
    data = SimpleNamespace(
        soc_model="SM8850",
        backend=QnnExecuTorchBackendType.kHtpBackend,
        skip_quantize=False,
    )
    config = SimpleNamespace(num_sharding=1)
    decode = SimpleNamespace(
        decoder=object(),
        meta={
            "get_ar_len": 1,
            "get_max_seq_len": 8320,
            "get_max_context_len": 8320,
            "get_dim": 1024,
            "get_vocab_size": 248320,
            "get_kv_io_bit_width": 8,
        },
    )
    prefill = SimpleNamespace(decoder=object(), meta={"get_ar_len": 128})
    return control, data, config, decode, prefill


def test_exact_scope_uses_logical_context_control_argument():
    values = scoped_inputs()
    assert values[3].meta["get_max_seq_len"] == 8320
    assert use_split_qwen35_output(*values)


def test_online_attention_keeps_output_split():
    values = scoped_inputs()
    values[3].meta.update(
        get_attention_valid=True,
        get_attention_cache_capacity=8192,
        get_max_context_len=8192,
        get_max_seq_len=8192,
    )
    assert use_split_qwen35_output(*values)
    values[3].meta["get_attention_cache_capacity"] = 4096
    assert not use_split_qwen35_output(*values)


@pytest.mark.parametrize(
    "index,changes",
    [
        (0, {"decoder_model": "qwen3_5-2b"}),
        (0, {"decoder_model": "lincal3_5-0_8b"}),
        (0, {"model_mode": "kv"}),
        (0, {"max_seq_len": 4096}),
        (0, {"max_seq_len": 8320}),
        (0, {"max_context_len": 8192}),
        (0, {"prefill_ar_len": 64}),
        (1, {"soc_model": "SM8750"}),
        (1, {"backend": QnnExecuTorchBackendType.kGpuBackend}),
        (1, {"skip_quantize": True}),
        (2, {"num_sharding": 2}),
        (3, {"decoder": None}),
        (4, {"decoder": None}),
    ],
)
def test_scope_rejects_unmeasured_configuration(index, changes):
    values = scoped_inputs()
    values[index].__dict__.update(changes)
    assert not use_split_qwen35_output(*values)


@pytest.mark.parametrize(
    "key,value",
    [
        ("get_kv_io_bit_width", 16),
        ("get_dim", 2048),
        ("get_vocab_size", 1000),
        ("get_max_context_len", 8192),
        ("get_ar_len", 2),
    ],
)
def test_scope_checks_actual_decoder_metadata(key, value):
    values = scoped_inputs()
    values[3].meta[key] = value
    assert not use_split_qwen35_output(*values)


class FrozenProjection(torch.nn.Module):
    def __init__(self, rows=83, zero_point=True):
        super().__init__()
        generator = torch.Generator().manual_seed(42)
        self.register_buffer(
            "codes",
            torch.randint(
                -7, 8, (rows, 32, 1, 1), generator=generator, dtype=torch.int8
            ),
        )
        self.register_buffer(
            "scales", torch.rand(rows, 1, 1, 1, generator=generator) / 10 + 0.001
        )
        self.register_buffer(
            "zero", torch.zeros_like(self.scales) if zero_point else None
        )

    def forward(self, hidden, cache):
        value = torch.relu(hidden)[:, -1:, :].transpose(1, 2).unsqueeze(2)
        encoded = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            value, 0.01, 32768, 0, 65535, torch.int32
        )
        value = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            encoded, 0.01, 32768, 0, 65535, torch.int32
        )
        weight = torch.ops.torchao.dequantize_affine.default(
            self.codes, [1, 32, 1, 1], self.scales, self.zero, torch.int8, -7, 7
        )
        output = F.conv2d(value, weight)
        encoded = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            output, 0.02, 32768, 0, 65535, torch.int32
        )
        output = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            encoded, 0.02, 32768, 0, 65535, torch.int32
        )
        return output.squeeze(2).transpose(1, 2), cache + 1


def frozen_projection(tokens=1, rows=83, zero_point=True):
    inputs = (torch.randn(1, tokens, 32), torch.randn(1, 2, 2))
    module = torch.export.export(
        FrozenProjection(rows, zero_point).eval(), inputs, strict=True
    ).module()
    return module, inputs


@pytest.mark.parametrize("tokens", [1, 128])
@pytest.mark.parametrize("zero_point", [False, True])
def test_split_preserves_frozen_codes_qparams_external_qdq_and_cache(
    tokens, zero_point
):
    original, inputs = frozen_projection(tokens, zero_point=zero_point)
    candidate = copy.deepcopy(original)
    head = next(
        node
        for node in candidate.graph.nodes
        if node.target == torch.ops.aten.conv2d.default
    )
    original_q = next(iter(head.users))
    original_dq = next(iter(original_q.users))
    grid = original_q.args[1:]
    before = {name: tensor.clone() for name, tensor in candidate.named_buffers()}
    assert split_qwen35_output(candidate, 32, 83, 32) == 3
    assert original_q in candidate.graph.nodes and original_dq in candidate.graph.nodes
    assert original_q.args[1:] == grid
    cat = original_q.args[0]
    assert cat.target == torch.ops.aten.cat.default and cat.args[1] == 1
    assert [chunk.args[0].args[0].meta["val"].shape[1] for chunk in cat.args[0]] == [
        32,
        32,
        19,
    ]
    for label, name in (("codes", "codes"), ("scales", "scales"), ("zero", "zero")):
        if name not in before:
            continue
        combined = torch.cat(
            [
                candidate.get_buffer(f"_lm_head_split_{label}_{index}")
                for index in range(3)
            ]
        )
        assert torch.equal(combined, before[name])
    for chunk in cat.args[0]:
        assert chunk.args[1:] == original_dq.args[1:]
        assert chunk.args[0].args[1:] == grid
    for seed in (7, 42, 81):
        generator = torch.Generator().manual_seed(seed)
        hidden = torch.randn(inputs[0].shape, generator=generator)
        with torch.inference_mode():
            expected = original(hidden, inputs[1])
            actual = candidate(hidden, inputs[1])
        assert torch.equal(expected[0], actual[0])
        assert torch.equal(expected[1], actual[1])
    reexported = torch.export.export(candidate, inputs, strict=True).module()
    with torch.inference_mode():
        assert all(
            torch.equal(a, b) for a, b in zip(candidate(*inputs), reexported(*inputs))
        )


def test_actual_vocab_produces_61_chunks_with_final_2560_rows():
    module, inputs = frozen_projection(rows=248320)
    expected = module(*inputs)
    assert split_qwen35_output(module, expected_columns=32) == 61
    assert module.get_buffer("_lm_head_split_codes_60").shape == (2560, 32, 1, 1)
    actual = module(*inputs)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))


@pytest.mark.parametrize(
    "kind",
    [
        "raw_consumer",
        "different_dq_grid",
        "different_weight_encoding",
        "bias",
        "wrong_shape",
        "nonpositive_chunk",
        "repeated_transform",
    ],
)
def test_split_rejects_unsafe_graphs(kind):
    module, _ = frozen_projection()
    head = next(
        node
        for node in module.graph.nodes
        if node.target == torch.ops.aten.conv2d.default
    )
    quantizer = next(iter(head.users))
    dequantizer = next(iter(quantizer.users))
    if kind == "raw_consumer":
        with module.graph.inserting_after(head):
            module.graph.call_function(torch.ops.aten.clone.default, (head,))
    elif kind == "different_dq_grid":
        dequantizer.args = (dequantizer.args[0], 0.03, *dequantizer.args[2:])
    elif kind == "different_weight_encoding":
        weight = head.args[1]
        weight.args = (*weight.args[:5], -127, 127)
    elif kind == "bias":
        head.kwargs = {"bias": head.args[1].args[0]}
    elif kind == "repeated_transform":
        split_qwen35_output(module, 32, 83, 32)
    with pytest.raises(ValueError):
        split_qwen35_output(
            module,
            0 if kind == "nonpositive_chunk" else 32,
            84 if kind == "wrong_shape" else 83,
            32,
        )


@pytest.mark.parametrize("mapped", [False, True])
def test_qwen35_wrapper_transforms_both_methods_before_original_lowering(mapped):
    from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
        TEXT_DECODER,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.wrappers import llm_wrappers

    control, data, config, decode, prefill = scoped_inputs()
    if mapped:
        decode.meta.update(
            get_attention_valid=True,
            get_attention_cache_capacity=8192,
            get_max_context_len=8192,
            get_max_seq_len=8192,
        )
    decode.decoder, _ = frozen_projection(1)
    prefill.decoder, _ = frozen_projection(128)
    wrapper = llm_wrappers.Qwen35TextDecoder.__new__(llm_wrappers.Qwen35TextDecoder)
    wrapper.control_args, wrapper.config = control, config
    wrapper.decode, wrapper.prefill = decode, prefill
    request = SimpleNamespace(method_data={TEXT_DECODER: data})

    def lower(instance, actual_request):
        assert instance is wrapper and actual_request is request
        for decoder in (decode, prefill):
            assert (
                len(
                    [
                        node
                        for node in decoder.decoder.graph.nodes
                        if node.target == torch.ops.aten.conv2d.default
                    ]
                )
                == 3
            )

    def split(module):
        return split_qwen35_output(module, 32, 83, 32)

    with (
        patch.object(
            llm_wrappers, "split_qwen35_output", side_effect=split
        ) as transform,
        patch.object(llm_wrappers.HybridTextDecoder, "_compile_models", lower),
    ):
        wrapper._compile_models(request)
    assert transform.call_count == 2


@pytest.mark.parametrize("soc_model", ["SM8750", "SM8850"])
@pytest.mark.parametrize(
    "backend,expected",
    [
        ("htp", QnnExecuTorchBackendType.kHtpBackend),
        ("gpu", QnnExecuTorchBackendType.kGpuBackend),
    ],
)
def test_compile_request_preserves_selected_target(soc_model, backend, expected):
    from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
        TEXT_DECODER,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.wrappers import llm_wrappers

    manager = llm_wrappers.MultiModalManager.__new__(llm_wrappers.MultiModalManager)
    manager.control_args = SimpleNamespace(backend=backend, soc_model=soc_model)
    manager._modalities = [TEXT_DECODER]
    with patch.object(manager, "process") as process:
        manager.compile(
            {TEXT_DECODER: []}, {TEXT_DECODER: "test"}, {TEXT_DECODER: False}
        )
    data = process.call_args.args[0].method_data[TEXT_DECODER]
    assert data.backend == expected
    assert data.soc_model == soc_model
