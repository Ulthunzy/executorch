import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import executorch.examples.qualcomm.custom_op.deltanet as deltanet
from export.prefill import padded_prefill_front
from export.graph_utils import resize_attention
from export.fp16_attention_core import fp16_attention_core
from export.gdn_state import fp16_gdn_state
from executorch.examples.qualcomm.oss_scripts.llama.quantize.generation import (
    replace_prefill_cores,
)


class Front(torch.nn.Module):
    def forward(self, x, conv, state):
        small = x[..., :4]
        joined = torch.cat((conv, small.transpose(1, 2)), dim=-1)
        q = torch.tanh(small).unsqueeze(2)
        k = torch.tanh(small * 0.5).unsqueeze(2)
        v = small.unsqueeze(2)
        decay = torch.full((1, 128, 1), 0.98)
        beta = torch.full((1, 128, 1), 0.1)
        out, state = torch.ops.qnn_custom_ops.deltanet_core.default(
            q, k, v, decay, beta, state
        )
        return out.flatten(2), x, joined[..., -4:], state


class QuantizedFront(torch.nn.Module):
    def forward(self, x, conv, state):
        def qdq(value, scale, zero):
            args = (scale, zero, 0, 65535, torch.int32)
            codes = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                value, *args
            )
            return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                codes, *args
            )

        small = x[..., :4]
        joined = torch.cat((conv, small.transpose(1, 2)), dim=-1)
        out, new_state = torch.ops.qnn_custom_ops.deltanet_core.default(
            qdq(torch.tanh(small).unsqueeze(2), 0.0001, 32768),
            qdq(torch.tanh(small * 0.5).unsqueeze(2), 0.0001, 32768),
            qdq(small.unsqueeze(2), 0.0001, 32768),
            qdq(torch.full((1, 128, 1), 0.98), 0.00002, 0),
            qdq(torch.full((1, 128, 1), 0.1), 0.00002, 0),
            qdq(state, 0.0001, 32768),
        )
        out = qdq(out, 0.0001, 32768)
        new_state = qdq(new_state, 0.0001, 32768)
        return out.flatten(2), x, joined[..., -4:], new_state


@pytest.mark.parametrize("valid", [0, 1, 63, 128])
def test_fp16_state_chunked_padding_and_interface(front, valid):
    inputs = front[2]
    source = torch.export.export(QuantizedFront(), inputs).module()
    source, inputs = fp16_gdn_state(source, inputs, decode=False)
    padded, _ = padded_prefill_front(source, inputs, 2048)
    assert not any("deltanet_core" in str(node.target) for node in padded.graph.nodes)
    index = torch.tensor([valid - 1], dtype=torch.int32)
    result = padded(*inputs, index)
    assert result[2].dtype == torch.float32
    assert result[3].dtype == torch.float16
    changed = inputs[0].clone()
    changed[:, valid:] = 20 * torch.randn_like(changed[:, valid:])
    alternate = padded(changed, *inputs[1:], index)
    for left, right in zip(
        (result[0][:, :valid], *result[2:]), (alternate[0][:, :valid], *alternate[2:])
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    if valid == 0:
        torch.testing.assert_close(result[3], inputs[2], rtol=0, atol=0)
    exported = torch.export.export(padded, (*inputs, index))
    assert exported.example_inputs[0][2].dtype == torch.float16


def test_fp16_state_decode_keeps_projection_and_attention_qdq(front):
    inputs = front[2]
    source = torch.export.export(QuantizedFront(), inputs).module()
    old_core = next(
        node for node in source.graph.nodes if "deltanet_core" in str(node.target)
    )
    old_state = next(user for user in old_core.users if user.args[1] == 1)
    for node in (old_core, old_state):
        node.meta["quant_attrs"] = {
            "scale": 0.0001,
            "zero_point": 32768,
            "dtype": torch.int32,
        }
    q = torch.ops.quantized_decomposed.quantize_per_tensor.default
    grids = [node.args[1:] for node in source.graph.nodes if node.target == q]
    source, inputs = fp16_gdn_state(source, inputs, decode=True)
    remaining = [node.args[1:] for node in source.graph.nodes if node.target == q]
    assert remaining == [*grids[:5], grids[6]]
    core = next(
        node for node in source.graph.nodes if "deltanet_core" in str(node.target)
    )
    assert core.target == torch.ops.qnn_custom_ops.deltanet_core_fp16_state.default
    assert core.args[5].meta["val"].dtype == torch.float16
    assert "quant_attrs" not in core.meta
    assert all(
        "quant_attrs" not in user.meta for user in core.users if user.args[1] == 1
    )
    result = source(*inputs)
    assert result[0].dtype == torch.float32
    assert result[3].dtype == torch.float16
    assert (
        torch.export.export(source, inputs).module()(*inputs)[3].dtype == torch.float16
    )


def test_fp16_state_removes_inner_and_outer_cache_qdq(front):
    class DoubleBoundary(torch.nn.Module):
        def forward(self, x, conv, state):
            values = QuantizedFront()(x, conv, state)
            args = (0.0002, 30000, 0, 65535, torch.int32)
            floating = torch.ops.aten._to_copy.default(values[3], dtype=torch.float32)
            codes = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                floating, *args
            )
            state = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                codes, *args
            )
            return (*values[:3], state)

    inputs = front[2]
    source = torch.export.export(DoubleBoundary(), inputs).module()
    source, inputs = fp16_gdn_state(source, inputs, decode=True)
    core = next(
        node for node in source.graph.nodes if "deltanet_core" in str(node.target)
    )
    state = next(user for user in core.users if user.args[1] == 1)
    assert all("quantize" not in str(user.target) for user in state.users)
    assert (
        sum(
            node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
            for node in source.graph.nodes
        )
        == 6
    )
    assert source(*inputs)[3].dtype == torch.float16


@pytest.fixture(scope="module")
def front():
    torch.set_num_threads(2)
    torch.manual_seed(43)
    inputs = (
        torch.randn(1, 128, 128) * 0.1,
        torch.randn(1, 4, 4) * 0.01,
        torch.randn(1, 1, 4, 4) * 0.01,
    )
    module = torch.export.export(Front(), inputs).module()
    baseline = copy.deepcopy(module)
    replace_prefill_cores(baseline)
    padded, _ = padded_prefill_front(module, inputs, 2048)
    return padded, baseline, inputs


def test_full_block_parity(front):
    padded, baseline, inputs = front
    actual = padded(*inputs, torch.tensor([127], dtype=torch.int32))
    for result, reference in zip(actual, baseline(*inputs)):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("valid", [1, 2, 63, 64, 65, 127])
def test_padding_does_not_advance_state(front, valid):
    padded, _, inputs = front
    x, conv, state = inputs
    index = torch.tensor([valid - 1], dtype=torch.int32)
    actual = padded(*inputs, index)
    changed = x.clone()
    changed[:, valid:] = torch.randn_like(changed[:, valid:]) * 30
    other = padded(changed, conv, state, index)
    for a, b in zip(
        (actual[0][:, :valid], *actual[2:]), (other[0][:, :valid], *other[2:])
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(
        actual[2],
        torch.cat((conv, x[..., :4].transpose(1, 2)), -1)[..., valid : valid + 4],
        rtol=0,
        atol=0,
    )
    q = torch.tanh(x[:, :valid, :4]).unsqueeze(2).half().float()
    k = torch.tanh(x[:, :valid, :4] * 0.5).unsqueeze(2).half().float()
    v = x[:, :valid, :4].unsqueeze(2).half().float()
    expected_out, expected_state = deltanet._deltanet_core_decay_reference(
        q,
        k,
        v,
        torch.full((1, valid, 1), 0.98).half().float(),
        torch.full((1, valid, 1), 0.1).half().float(),
        state.half().float(),
    )
    torch.testing.assert_close(
        actual[0][:, :valid], expected_out.flatten(2), rtol=0.02, atol=0.0002
    )
    torch.testing.assert_close(actual[3], expected_state, rtol=0.03, atol=0.0002)


def test_129_tokens_use_two_blocks(front):
    padded, _, (_, conv, state) = front
    torch.manual_seed(44)
    x = torch.randn(1, 129, 128) * 0.1
    first = padded(x[:, :128], conv, state, torch.tensor([127], dtype=torch.int32))
    tail = torch.zeros(1, 128, 128)
    tail[:, :1] = x[:, 128:]
    second = padded(tail, first[2], first[3], torch.tensor([0], dtype=torch.int32))
    small = x[..., :4]
    expected, expected_state = deltanet._deltanet_core_decay_reference(
        torch.tanh(small).unsqueeze(2).half().float(),
        torch.tanh(small * 0.5).unsqueeze(2).half().float(),
        small.unsqueeze(2).half().float(),
        torch.full((1, 129, 1), 0.98).half().float(),
        torch.full((1, 129, 1), 0.1).half().float(),
        state.half().float(),
    )
    torch.testing.assert_close(
        torch.cat((first[0], second[0][:, :1]), dim=1),
        expected.flatten(2),
        rtol=0.02,
        atol=0.0002,
    )
    torch.testing.assert_close(second[3], expected_state, rtol=0.03, atol=0.0002)
    torch.testing.assert_close(second[2], small[:, -4:].transpose(1, 2), rtol=0, atol=0)


@pytest.fixture(scope="module")
def custom_front(front):
    inputs = front[2]
    source = torch.export.export(Front(), inputs).module()
    padded, _ = padded_prefill_front(source, inputs, 2048, "custom_fp16")
    cores = [
        node
        for node in padded.graph.nodes
        if str(node.target) == "qnn_custom_ops.deltanet_core.default"
    ]
    assert len(cores) == 1
    assert all(value.meta["val"].dtype == torch.float16 for value in cores[0].args)
    return padded, inputs


@pytest.mark.parametrize("valid", [1, 63, 127, 128])
def test_custom_fp16_padding_and_float_boundaries(custom_front, valid):
    module, inputs = custom_front
    x, conv, state = inputs
    index = torch.tensor([valid - 1], dtype=torch.int32)
    actual = module(*inputs, index)
    changed = x.clone()
    changed[:, valid:] = 40 * torch.randn_like(changed[:, valid:])
    alternate = module(changed, conv, state, index)
    for left, right in zip(
        (actual[0][:, :valid], *actual[2:]), (alternate[0][:, :valid], *alternate[2:])
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert all(value.dtype == torch.float32 for value in actual)
    small = x[:, :valid, :4]
    output, expected_state = deltanet._deltanet_core_decay_reference(
        torch.tanh(small).unsqueeze(2).half(),
        torch.tanh(small * 0.5).unsqueeze(2).half(),
        small.unsqueeze(2).half(),
        torch.full((1, valid, 1), 0.98).half(),
        torch.full((1, valid, 1), 0.1).half(),
        state.half(),
    )
    torch.testing.assert_close(
        actual[0][:, :valid], output.flatten(2).float(), rtol=0, atol=0
    )
    torch.testing.assert_close(actual[3], expected_state.float(), rtol=0, atol=0)


def test_custom_fp16_129_tokens_keep_only_valid_state(custom_front):
    module, (_, conv, state) = custom_front
    torch.manual_seed(45)
    x = torch.randn(1, 129, 128) * 0.1
    first = module(x[:, :128], conv, state, torch.tensor([127], dtype=torch.int32))
    padded = torch.zeros(1, 128, 128)
    padded[:, 0] = x[:, 128]
    actual = module(padded, first[2], first[3], torch.tensor([0], dtype=torch.int32))
    padded[:, 1:] = 40 * torch.randn_like(padded[:, 1:])
    changed = module(padded, first[2], first[3], torch.tensor([0], dtype=torch.int32))
    torch.testing.assert_close(actual[3], changed[3], rtol=0, atol=0)
    torch.testing.assert_close(
        actual[2], x[..., :4][:, -4:].transpose(1, 2), rtol=0, atol=0
    )


@pytest.mark.parametrize("valid", [0, 1, 63, 128])
def test_custom_fp32_keeps_calibrated_boundaries_and_padding(front, valid):
    inputs = front[2]
    source = torch.export.export(QuantizedFront(), inputs).module()
    baseline = copy.deepcopy(source)
    module, _ = padded_prefill_front(source, inputs, 2048, "custom_fp32")
    core = next(
        node
        for node in module.graph.nodes
        if str(node.target) == "qnn_custom_ops.deltanet_core.default"
    )
    assert all(value.meta["val"].dtype == torch.float32 for value in core.args)
    index = torch.tensor([valid - 1], dtype=torch.int32)
    actual = module(*inputs, index)
    changed = inputs[0].clone()
    changed[:, valid:] = 40 * torch.randn_like(changed[:, valid:])
    alternate = module(changed, *inputs[1:], index)
    torch.testing.assert_close(
        actual[0][:, :valid], alternate[0][:, :valid], rtol=0, atol=0
    )
    torch.testing.assert_close(actual[3], alternate[3], rtol=0, atol=0)
    if valid == 128:
        for result, expected in zip(actual, baseline(*inputs)):
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
    if valid == 0:
        params = (0.0001, 32768, 0, 65535, torch.int32)
        q = torch.ops.quantized_decomposed.quantize_per_tensor.default
        dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
        torch.testing.assert_close(
            actual[3], dq(q(inputs[2], *params), *params), rtol=0, atol=0
        )


@pytest.mark.parametrize("valid", [1, 63, 128])
def test_custom_u16_padding_preserves_calibrated_state_grid(front, valid):
    _, _, inputs = front
    source = torch.export.export(QuantizedFront(), inputs).module()
    module, _ = padded_prefill_front(source, inputs, 2048, "custom_u16")
    core = next(
        node
        for node in module.graph.nodes
        if str(node.target) == "qnn_custom_ops.deltanet_core.default"
    )
    assert all(
        value.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default
        for value in core.args
    )
    assert core.args[5].args[1:] == (0.0001, 32768, 0, 65535, torch.int32)
    assert core.args[3].args[1] == 1 / 65535
    assert torch.tensor(65535, dtype=torch.float32) * core.args[3].args[1] == 1
    assert not any(
        node.target == torch.ops.aten._to_copy.default
        and node.kwargs.get("dtype") == torch.float16
        for node in module.graph.nodes
    )
    x, conv, state = inputs
    index = torch.tensor([valid - 1], dtype=torch.int32)
    expected = module(x, conv, state, index)
    altered = x.clone()
    altered[:, valid:] = 50 * torch.randn_like(altered[:, valid:])
    actual = module(altered, conv, state, index)
    torch.testing.assert_close(
        expected[0][:, :valid], actual[0][:, :valid], rtol=0, atol=0
    )
    torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
    torch.testing.assert_close(expected[3], actual[3], rtol=0, atol=0)


def test_custom_u16_requires_calibrated_inputs(front):
    inputs = front[2]
    source = torch.export.export(Front(), inputs).module()
    with pytest.raises(ValueError, match="six calibrated UINT16"):
        padded_prefill_front(source, inputs, 2048, "custom_u16")


def test_context_resize_keeps_hidden_width():
    class Attention(torch.nn.Module):
        def forward(self, hidden, cos, sin, mask, key, value):
            scores = torch.matmul(hidden[:, :, :256].unsqueeze(1), key)
            scores = scores.reshape(1, 2, 128, 1920)
            return hidden, scores, mask, value

    inputs = (
        torch.zeros(1, 128, 2048),
        torch.ones(128, 32),
        torch.zeros(128, 32),
        torch.zeros(1, 1, 128, 2048),
        torch.zeros(1, 2, 256, 1920),
        torch.zeros(1, 2, 1920, 256),
    )
    module = torch.export.export(Attention(), inputs).module()
    resized = resize_attention(module, 4224)
    assert resized[0].shape == (1, 128, 2048)
    assert resized[4].shape == (1, 2, 256, 4096)
    assert module(*resized)[0].shape == (1, 128, 2048)


def test_long_gqa_tiles_preserve_global_softmax():
    def qdq(x, scale=0.0001):
        codes = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            x, scale, 0, -32768, 32767, torch.int32
        )
        return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            codes, scale, 0, -32768, 32767, torch.int32
        )

    class Gqa(torch.nn.Module):
        def forward(self, query, key, value, new_key, new_value):
            key = qdq(torch.cat((qdq(key), qdq(new_key)), dim=3))
            value = qdq(torch.cat((qdq(value), qdq(new_value)), dim=2))
            key = key.unsqueeze(2).expand(1, 2, 8, 4, 4203).reshape(1, 16, 4, 4203)
            value = value.unsqueeze(2).expand(1, 2, 8, 4203, 4).reshape(1, 16, 4203, 4)
            scores = qdq(torch.matmul(qdq(query), qdq(key)))
            probabilities = qdq(torch.softmax(scores, dim=-1), 0.000001)
            return qdq(torch.matmul(probabilities, qdq(value)))

    torch.manual_seed(45)
    inputs = tuple(
        torch.randn(shape) * 0.1
        for shape in (
            (1, 16, 3, 4),
            (1, 2, 4, 4200),
            (1, 2, 4200, 4),
            (1, 2, 4, 3),
            (1, 2, 3, 4),
        )
    )
    module = torch.export.export(Gqa(), inputs).module()
    expected = module(*inputs)
    fp16_attention_core(module, inputs)
    torch.testing.assert_close(module(*inputs), expected, rtol=0.03, atol=0.0003)
    assert (
        sum(node.target == torch.ops.aten.softmax.int for node in module.graph.nodes)
        == 1
    )
