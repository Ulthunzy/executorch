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
