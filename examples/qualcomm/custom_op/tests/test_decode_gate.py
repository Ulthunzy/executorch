# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch.ao.quantization.fx import _decomposed  # noqa: F401

import executorch.examples.qualcomm.custom_op.deltanet  # noqa: F401
from executorch.examples.qualcomm.custom_op.deltanet_support.decode_gate import (
    core_gated,
    fuse_decode_gates,
    gate_activation,
    gate_signature,
)


class GateCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.tensor([-1.0, -2.0]))
        self.register_buffer("coefficient", torch.tensor([-0.5, -1.5]))

    def forward(self, query, key, value, activation, beta, state):
        q = torch.ops.quantized_decomposed.quantize_per_tensor.default
        dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
        activation = dq(
            q(activation, 1 / 128, 32768, 0, 65535, torch.int32),
            1 / 128,
            32768,
            0,
            65535,
            torch.int32,
        )
        x = activation + self.bias
        softplus = torch.relu(x) + torch.log(1 + torch.exp(-torch.abs(x)))
        decay = torch.exp(self.coefficient * softplus)
        decay = dq(
            q(decay, 1 / 65535, 0, 0, 65535, torch.int32),
            1 / 65535,
            0,
            0,
            65535,
            torch.int32,
        )
        return torch.ops.qnn_custom_ops.deltanet_core.default(
            query, key, value, decay, beta, state
        )


@pytest.fixture
def gate_case(tmp_path):
    inputs = (
        torch.zeros(1, 1, 2, 32),
        torch.zeros(1, 1, 2, 32),
        torch.zeros(1, 1, 2, 64),
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, 2),
        torch.zeros(1, 2, 32, 64),
    )
    module = torch.export.export(GateCore(), inputs, strict=True).module()
    core = next(
        n
        for n in module.graph.nodes
        if n.target == torch.ops.qnn_custom_ops.deltanet_core.default
    )
    output = core.args[3]
    activation = gate_activation(output)
    payload = torch.zeros(1, 2, 256, 128, dtype=torch.int32).numpy().tobytes()
    (tmp_path / "layer-0.bin").write_bytes(payload)
    entry = dict(
        file="layer-0.bin",
        heads=2,
        input_grid=list(activation.args[1:-1]),
        output_grid=list(output.args[1:-1]),
        signature=gate_signature(module, activation, output),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    index = dict(version=1, ar_len=1, packing="u16le", layers=[entry])
    (tmp_path / "index.json").write_text(json.dumps(index))
    return module, inputs, tmp_path


def test_fusion_preserves_exported_recurrence_boundary(gate_case):
    module, inputs, directory = gate_case
    records = fuse_decode_gates(module, directory)
    assert len(records) == 1
    exported = torch.export.export(module, inputs, strict=True)
    assert sum(n.target == core_gated for n in exported.graph.nodes) == 1
    assert not any(n.target == torch.ops.aten.exp.default for n in exported.graph.nodes)


def test_rejects_changed_gate_constants(gate_case):
    module, _, directory = gate_case
    module.get_buffer("bias").add_(0.25)
    with pytest.raises(ValueError, match="constants or arithmetic changed"):
        fuse_decode_gates(module, directory)


def test_rejects_corrupt_table(gate_case):
    module, _, directory = gate_case
    path = directory / "layer-0.bin"
    payload = bytearray(path.read_bytes())
    payload[0] = 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="invalid SDK table data"):
        fuse_decode_gates(module, directory)


def test_default_llama_decode_fuses_gates(gate_case, monkeypatch):
    from executorch.examples.qualcomm.custom_op.deltanet_support import gate_lut
    from executorch.examples.qualcomm.oss_scripts.llama.wrappers.llm_wrappers import (
        Qwen35TextDecoder,
    )

    module, inputs, directory = gate_case
    requests = []

    def prepare(graph, target, soc_model, **kwargs):
        requests.append((graph, target, soc_model))
        return directory

    monkeypatch.setattr(gate_lut, "prepare_gate_tables", prepare)
    wrapper = Qwen35TextDecoder.__new__(Qwen35TextDecoder)
    wrapper.control_args = SimpleNamespace(
        model_mode="kv",
        artifact=str(directory),
        soc_model="SM8850",
        device=None,
        host=None,
    )
    wrapper.fused_gdn, wrapper.online_attention = True, False
    wrapper.decode = SimpleNamespace(
        decoder=module, export_input=inputs, meta={"get_max_context_len": 1024}
    )
    wrapper.prefill = SimpleNamespace()
    base = SimpleNamespace(
        tok_embedding=torch.nn.Embedding(4, 8),
        tok_embedding_export_input=(torch.zeros(1, 128, dtype=torch.int64),),
    )
    wrapper._prepare_generation_graphs(base, inputs)
    assert requests == [(module, directory / "decode_gate_tables", "SM8850")]
    assert len(wrapper.decode_gate_tables) == 1
    targets = [n.target for n in wrapper.decode.decoder.graph.nodes]
    assert targets.count(core_gated) == 1
    assert torch.ops.aten.exp.default not in targets


def test_automatic_table_cache_and_invalidation(gate_case, tmp_path, monkeypatch):
    from executorch.examples.qualcomm.custom_op.deltanet_support import gate_lut

    module, _, _ = gate_case
    sdk = tmp_path / "sdk"
    for name in (
        "lib/aarch64-android/libQnnHtp.so",
        "lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so",
        "lib/x86_64-linux-clang/libQnnHtp.so",
        "lib/aarch64-android/libQnnHtpPrepare.so",
    ):
        path = sdk / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sdk version one")
    monkeypatch.setenv("QNN_SDK_ROOT", str(sdk))
    generated = []

    class Device:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def generate(self, context, output, log):
            generated.append(context)
            output.write_bytes(bytes(2 * 65536 * 2))

    monkeypatch.setattr(gate_lut, "_DeviceGenerator", Device)
    monkeypatch.setattr(gate_lut, "_export_gate", lambda *args: "gate.bin")
    directory = tmp_path / "automatic"
    gate_lut.prepare_gate_tables(module, directory, "SM8850")
    gate_lut.prepare_gate_tables(module, directory, "SM8850", device="unavailable")
    assert len(generated) == 1
    module.get_buffer("bias").add_(0.25)
    gate_lut.prepare_gate_tables(module, directory, "SM8850")
    assert len(generated) == 2
    (sdk / "lib/aarch64-android/libQnnHtp.so").write_bytes(b"sdk version two")
    gate_lut.prepare_gate_tables(module, directory, "SM8850")
    assert len(generated) == 3
    entry = json.loads((directory / "index.json").read_text())["layers"][0]
    (directory / entry["file"]).write_bytes(bytes(2 * 65536 * 2 - 1))
    gate_lut.prepare_gate_tables(module, directory, "SM8850")
    assert len(generated) == 4
    assert len(fuse_decode_gates(module, directory)) == 1
