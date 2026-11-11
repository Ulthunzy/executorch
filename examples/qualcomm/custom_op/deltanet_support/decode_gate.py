# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
from pathlib import Path

import torch

from executorch.examples.qualcomm.custom_op.htp_op_package.package import op_package

library = torch.library.Library("qnn_custom_ops", "FRAGMENT")
library.define(
    "deltanet_core_gated(Tensor query, Tensor key, Tensor value, Tensor activation, "
    "Tensor beta, Tensor state, Tensor table, Tensor grids) -> (Tensor, Tensor)"
)


@torch.library.impl(library, "deltanet_core_gated", "CompositeExplicitAutograd")
def _reference(query, key, value, activation, beta, state, table, grids):
    import executorch.examples.qualcomm.custom_op.deltanet  # noqa: F401

    p = grids.flatten()
    code = (activation / p[0]).round().add(p[1]).clamp(0, 65535).long()
    heads = activation.shape[-1]
    values = table.view(torch.uint16).reshape(heads, 65536).to(torch.int32)
    head = torch.arange(heads, device=activation.device)
    decay = (values[head, code].float() - p[3]) * p[2]
    return torch.ops.qnn_custom_ops.deltanet_core.default(
        query, key, value, decay, beta, state
    )


@torch.library.register_fake("qnn_custom_ops::deltanet_core_gated")
def _fake(query, key, value, activation, beta, state, table, grids):
    torch._check(query.ndim == 4)
    batch, length, heads, key_dim = query.shape
    torch._check(length == 1)
    torch._check(key.shape == query.shape)
    torch._check(value.shape[:3] == (batch, length, heads))
    torch._check(activation.shape == (batch, length, heads))
    torch._check(beta.shape == activation.shape)
    torch._check(state.shape == (batch, heads, key_dim, value.shape[-1]))
    torch._check(table.shape == (1, heads, 256, 128))
    torch._check(table.dtype == torch.int32)
    torch._check(table.is_contiguous())
    torch._check(grids.shape == (1, 1, 1, 4))
    torch._check(grids.dtype == torch.float32)
    return torch.empty_like(value), torch.empty_like(state)


core_gated = torch.ops.qnn_custom_ops.deltanet_core_gated.default


def decode_gate_package(host, device):
    return op_package(host, device, [(core_gated, "DeltaNetCoreGatedVtcm")])


def _strip(node):
    operations = {
        "quantized_decomposed.quantize_per_tensor.default",
        "quantized_decomposed.dequantize_per_tensor.default",
        "aten.unsqueeze.default",
        "aten.squeeze.dim",
        "aten.squeeze.dims",
        "aten.to.dtype",
    }
    while str(node.target) in operations:
        node = node.args[0]
    return node


def _constant(node):
    if node.op == "get_attr":
        return True
    if node.op == "placeholder":
        return False
    return all(_constant(n) for n in node.all_input_nodes)


def gate_activation(output):
    exponential = _strip(output)
    if exponential.target != torch.ops.aten.exp.default:
        raise ValueError("Expected the calibrated GDN decay exponential")
    product = _strip(exponential.args[0])
    if product.target != torch.ops.aten.mul.Tensor:
        raise ValueError("Expected the calibrated GDN decay product")
    softplus = next(_strip(n) for n in product.all_input_nodes if not _constant(n))
    if softplus.target != torch.ops.aten.add.Tensor:
        raise ValueError("Expected the calibrated GDN softplus decomposition")
    relu = next(
        (
            _strip(n)
            for n in softplus.all_input_nodes
            if _strip(n).target == torch.ops.aten.relu.default
        ),
        None,
    )
    if relu is None:
        raise ValueError("Expected the calibrated GDN softplus ReLU")
    addition = _strip(relu.args[0])
    if addition.target != torch.ops.aten.add.Tensor:
        raise ValueError("Expected GDN activation plus dt_bias")
    activation = next(n for n in addition.all_input_nodes if not _constant(n))
    if str(activation.target) != "quantized_decomposed.dequantize_per_tensor.default":
        raise ValueError("Expected a calibrated U16 GDN activation")
    return activation


def gate_signature(module, activation, output):
    state = module.state_dict()

    def encode(value):
        if isinstance(value, torch.fx.Node):
            if value is activation:
                return ["activation"]
            if value.op == "get_attr":
                tensor = state[value.target].detach().cpu().contiguous()
                return [
                    "constant",
                    str(tensor.dtype),
                    list(tensor.shape),
                    hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                ]
            return [str(value.target), encode(value.args), encode(value.kwargs)]
        if isinstance(value, (tuple, list)):
            return [encode(x) for x in value]
        if isinstance(value, dict):
            return {k: encode(v) for k, v in sorted(value.items())}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    return hashlib.sha256(
        json.dumps(encode(output), sort_keys=True).encode()
    ).hexdigest()


def fuse_decode_gates(module, directory):
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())
    if index["version"] != 1 or index["ar_len"] != 1 or index["packing"] != "u16le":
        raise ValueError("Expected an AR1 SDK-generated GDN gate table directory")
    cores = [
        n
        for n in module.graph.nodes
        if str(n.target) == "qnn_custom_ops.deltanet_core.default"
    ]
    if len(cores) != len(index["layers"]):
        raise ValueError("GDN gate table count does not match the decode graph")
    records = []
    for ordinal, (core, entry) in enumerate(zip(cores, index["layers"])):
        output = core.args[3]
        activation = gate_activation(output)
        if (
            list(activation.args[1:-1]) != entry["input_grid"]
            or list(output.args[1:-1]) != entry["output_grid"]
        ):
            raise ValueError(
                f"GDN layer {ordinal}: gate quantization does not match the SDK table"
            )
        if gate_signature(module, activation, output) != entry["signature"]:
            raise ValueError(
                f"GDN layer {ordinal}: gate constants or arithmetic changed; regenerate the SDK table"
            )
        payload = (directory / entry["file"]).read_bytes()
        heads = entry["heads"]
        if (
            len(payload) != heads * 65536 * 2
            or hashlib.sha256(payload).hexdigest() != entry["sha256"]
        ):
            raise ValueError(f"GDN layer {ordinal}: invalid SDK table data")
        table = torch.frombuffer(bytearray(payload), dtype=torch.int32).reshape(
            1, heads, 256, 128
        )
        grids = torch.tensor(
            activation.args[1:3] + output.args[1:3], dtype=torch.float32
        ).reshape(1, 1, 1, 4)
        table_name, grids_name = (
            f"decode_gate_table_{ordinal}",
            f"decode_gate_grids_{ordinal}",
        )
        module.register_buffer(table_name, table)
        module.register_buffer(grids_name, grids)
        with module.graph.inserting_before(core):
            table_node = module.graph.get_attr(table_name)
            grid_node = module.graph.get_attr(grids_name)
        values = list(core.args)
        values[3] = activation
        core.target = core_gated
        core.args = (*values, table_node, grid_node)
        records.append(
            {"core": core.name, "table": entry["file"], "sha256": entry["sha256"]}
        )
    module.graph.eliminate_dead_code()
    module.recompile()
    return records
