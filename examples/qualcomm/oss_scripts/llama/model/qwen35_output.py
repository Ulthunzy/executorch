# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchao.quantization.pt2e.quantize_pt2e  # noqa: F401
import torchao.quantization.quant_primitives  # noqa: F401

from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)


_WEIGHT_DQ = torch.ops.torchao.dequantize_affine.default
_Q_OPS = {
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
}
_DQ_OPS = {
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
}
_CONV_OPS = {torch.ops.aten.conv2d.default, torch.ops.aten.convolution.default}
_LAYOUT_OPS = {
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.dims,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.clone.default,
    torch.ops.aten._to_copy.default,
}


def use_split_qwen35_output(control_args, data, config, decode, prefill):
    return (
        control_args.decoder_model == "qwen3_5-0_8b"
        and data.soc_model == "SM8850"
        and data.backend == QnnExecuTorchBackendType.kHtpBackend
        and not data.skip_quantize
        and control_args.model_mode == "hybrid"
        and control_args.max_seq_len == 8192
        and control_args.max_context_len == 8320
        and control_args.prefill_ar_len == 128
        and config.num_sharding == 1
        and decode is not None
        and decode.decoder is not None
        and decode.meta["get_ar_len"] == 1
        and decode.meta["get_max_context_len"]
        == (
            control_args.max_seq_len
            if decode.meta.get("get_attention_valid", False)
            else control_args.max_context_len
        )
        and (
            not decode.meta.get("get_attention_valid", False)
            or decode.meta.get("get_attention_cache_capacity")
            == control_args.max_seq_len
        )
        and decode.meta["get_dim"] == 1024
        and decode.meta["get_vocab_size"] == 248320
        and decode.meta["get_kv_io_bit_width"] == 8
        and prefill is not None
        and prefill.decoder is not None
        and prefill.meta["get_ar_len"] == 128
    )


def _constant(module, node):
    if not isinstance(node, torch.fx.Node) or node.op != "get_attr":
        raise ValueError(f"Expected frozen output tensor, received {node}")
    try:
        return module.get_buffer(node.target)
    except AttributeError:
        return module.get_parameter(node.target)


def _reaches_output(node):
    pending, seen = [node], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for user in current.users:
            if user.op == "output":
                return True
            if (
                user.op == "call_function"
                and user.target in _LAYOUT_OPS | _Q_OPS | _DQ_OPS
            ):
                pending.append(user)
    return False


def _copy_metadata(source, node, start=None, stop=None, axis=0):
    node.meta = dict(source.meta)
    value = source.meta.get("val")
    if start is not None and isinstance(value, torch.Tensor):
        indices = [slice(None)] * value.ndim
        indices[axis] = slice(start, stop)
        node.meta["val"] = value[tuple(indices)]


def split_qwen35_output(
    module, max_rows=4096, expected_rows=248320, expected_columns=1024
):
    if max_rows <= 0:
        raise ValueError("Output chunk rows must be positive")
    matches = []
    for node in module.graph.nodes:
        if node.target not in _CONV_OPS:
            continue
        weight = node.args[1]
        if not isinstance(weight, torch.fx.Node) or weight.target != _WEIGHT_DQ:
            continue
        codes = _constant(module, weight.args[0])
        if tuple(codes.shape) == (
            expected_rows,
            expected_columns,
            1,
            1,
        ) and _reaches_output(node):
            matches.append(node)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one frozen Qwen35 output projection, found {len(matches)}"
        )
    head = matches[0]
    weight = head.args[1]
    codes = _constant(module, weight.args[0])
    rows, columns = codes.shape[:2]
    if (
        tuple(weight.args[1]) != (1, 32, 1, 1)
        or tuple(weight.args[5:7]) != (-7, 7)
        or codes.dtype != torch.int8
        or columns % 32
    ):
        raise ValueError(
            "Qwen35 output splitting requires unchanged W4 group32 weights"
        )
    scales = _constant(module, weight.args[2])
    zero = _constant(module, weight.args[3]) if weight.args[3] is not None else None
    if scales.shape[0] != rows or (zero is not None and zero.shape[0] != rows):
        raise ValueError("Output weight qparams must have the output-row axis first")
    quantizers = [
        node for node in head.users if node.target in _Q_OPS and node.args[0] is head
    ]
    if len(quantizers) != 1:
        raise ValueError("Expected one original output quantizer")
    quantizer = quantizers[0]
    if set(head.users) != {quantizer}:
        raise ValueError("Output projection has consumers outside the calibrated QDQ")
    dequantizers = [
        node
        for node in quantizer.users
        if node.target in _DQ_OPS and node.args[0] is quantizer
    ]
    if not dequantizers:
        raise ValueError("Expected an original output dequantizer")
    dequantizer = dequantizers[0]
    if tuple(quantizer.args[1:]) != tuple(dequantizer.args[1:]) or quantizer.kwargs != {
        key: value for key, value in dequantizer.kwargs.items() if key != "out_dtype"
    }:
        raise ValueError("Output quantizer and dequantizer must use the same encoding")
    if any(
        tuple(node.args[1:]) != tuple(dequantizer.args[1:])
        or node.kwargs != dequantizer.kwargs
        for node in dequantizers
    ):
        raise ValueError("Output dequantizers must use the same quantization encoding")
    if head.target == torch.ops.aten.conv2d.default:
        stride = (
            head.args[3] if len(head.args) > 3 else head.kwargs.get("stride", [1, 1])
        )
        padding = (
            head.args[4] if len(head.args) > 4 else head.kwargs.get("padding", [0, 0])
        )
        dilation = (
            head.args[5] if len(head.args) > 5 else head.kwargs.get("dilation", [1, 1])
        )
        groups = head.args[6] if len(head.args) > 6 else head.kwargs.get("groups", 1)
        valid_conv = (
            list(stride) == [1, 1]
            and list(padding) == [0, 0]
            and list(dilation) == [1, 1]
            and groups == 1
        )
    else:
        valid_conv = (
            list(head.args[3]) == [1, 1]
            and list(head.args[4]) == [0, 0]
            and list(head.args[5]) == [1, 1]
            and not head.args[6]
            and list(head.args[7]) == [0, 0]
            and head.args[8] == 1
        )
    if not valid_conv:
        raise ValueError("Unexpected Qwen35 output convolution configuration")
    bias = head.args[2] if len(head.args) > 2 else head.kwargs.get("bias")
    if bias is not None:
        raise ValueError("Qwen35 output splitting requires the original bias-free head")
    graph = module.graph

    def slice_attribute(source, label, index, start, stop):
        name = f"_lm_head_split_{label}_{index}"
        if name in module._buffers or name in module._parameters:
            raise ValueError(f"Output chunk attribute already exists: {name}")
        module.register_buffer(name, _constant(module, source)[start:stop].contiguous())
        node = graph.get_attr(name)
        _copy_metadata(source, node, start, stop)
        return node

    chunks = []
    with graph.inserting_before(head):
        for index, start in enumerate(range(0, rows, max_rows)):
            stop = min(start + max_rows, rows)
            code_node = slice_attribute(weight.args[0], "codes", index, start, stop)
            scale_node = slice_attribute(weight.args[2], "scales", index, start, stop)
            zero_node = (
                slice_attribute(weight.args[3], "zero", index, start, stop)
                if zero is not None
                else None
            )
            weight_args = (
                code_node,
                weight.args[1],
                scale_node,
                zero_node,
                *weight.args[4:],
            )
            chunk_weight = graph.call_function(
                weight.target, weight_args, dict(weight.kwargs)
            )
            _copy_metadata(weight, chunk_weight, start, stop)
            head_args = list(head.args)
            head_args[1] = chunk_weight
            chunk_head = graph.call_function(
                head.target, tuple(head_args), dict(head.kwargs)
            )
            _copy_metadata(head, chunk_head, start, stop, axis=1)
            chunk_q = graph.call_function(
                quantizer.target,
                (chunk_head, *quantizer.args[1:]),
                dict(quantizer.kwargs),
            )
            _copy_metadata(quantizer, chunk_q, start, stop, axis=1)
            chunk_dq = graph.call_function(
                dequantizer.target,
                (chunk_q, *dequantizer.args[1:]),
                dict(dequantizer.kwargs),
            )
            _copy_metadata(dequantizer, chunk_dq, start, stop, axis=1)
            chunks.append(chunk_dq)
        concatenated = graph.call_function(torch.ops.aten.cat.default, (chunks, 1))
        _copy_metadata(head, concatenated)
    # Each branch uses the calibrated output grid before the original external QDQ.
    head.replace_all_uses_with(concatenated)
    graph.erase_node(head)
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    return len(chunks)
