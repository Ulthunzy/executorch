# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Group queries without repeating KV and preserve calibrated cache precision."""

import copy
import math
import operator

import torch
import torchao.quantization.pt2e.quantize_pt2e  # noqa: F401

from executorch.examples.qualcomm.custom_op.deltanet_support.chunk import ChunkGDN
from torch._subclasses.fake_tensor import FakeTensorMode


Q = torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ = torch.ops.quantized_decomposed.dequantize_per_tensor.default


def remove_export_checks(module):
    module.graph.set_codegen(torch.fx.graph.CodeGen())
    # ExportedModule's input-tree guard becomes stale when adding row selection
    # or specializing shapes. The next export validates the new signature.
    module._forward_pre_hooks.clear()
    for node in list(module.graph.nodes):
        if "aten._assert_tensor_metadata" in str(node.target) or (
            node.op == "call_module" and node.target == "_guards_fn"
        ):
            assert not node.users
            module.graph.erase_node(node)
    module.recompile()


def position_input_index(inputs):
    return next(
        index
        for index, value in enumerate(inputs)
        if value.dtype in (torch.int32, torch.int64) and value.ndim in (2, 3)
    )


def specialize_decode(module, inputs, sliding_layers=()):
    """Share calibrated weights and grids while specializing AR128 to AR1."""
    result = torch.fx.GraphModule(module, copy.deepcopy(module.graph))
    remove_export_checks(result)
    ar = inputs[0].shape[1]
    capacity = inputs[1].shape[-1]
    position_index = position_input_index(inputs)
    caches = inputs[position_index + 1 :]
    layers = len(caches) // 2
    values = [inputs[0][:, :1].clone()]
    for index, mask in enumerate(inputs[1:position_index]):
        width = mask.shape[-1] if index == 0 else mask.shape[-1] - ar + 1
        values.append(torch.zeros(1, 1, 1, width, dtype=mask.dtype))
    values.append(inputs[position_index][..., :1].clone())
    for index, value in enumerate(caches):
        shape = list(value.shape)
        layer = index % layers
        full_attention = caches[layer].ndim == 4 and layer not in sliding_layers
        if full_attention and shape[-1] == capacity - ar:
            shape[-1] = capacity - 1
        elif full_attention and shape[-2] == capacity - ar:
            shape[-2] = capacity - 1
        values.append(torch.zeros(shape, dtype=value.dtype))
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    tensors = dict(result.named_buffers()) | dict(result.named_parameters())
    env, arguments = {}, iter(values)
    with mode:
        for node in result.graph.nodes:
            if node.op == "placeholder":
                value = mode.from_tensor(next(arguments))
            elif node.op == "get_attr":
                value = mode.from_tensor(tensors[node.target])
            elif node.op == "output":
                value = torch.fx.map_arg(node.args[0], lambda n: env[n])
            else:
                args = list(node.args)
                if node.target in (
                    torch.ops.aten.reshape.default,
                    torch.ops.aten.view.default,
                    torch.ops.aten.expand.default,
                ):
                    shape = list(args[1])
                    old, new = args[0].meta["val"], env[args[0]]
                    if old.numel() == ar * new.numel():
                        if len(shape) <= 2:
                            if shape[0] > 0:
                                assert shape[0] % ar == 0, (node.name, shape)
                                shape[0] //= ar
                        elif ar in shape:
                            shape[shape.index(ar)] = 1
                    else:
                        for mask in inputs[2:position_index]:
                            width, decoded_width = (
                                mask.shape[-1],
                                mask.shape[-1] - ar + 1,
                            )
                            if width not in shape:
                                continue
                            if old.numel() * decoded_width == new.numel() * width:
                                shape[shape.index(width)] = decoded_width
                                break
                            if (
                                ar in shape
                                and old.numel() * decoded_width
                                == new.numel() * width * ar
                            ):
                                shape[shape.index(width)] = decoded_width
                                shape[shape.index(ar)] = 1
                                break
                    args[1] = shape
                elif node.target == torch.ops.aten.slice.Tensor:
                    axis = args[1]
                    old_size, new_size = (
                        args[0].meta["val"].shape[axis],
                        env[args[0]].shape[axis],
                    )
                    if (
                        len(args) > 2
                        and args[2] == -ar
                        and new_size == old_size - ar + 1
                    ):
                        args[2] = -1
                    if len(args) > 3 and args[3] == old_size == ar and new_size == 1:
                        args[3] = 1
                node.args = tuple(args)
                value = node.target(
                    *torch.fx.map_arg(node.args, lambda n: env[n]),
                    **torch.fx.map_arg(node.kwargs, lambda n: env[n]),
                )
            env[node] = value
    for node, value in env.items():
        node.meta["val"] = value
    result.graph.lint()
    result.recompile()
    return result, tuple(values)


def replace_prefill_cores(module):
    core_target = torch.ops.qnn_custom_ops.deltanet_core.default
    cores = [n for n in module.graph.nodes if n.target == core_target]
    assert cores, "Expected GDN cores in the prefill graph"
    inputs = tuple(torch.zeros(tuple(n.meta["val"].shape)) for n in cores[0].args)
    replacement = torch.export.export(ChunkGDN().eval(), inputs, strict=False).module()
    remove_export_checks(replacement)
    buffers = dict(replacement.named_buffers())
    for ordinal, core in enumerate(cores):
        env, arguments = {}, iter(core.args)
        with module.graph.inserting_before(core):
            for node in replacement.graph.nodes:
                if node.op == "placeholder":
                    env[node] = next(arguments)
                elif node.op == "get_attr":
                    name = f"gdn_chunk_{ordinal}_{node.target.replace('.', '_')}"
                    module.register_buffer(name, buffers[node.target])
                    env[node] = module.graph.get_attr(name)
                elif node.op == "output":
                    outputs = torch.fx.map_arg(node.args[0], lambda n: env[n])
                else:
                    env[node] = module.graph.node_copy(node, lambda n: env[n])
                    env[node].meta = {}
        for user in list(core.users):
            assert user.target == operator.getitem
            user.replace_all_uses_with(outputs[user.args[1]])
            module.graph.erase_node(user)
        module.graph.erase_node(core)
    module.graph.eliminate_dead_code()
    module.graph.lint()
    module.recompile()
    return len(cores)


def trim_prefill_capacity(module, inputs, capacity, sliding_layers=()):
    """Keep the external cache layout while limiting internal FA computation."""
    graph = module.graph
    original, ar = inputs[1].shape[-1], inputs[0].shape[1]
    keep = capacity - ar
    if capacity == original:
        return
    assert 0 < keep < original - ar
    mask_input = [n for n in graph.nodes if n.op == "placeholder"][1]
    masks = [
        dq
        for q in mask_input.users
        if q.target == Q
        for dq in q.users
        if dq.target == DQ
    ]
    for mask in masks:
        users = list(mask.users)
        encoding = tuple(mask.args[1:])
        with graph.inserting_before(mask.next):
            past = _qdq(
                graph,
                graph.call_function(torch.ops.aten.slice.Tensor, (mask, -1, 0, keep)),
                encoding,
            )
            current = _qdq(
                graph,
                graph.call_function(
                    torch.ops.aten.slice.Tensor, (mask, -1, original - ar, original)
                ),
                encoding,
            )
            reduced = _qdq(
                graph,
                graph.call_function(torch.ops.aten.cat.default, ([past, current], -1)),
                encoding,
            )
        for user in users:
            user.replace_input_with(mask, reduced)
    matmuls = [n for n in graph.nodes if n.target == torch.ops.aten.matmul.default]
    assert matmuls
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    cache_start = position_input_index(inputs) + 1
    layers = (len(inputs) - cache_start - int(inputs[-1].ndim == 1)) // 2
    for node in matmuls:
        cat = node.args[1].args[0].args[0].args[0].args[0]
        assert cat.target == torch.ops.aten.cat.default, cat
        cached, current = cat.args[0]
        transport = cached
        source = _decoded_source(transport)
        consumer_quant = None
        while source.target == DQ:
            consumer_quant = transport.args[0]
            transport = source
            source = _decoded_source(transport)
        assert source.op == "placeholder", source
        if (placeholders.index(source) - cache_start) % layers in sliding_layers:
            continue
        axis = cat.args[1]
        # Slice on the input grid so short buckets only materialize their history.
        with graph.inserting_before(transport.next):
            prefix = _qdq(
                graph,
                graph.call_function(
                    torch.ops.aten.slice.Tensor, (transport, axis, 0, keep)
                ),
                tuple(transport.args[1:]),
            )
        if consumer_quant is None:
            cat.args = ([prefix, current], axis)
        else:
            consumer_quant.replace_input_with(transport, prefix)
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    tensors = dict(module.named_buffers()) | dict(module.named_parameters())
    env, arguments = {}, iter(inputs)
    with mode:
        for node in graph.nodes:
            if node.op == "placeholder":
                value = mode.from_tensor(next(arguments))
            elif node.op == "get_attr":
                value = mode.from_tensor(tensors[node.target])
            elif node.op == "output":
                value = torch.fx.map_arg(node.args[0], lambda n: env[n])
            else:
                args = list(node.args)
                if node.target in (
                    torch.ops.aten.reshape.default,
                    torch.ops.aten.view.default,
                    torch.ops.aten.expand.default,
                ):
                    old, new = args[0].meta.get("val"), env[args[0]]
                    shape = list(args[1])
                    if (
                        old is not None
                        and original in shape
                        and new.numel() * original == old.numel() * capacity
                    ):
                        changed_axis = next(
                            index
                            for index, (before, after) in enumerate(
                                zip(old.shape, new.shape)
                            )
                            if before == original and after == capacity
                        )
                        suffix = math.prod(old.shape[changed_axis + 1 :])
                        axis = next(
                            index
                            for index, size in enumerate(shape)
                            if size == original
                            and math.prod(shape[index + 1 :]) == suffix
                        )
                        shape[axis] = capacity
                        args[1] = shape
                        node.args = tuple(args)
                value = node.target(
                    *torch.fx.map_arg(tuple(args), lambda n: env[n]),
                    **torch.fx.map_arg(node.kwargs, lambda n: env[n]),
                )
            env[node] = value
    for node, value in env.items():
        node.meta["val"] = value
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()


def _qdq(graph, value, encoding):
    quant = graph.call_function(Q, (value, *encoding))
    return graph.call_function(DQ, (quant, *encoding))


def _decoded_source(node):
    assert node.target == DQ and node.args[0].target == Q, node
    return node.args[0].args[0]


def _set_encoding(quant, encoding):
    assert quant.target == Q
    quant.args = (quant.args[0], *encoding)
    for decoded in list(quant.users):
        assert decoded.target == DQ
        decoded.args = (quant, *encoding)


def select_generation_logits(module):
    """Append INT32[1] row selection before the calibrated convolution LM head.

    Only generation graphs should use this transformation. Calibration and
    teacher-forced PPL graphs need all token logits. Cache outputs are retained;
    the caller appends the valid token index to its sample and runtime inputs.
    Refresh tensor metadata before exporting the modified graph.
    """
    graph = module.graph
    graph.set_codegen(torch.fx.graph.CodeGen())
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    output = next(n for n in graph.nodes if n.op == "output")
    vocabulary = output.args[0][0].meta["val"].shape[-1]
    heads = [
        n
        for n in graph.nodes
        if n.target == torch.ops.aten.conv2d.default
        and n.meta["val"].shape[1] == vocabulary
    ]
    assert heads, "Expected a calibrated convolution LM head"
    head = heads[-1]
    original = head.args[0]
    assert original.target == DQ
    with graph.inserting_after(placeholders[-1]):
        index = graph.placeholder("last_token_index")
    index.meta["val"] = torch.zeros(1, dtype=torch.int32)
    with graph.inserting_before(head):
        selected = graph.call_function(
            torch.ops.aten.index_select.default, (original, 3, index)
        )
        selected = _qdq(graph, selected, tuple(original.args[1:]))
    head.args = (selected, *head.args[1:])
    graph.lint()
    module.recompile()


def mask_chunk_prefill_padding(module):
    """Keep GDN/conv states at last_token_index before FP16 chunk replacement."""
    graph = module.graph
    index = next(
        n
        for n in graph.nodes
        if n.op == "placeholder" and n.target == "last_token_index"
    )
    cores = [
        n
        for n in graph.nodes
        if str(n.target) == "qnn_custom_ops.deltanet_core.default"
    ]
    assert cores
    ar = cores[0].args[0].meta["val"].shape[1]
    assert ar > 1
    module.register_buffer("gdn_prefill_positions", torch.arange(ar, dtype=torch.int32))
    with graph.inserting_before(cores[0]):
        positions = graph.get_attr("gdn_prefill_positions")
        active = graph.call_function(torch.ops.aten.le.Tensor, (positions, index))
        active = graph.call_function(
            torch.ops.aten._to_copy.default, (active,), {"dtype": torch.float16}
        )
        active = graph.call_function(
            torch.ops.aten.reshape.default, (active, [1, ar, 1])
        )
        inactive = graph.call_function(torch.ops.aten.rsub.Scalar, (active, 1))
    for core in cores:
        with graph.inserting_before(core):
            decay = graph.call_function(
                torch.ops.aten._to_copy.default,
                (core.args[3],),
                {"dtype": torch.float16},
            )
            beta = graph.call_function(
                torch.ops.aten._to_copy.default,
                (core.args[4],),
                {"dtype": torch.float16},
            )
            decay = graph.call_function(torch.ops.aten.mul.Tensor, (decay, active))
            decay = graph.call_function(torch.ops.aten.add.Tensor, (decay, inactive))
            beta = graph.call_function(torch.ops.aten.mul.Tensor, (beta, active))
        core.args = (*core.args[:3], decay, beta, core.args[5])

    outputs = next(n for n in graph.nodes if n.op == "output").args[0]
    offsets, count = {}, 0
    for value in outputs[1:]:
        shape = value.meta["val"].shape
        if len(shape) != 3 or not 0 < shape[-1] < ar:
            continue
        tail = value
        while tail.target in (
            Q,
            DQ,
            torch.ops.aten.to.dtype,
            torch.ops.aten._to_copy.default,
        ):
            tail = tail.args[0]
        assert tail.target == torch.ops.aten.slice.Tensor, tail
        span = shape[-1]
        axis = int(tail.args[1])
        assert tail.args[0].meta["val"].shape[axis] == ar + span
        with graph.inserting_before(tail):
            if span not in offsets:
                name = f"gdn_conv_offsets_{span}"
                module.register_buffer(
                    name, torch.arange(1, span + 1, dtype=torch.int32)
                )
                offsets[span] = graph.get_attr(name)
            indices = graph.call_function(
                torch.ops.aten.add.Tensor, (offsets[span], index)
            )
            gathered = graph.call_function(
                torch.ops.aten.index_select.default, (tail.args[0], axis, indices)
            )
            gathered.meta = dict(tail.meta)
            # QNN Gather requires matching input/output encodings. The
            # existing cache-output Q/DQ performs any subsequent conversion.
            if tail.args[0].target == DQ:
                gathered = _qdq(graph, gathered, tuple(tail.args[0].args[1:]))
        tail.replace_all_uses_with(gathered)
        graph.erase_node(tail)
        count += 1
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    return {"gdn_cores": len(cores), "conv_states": count}


def optimize_grouped_attention(module):
    """Rewrite calibrated Qwen3.5 Q/DQ attention for unsigned cache transport.

    Returns cache indices mapped to their new input/output encodings, plus
    a change log. The caller must publish these encodings as runner metadata
    and initialize caches using the returned zero points. Both prefill and
    decode methods must use the same encodings. Existing populated caches
    require conversion before reuse.

    Query heads become rows of each KV head. Cache storage uses the calibrated
    MatMul RHS grid, represented as unsigned codes of the same bit width. New
    entries enter that grid before storage; historical entries retain their
    bit width through concatenation and MatMul. Weight codes remain unchanged.
    Run after calibration and cache-output alignment, before QNN lowering.
    Refresh tensor metadata afterwards. Unsupported patterns raise.
    """
    graph = module.graph
    graph.set_codegen(torch.fx.graph.CodeGen())
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    output = next(n for n in graph.nodes if n.op == "output")
    outputs = list(output.args[0])
    cache_count = len(outputs) - 1
    cache_start = len(placeholders) - cache_count
    assert cache_count > 0 and cache_count % 2 == 0
    assert cache_start in (3, 4), "Run before adding the selected-logit index input"
    changes = []
    cache_encodings = {}
    matmuls = [n for n in graph.nodes if n.target == torch.ops.aten.matmul.default]
    assert matmuls, "No Qwen3.5 grouped attention found"
    for matmul in matmuls:
        left, right = matmul.args
        assert left.target == DQ and right.target == DQ
        rhs_quant = right.args[0]
        assert rhs_quant.target == Q
        assert rhs_quant.args[-1] in (torch.uint8, torch.int8, torch.int16)
        repeated = rhs_quant.args[0]
        if repeated.target != DQ:
            repeated = right
        reshape = _decoded_source(repeated)
        assert reshape.target == torch.ops.aten.reshape.default
        expand = _decoded_source(reshape.args[0])
        assert expand.target == torch.ops.aten.expand.default
        unsqueeze = _decoded_source(expand.args[0])
        assert unsqueeze.target == torch.ops.aten.unsqueeze.default
        base = unsqueeze.args[0]
        cat = _decoded_source(base)
        assert cat.target == torch.ops.aten.cat.default
        old_cat_encoding = tuple(base.args[1:])
        for value in (repeated, reshape.args[0], expand.args[0]):
            assert tuple(value.args[1:]) == old_cat_encoding
        cached, new_value = cat.args[0]
        input_quant = cached.args[0]
        cache_placeholder = _decoded_source(cached)
        assert cache_placeholder.op == "placeholder"
        input_index = placeholders.index(cache_placeholder)
        cache_index = input_index - cache_start
        assert 0 <= cache_index < cache_count
        assert cache_index not in cache_encodings
        rhs_encoding = tuple(rhs_quant.args[1:])
        if rhs_encoding[-1] == torch.uint8:
            unsigned = rhs_encoding
        else:
            assert rhs_encoding[1] == 0
            if rhs_encoding[-1] == torch.int8:
                unsigned = (rhs_encoding[0], 128, 0, 255, torch.uint8)
            else:
                unsigned = (rhs_encoding[0], 32768, 0, 65535, torch.int32)
        cache_encodings[cache_index] = unsigned

        # Convert new entries once; the stored history already uses this grid.
        with graph.inserting_before(cat):
            converted_new = _qdq(
                graph, _qdq(graph, new_value, old_cat_encoding), unsigned
            )
        cat.args = ([cached, converted_new], *cat.args[1:])
        _set_encoding(input_quant, unsigned)
        _set_encoding(base.args[0], unsigned)
        with graph.inserting_before(output):
            stored = _qdq(
                graph, _qdq(graph, outputs[cache_index + 1], old_cat_encoding), unsigned
            )
        outputs[cache_index + 1] = stored
        rhs_quant.args = (base, *rhs_quant.args[1:])

        batch, heads, tokens, inner = tuple(left.meta["val"].shape)
        kv_heads = base.meta["val"].shape[1]
        assert heads % kv_heads == 0
        groups = heads // kv_heads
        with graph.inserting_before(matmul):
            grouped = graph.call_function(
                torch.ops.aten.reshape.default,
                (left, [batch, kv_heads, groups * tokens, inner]),
            )
            grouped = _qdq(graph, grouped, tuple(left.args[1:]))
        matmul.args = (grouped, right)

        output_shape = list(matmul.meta["val"].shape)
        quant_users = list(matmul.users)
        assert quant_users and all(n.target == Q for n in quant_users)
        for quant in quant_users:
            for decoded in list(quant.users):
                assert decoded.target == DQ
                users = list(decoded.users)
                with graph.inserting_before(decoded.next):
                    restored = graph.call_function(
                        torch.ops.aten.reshape.default, (decoded, output_shape)
                    )
                    restored = _qdq(graph, restored, tuple(decoded.args[1:]))
                for user in users:
                    user.replace_input_with(decoded, restored)
        changes.append(
            {
                "matmul": matmul.name,
                "cache_input_index": input_index,
                "old_heads": heads,
                "kv_heads": kv_heads,
                "query_groups": groups,
                "calibrated_rhs_encoding": list(rhs_encoding[:4]),
                "cache_encoding": list(unsigned[:4]),
                "cache_storage_bits": 8 if unsigned[-1] == torch.uint8 else 16,
                "attention_rhs_bits": 16 if rhs_encoding[-1] == torch.int16 else 8,
                "old_cat_encoding": list(old_cat_encoding[:4]),
            }
        )
    output.args = (tuple(outputs),)
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    assert len(cache_encodings) == len(matmuls)
    return cache_encodings, changes
