"""AR128 fronts whose padding does not advance recurrent or convolution state."""

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from .graph_utils import resize_attention


def masked_fp32_core(graph, core, original, index):
    from .gdn_precision import float32_boundary

    module = graph.owning_module
    length = original[0].meta["val"].shape[1]
    names = {
        key: f"gdn_fp32_{core.name}_{key}"
        for key in ("positions", "zero", "one")
    }
    module.register_buffer(
        names["positions"],
        torch.arange(length, dtype=torch.int32).reshape(1, length, 1),
    )
    module.register_buffer(names["zero"], torch.zeros(1))
    module.register_buffer(names["one"], torch.ones(1))
    with graph.inserting_before(core):
        constants = {key: graph.get_attr(name) for key, name in names.items()}
        active = graph.call_function(
            torch.ops.aten.le.Tensor, (constants["positions"], index)
        )
        wide = graph.call_function(torch.ops.aten.unsqueeze.default, (active, -1))
        values = [
            graph.call_function(
                float32_boundary,
                (value,),
            )
            for value in original
        ]
        masked = [
            graph.call_function(
                torch.ops.aten.where.self,
                (
                    wide if ordinal < 3 else active,
                    value,
                    constants["one" if ordinal == 3 else "zero"],
                ),
            )
            for ordinal, value in enumerate(values[:5])
        ]
    core.args = (*masked, values[5])
    # Keep the original output QDQ on a separate conversion, so QNN selects
    # floating-point custom-op outputs rather than folding it into the core.
    for output in list(core.users):
        with graph.inserting_after(output):
            restored = graph.call_function(
                float32_boundary,
                (output,),
            )
        output.replace_all_uses_with(restored)
        restored.args = (output,)


def masked_u16_core(graph, core, original, index):
    quantize = torch.ops.quantized_decomposed.quantize_per_tensor.default
    dequantize = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    if any(
        value.target != dequantize or value.args[3:] != (0, 65535, torch.int32)
        for value in original
    ):
        raise ValueError("custom_u16 requires six calibrated UINT16 core inputs")
    outputs = {user.args[1]: next(iter(user.users)) for user in core.users}
    if set(outputs) != {0, 1} or any(
        value.target != quantize for value in outputs.values()
    ):
        raise ValueError("custom_u16 requires calibrated output grids")
    if outputs[1].args[1:] != original[5].args[1:]:
        raise ValueError("custom_u16 requires matching recurrent state grids")
    module = graph.owning_module
    length = original[0].meta["val"].shape[1]
    names = {
        name: f"gdn_u16_{core.name}_{name}"
        for name in ("positions", "wide", "zero", "one")
    }
    module.register_buffer(
        names["positions"],
        torch.arange(length, dtype=torch.int32).reshape(1, length, 1),
    )
    module.register_buffer(
        names["wide"], torch.arange(length, dtype=torch.int32).reshape(1, length, 1, 1)
    )
    module.register_buffer(names["zero"], torch.zeros(1))
    module.register_buffer(names["one"], torch.ones(1))
    with graph.inserting_before(core):
        constants = {name: graph.get_attr(target) for name, target in names.items()}
        active = graph.call_function(
            torch.ops.aten.le.Tensor, (constants["positions"], index)
        )
        wide = graph.call_function(torch.ops.aten.le.Tensor, (constants["wide"], index))
        masked = []
        for ordinal, value in enumerate(original[:5]):
            params = value.args[1:]
            fill = constants["zero"]
            if ordinal == 3:
                params = (1 / 65535, 0, 0, 65535, torch.int32)
                codes = graph.call_function(quantize, (value, *params))
                value = graph.call_function(dequantize, (codes, *params))
                fill = constants["one"]
            fill_codes = graph.call_function(quantize, (fill, *params))
            fill = graph.call_function(dequantize, (fill_codes, *params))
            selected = graph.call_function(
                torch.ops.aten.where.self,
                (wide if ordinal < 3 else active, value, fill),
            )
            codes = graph.call_function(quantize, (selected, *params))
            masked.append(graph.call_function(dequantize, (codes, *params)))
    core.args = (*masked, original[5])


def padded_prefill_front(module, inputs, capacity, gdn_core="chunked_mixed_fp16"):
    if gdn_core not in (
        "chunked_mixed_fp16", "custom_fp16", "custom_fp32", "custom_u16"
    ):
        raise ValueError("Unsupported prefill GDN implementation")
    from executorch.examples.qualcomm.oss_scripts.llama.quantize.generation import (
        mask_chunk_prefill_padding,
        remove_export_checks,
        replace_prefill_cores,
    )

    remove_export_checks(module)
    if len(inputs) == 6:
        # Keep the complete logical history plus a physical AR128 append area.
        # A partial final block can then fit without dropping historical KV.
        inputs = resize_attention(module, capacity + 128)
        if capacity > 4096:
            from .fp16_attention_core import fp16_attention_core

            fp16_attention_core(module, inputs)
        elif capacity + 128 <= 4096:
            # Sequence tiling would introduce unquantized partial MatMuls.
            from .split_attention_core import split_attention_core

            split_attention_core(module, inputs)
        return module, inputs
    if len(inputs) != 3 or inputs[0].shape[1] != 128:
        raise ValueError("Expected an AR128 GDN or full-attention front")

    graph = module.graph
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    with graph.inserting_after(placeholders[-1]):
        index = graph.placeholder("last_token_index")
    index.meta["val"] = torch.tensor([127], dtype=torch.int32)
    inputs = (*inputs, index.meta["val"])
    originals = {
        node: tuple(node.args)
        for node in graph.nodes
        if str(node.target) == "qnn_custom_ops.deltanet_core.default"
    }
    mask_chunk_prefill_padding(module)
    for core in list(graph.nodes):
        if str(core.target) != "qnn_custom_ops.deltanet_core.default":
            continue
        if gdn_core == "custom_u16":
            masked_u16_core(graph, core, originals[core], index)
            continue
        if gdn_core == "custom_fp32":
            masked_fp32_core(graph, core, originals[core], index)
            continue
        active = core.args[4].args[1]
        with graph.inserting_before(core):
            active = graph.call_function(torch.ops.aten.unsqueeze.default, (active, -1))
            masked = []
            for value in core.args[:3]:
                half = graph.call_function(
                    torch.ops.aten._to_copy.default, (value,), {"dtype": torch.float16}
                )
                masked.append(
                    graph.call_function(torch.ops.aten.mul.Tensor, (half, active))
                )
        core.args = (*masked, *core.args[3:])
        if gdn_core == "custom_fp16":
            with graph.inserting_before(core):
                state = graph.call_function(
                    torch.ops.aten._to_copy.default,
                    (core.args[5],),
                    {"dtype": torch.float16},
                )
            core.args = (*core.args[:5], state)
            for output in list(core.users):
                with graph.inserting_after(output):
                    restored = graph.call_function(
                        torch.ops.aten._to_copy.default,
                        (output,),
                        {"dtype": torch.float32},
                    )
                output.replace_all_uses_with(restored)
                restored.args = (output,)
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    FakeTensorProp(module, mode=FakeTensorMode(allow_non_fake_inputs=True)).propagate(
        *inputs
    )
    if gdn_core == "chunked_mixed_fp16":
        replace_prefill_cores(module)
    module = torch.export.export(module, inputs, strict=False).module()
    remove_export_checks(module)
    return module, inputs
