"""Calibrated padding for the shared DeltaNet operators."""

import torch

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
