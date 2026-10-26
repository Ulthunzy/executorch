"""AR128 fronts whose padding does not advance recurrent or convolution state."""

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from .graph_utils import resize_attention


def padded_prefill_front(module, inputs, capacity):
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
    mask_chunk_prefill_padding(module)
    for core in list(graph.nodes):
        if str(core.target) != "qnn_custom_ops.deltanet_core.default":
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
    graph.lint()
    module.recompile()
    FakeTensorProp(module, mode=FakeTensorMode(allow_non_fake_inputs=True)).propagate(
        *inputs
    )
    replace_prefill_cores(module)
    module = torch.export.export(module, inputs, strict=False).module()
    remove_export_checks(module)
    return module, inputs
